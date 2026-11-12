import argparse
import glob
import json
import os
import re
import time
from collections import defaultdict
from math import log
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openai
from openai import OpenAI

from ..prompts.sroi_rag_question import PROMPTS_RAG_QUESTION
from ..utils.paths import find_repo_root, get_outputs_root, get_questions_path, get_results_root
from ..utils.run_log import utc_timestamp, write_run_manifest
from ..validation.output_validation import (
    atomic_write_json,
    build_topic_criteria_len,
    validate_rag_payload,
    validate_rag_topic_result,
)


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "was", "were",
    "will", "with", "this", "these", "those", "their", "them", "than", "then", "into",
    "about", "also", "such", "can", "could", "should", "would", "may", "might",
}


def load_retrieval_dependencies() -> Tuple[Any, Any]:
    """Load local retrieval dependencies lazily so the error is actionable at runtime."""
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Hybrid retrieval requires numpy. Run this pipeline from the project .venv."
        ) from exc

    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Hybrid retrieval requires sentence-transformers. Install it in the project .venv "
            "and run this pipeline with .venv/bin/python."
        ) from exc

    return np, SentenceTransformer


def load_embedding_model(model_name: str) -> Any:
    """Load the local embedding model once per run and reuse it across files."""
    _, sentence_transformer_cls = load_retrieval_dependencies()
    return sentence_transformer_cls(model_name)


class LocalBM25Index:
    """Small local BM25 implementation to avoid adding another runtime package."""

    def __init__(self, tokenized_docs: List[List[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.tokenized_docs = tokenized_docs
        self.k1 = k1
        self.b = b
        self.doc_count = len(tokenized_docs)
        self.doc_lengths = [len(doc) for doc in tokenized_docs]
        self.avgdl = (sum(self.doc_lengths) / self.doc_count) if self.doc_count else 0.0
        self.term_freqs: List[Dict[str, int]] = []
        doc_freqs: Dict[str, int] = defaultdict(int)

        for doc in tokenized_docs:
            tf: Dict[str, int] = {}
            for token in doc:
                tf[token] = tf.get(token, 0) + 1
            self.term_freqs.append(tf)
            for token in tf:
                doc_freqs[token] += 1

        self.idf = {
            token: log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))
            for token, df in doc_freqs.items()
        }

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        if not self.doc_count or not query_tokens:
            return [0.0] * self.doc_count

        unique_query_tokens = list(dict.fromkeys(query_tokens))
        scores: List[float] = []
        avgdl = self.avgdl or 1.0

        for doc_idx, tf in enumerate(self.term_freqs):
            score = 0.0
            doc_len = self.doc_lengths[doc_idx] or 1
            norm = self.k1 * (1.0 - self.b + self.b * doc_len / avgdl)
            for token in unique_query_tokens:
                freq = tf.get(token, 0)
                if not freq:
                    continue
                idf = self.idf.get(token, 0.0)
                score += idf * (freq * (self.k1 + 1.0)) / (freq + norm)
            scores.append(score)

        return scores


def slugify_model_name(model: str) -> str:
    """Convert provider/model ids into safe folder/log name fragments."""
    return re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_").lower()


def parse_startup_and_version(stem: str) -> Optional[Tuple[str, int]]:
    match = re.match(r"^(?P<startup>.+)_V(?P<version>\d+)$", stem)
    if not match:
        return None
    return match.group("startup"), int(match.group("version"))


def select_latest_version_files(bp_files: List[str]) -> List[str]:
    """Choose the highest versioned plan per startup; keep unmatched names unchanged."""
    grouped: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    passthrough: List[str] = []

    for bp_file in bp_files:
        parsed = parse_startup_and_version(Path(bp_file).stem)
        if parsed is None:
            passthrough.append(bp_file)
            continue
        startup, version = parsed
        grouped[startup].append((version, bp_file))

    selected = list(passthrough)
    for versioned in grouped.values():
        versioned.sort(key=lambda item: item[0])
        selected.append(versioned[-1][1])
    return sorted(selected, key=lambda path: Path(path).name.lower())


def resolve_input_files(
    input_files: Optional[List[str]],
    repo_root: Path,
    data_folder: Path,
) -> List[str]:
    if not input_files:
        return sorted(glob.glob(str(data_folder / "*.md")))

    resolved_files: List[str] = []
    seen: set[str] = set()
    for raw_path in input_files:
        candidate = Path(raw_path).expanduser()
        candidates = [candidate] if candidate.is_absolute() else [Path.cwd() / candidate, repo_root / candidate]

        resolved_path: Optional[Path] = None
        for option in candidates:
            option = option.resolve()
            if option.exists():
                resolved_path = option
                break
        if resolved_path is None:
            raise FileNotFoundError(f"Input markdown file not found: {raw_path}")
        if resolved_path.suffix.lower() != ".md":
            raise ValueError(f"Input file must be markdown (.md): {resolved_path}")

        resolved_str = str(resolved_path)
        if resolved_str not in seen:
            resolved_files.append(resolved_str)
            seen.add(resolved_str)

    return resolved_files


def sleep_if_needed(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def tokenize(text: str) -> List[str]:
    """Lowercase + alphanumeric tokenization with light stopword filtering."""
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def parse_markdown_paragraphs(text: str) -> List[Dict[str, str]]:
    """Split markdown into paragraph blocks while tracking the most recent heading."""
    current_heading = "Document"
    current_lines: List[str] = []
    paragraphs: List[Dict[str, str]] = []

    def flush_current() -> None:
        nonlocal current_lines
        paragraph = " ".join(line.strip() for line in current_lines).strip()
        if paragraph:
            paragraphs.append({"heading": current_heading, "text": paragraph})
        current_lines = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if re.match(r"^\s*#{1,6}\s+", line):
            flush_current()
            current_heading = re.sub(r"^\s*#{1,6}\s+", "", line).strip() or current_heading
            continue
        if not line.strip():
            flush_current()
            continue
        current_lines.append(line)

    flush_current()
    return paragraphs


def build_contextual_chunks(
    text: str,
    target_chars: int = 1400,
    max_chars: int = 1800,
    overlap_paragraphs: int = 1,
) -> List[Dict[str, Any]]:
    """Create paragraph-aware chunks that preserve local context and section headings."""
    paragraphs = parse_markdown_paragraphs(text)
    if not paragraphs:
        return []

    chunks: List[Dict[str, Any]] = []
    i = 0
    chunk_index = 0
    while i < len(paragraphs):
        chunk_paragraphs: List[Dict[str, str]] = []
        chunk_len = 0
        j = i
        while j < len(paragraphs):
            para = paragraphs[j]
            piece = f"[{para['heading']}] {para['text']}"
            piece_len = len(piece)
            if chunk_paragraphs and (chunk_len + piece_len > max_chars):
                break
            chunk_paragraphs.append(para)
            chunk_len += piece_len + 2
            j += 1
            if chunk_len >= target_chars:
                break

        if not chunk_paragraphs:
            chunk_paragraphs.append(paragraphs[i])
            j = i + 1

        chunk_text = "\n\n".join(f"[{para['heading']}] {para['text']}" for para in chunk_paragraphs)
        chunks.append(
            {
                "index": chunk_index,
                "heading": chunk_paragraphs[0]["heading"],
                "text": chunk_text,
            }
        )
        chunk_index += 1

        next_i = max(i + 1, j - overlap_paragraphs) if j < len(paragraphs) else j
        i = next_i

    return chunks


class HybridRetrievalIndex:
    """Fuse local BM25 and dense embedding retrieval for criterion-level evidence search."""

    def __init__(self, chunks: List[Dict[str, Any]], embedding_model_name: str, embedding_model: Any) -> None:
        np, _ = load_retrieval_dependencies()
        self.np = np
        self.chunks = chunks
        self.embedding_model_name = embedding_model_name
        self.embedding_model = embedding_model
        self.chunk_texts = [
            f"[{chunk.get('heading', 'Document')}] {chunk['text']}" for chunk in chunks
        ]
        self.chunk_tokens = [tokenize(text) for text in self.chunk_texts]
        self.bm25 = LocalBM25Index(self.chunk_tokens)
        if self.chunk_texts:
            self.chunk_embeddings = np.asarray(
                self.embedding_model.encode(
                    self.chunk_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
            self.chunk_embeddings = np.nan_to_num(
                self.chunk_embeddings,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        else:
            self.chunk_embeddings = np.zeros((0, 0), dtype=float)
        self.query_cache: Dict[str, Dict[str, Any]] = {}

    def _get_scores_for_query(self, query: str) -> Dict[str, Any]:
        cached = self.query_cache.get(query)
        if cached is not None:
            return cached

        query_tokens = tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        if self.chunk_texts:
            query_embedding = self.np.asarray(
                self.embedding_model.encode(
                    [query],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )[0],
                dtype=self.np.float32,
            )
            query_embedding = self.np.nan_to_num(
                query_embedding,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            with self.np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                dense_values = (
                    self.chunk_embeddings.astype(self.np.float64, copy=False)
                    @ query_embedding.astype(self.np.float64, copy=False)
                )
            dense_scores = self.np.nan_to_num(
                dense_values,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).tolist()
        else:
            dense_scores = []

        result = {
            "query": query,
            "query_tokens": query_tokens,
            "bm25_scores": bm25_scores,
            "dense_scores": dense_scores,
        }
        self.query_cache[query] = result
        return result

    def _build_fusion_result(self, query: str, top_k: int, rrf_k: int = 60) -> Dict[str, Any]:
        scores = self._get_scores_for_query(query)
        bm25_scores = scores["bm25_scores"]
        dense_scores = scores["dense_scores"]

        sparse_ranked = sorted(
            [idx for idx, score in enumerate(bm25_scores) if score > 0],
            key=lambda idx: bm25_scores[idx],
            reverse=True,
        )
        dense_ranked = sorted(
            [idx for idx, score in enumerate(dense_scores) if score > 0],
            key=lambda idx: dense_scores[idx],
            reverse=True,
        )

        rank_window = max(top_k * 4, 12)
        fused_scores: Dict[int, float] = defaultdict(float)
        sparse_ranks: Dict[int, int] = {}
        dense_ranks: Dict[int, int] = {}

        for rank, idx in enumerate(sparse_ranked[:rank_window], start=1):
            sparse_ranks[idx] = rank
            fused_scores[idx] += 1.0 / (rrf_k + rank)

        for rank, idx in enumerate(dense_ranked[:rank_window], start=1):
            dense_ranks[idx] = rank
            fused_scores[idx] += 1.0 / (rrf_k + rank)

        if fused_scores:
            ranked_indices = sorted(
                fused_scores,
                key=lambda idx: (fused_scores[idx], dense_scores[idx], bm25_scores[idx]),
                reverse=True,
            )
        elif dense_ranked:
            ranked_indices = dense_ranked[:top_k]
        elif sparse_ranked:
            ranked_indices = sparse_ranked[:top_k]
        else:
            ranked_indices = list(range(min(top_k, len(self.chunks))))

        base_indices = ranked_indices[:top_k]
        query_token_set = set(scores["query_tokens"])
        if query_token_set and base_indices:
            covered_tokens = set()
            for idx in base_indices:
                covered_tokens.update(query_token_set.intersection(self.chunk_tokens[idx]))
            coverage_ratio = len(covered_tokens) / len(query_token_set)
        else:
            coverage_ratio = 0.0

        bm25_top = max((bm25_scores[idx] for idx in base_indices), default=0.0)
        dense_top = max((dense_scores[idx] for idx in base_indices), default=0.0)

        return {
            "query": query,
            "query_tokens": scores["query_tokens"],
            "base_indices": base_indices,
            "bm25_scores": bm25_scores,
            "dense_scores": dense_scores,
            "fused_scores": dict(fused_scores),
            "sparse_ranks": sparse_ranks,
            "dense_ranks": dense_ranks,
            "coverage_ratio": coverage_ratio,
            "bm25_top": bm25_top,
            "dense_top": dense_top,
        }

def retrieve_question_chunks(
    retriever: HybridRetrievalIndex,
    topic: str,
    criterion: str,
    top_k: int = 4,
    neighbor_window: int = 1,
    weak_coverage_threshold: float = 0.18,
    weak_dense_threshold: float = 0.25,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Retrieve evidence per criterion using BM25 + embeddings and save component diagnostics."""
    primary_query = f"{topic}. {criterion}"
    primary = retriever._build_fusion_result(primary_query, top_k=top_k)
    fallback_used = False
    effective = primary

    if (
        not primary["base_indices"]
        or (
            primary["coverage_ratio"] < weak_coverage_threshold
            and primary["dense_top"] < weak_dense_threshold
            and primary["bm25_top"] <= 0
        )
    ):
        fallback = retriever._build_fusion_result(topic, top_k=top_k)
        if fallback["base_indices"]:
            effective = fallback
            fallback_used = True

    expanded_indices = set()
    for idx in effective["base_indices"]:
        for neighbor_idx in range(
            max(0, idx - neighbor_window),
            min(len(retriever.chunks), idx + neighbor_window + 1),
        ):
            expanded_indices.add(neighbor_idx)

    selected_indices = sorted(expanded_indices)
    selected_chunks = [retriever.chunks[idx] for idx in selected_indices]

    if not selected_indices:
        retrieval_strength = "none"
    elif effective["coverage_ratio"] >= 0.45 or (
        effective["dense_top"] >= 0.45 and effective["bm25_top"] >= 1.0
    ):
        retrieval_strength = "strong"
    elif effective["coverage_ratio"] >= 0.25 or effective["dense_top"] >= 0.30 or effective["bm25_top"] >= 0.5:
        retrieval_strength = "moderate"
    else:
        retrieval_strength = "weak"

    base_hits = []
    for idx in effective["base_indices"]:
        base_hits.append(
            {
                "chunk_index": idx,
                "heading": retriever.chunks[idx].get("heading", "Document"),
                "bm25_score": round(effective["bm25_scores"][idx], 6),
                "dense_score": round(effective["dense_scores"][idx], 6),
                "fused_score": round(effective["fused_scores"].get(idx, 0.0), 6),
                "sparse_rank": effective["sparse_ranks"].get(idx),
                "dense_rank": effective["dense_ranks"].get(idx),
                "char_length": len(retriever.chunks[idx]["text"]),
            }
        )

    diagnostics = {
        "topic": topic,
        "criterion": criterion,
        "primary_query": primary_query,
        "effective_query": effective["query"],
        "fallback_used": fallback_used,
        "retrieval_backend": {
            "sparse": "local_bm25",
            "dense": f"sentence_transformers:{retriever.embedding_model_name}",
            "fusion": "reciprocal_rank_fusion",
        },
        "retrieval_strength": retrieval_strength,
        "query_token_count": len(effective["query_tokens"]),
        "coverage_ratio": round(effective["coverage_ratio"], 6),
        "bm25_top_score": round(effective["bm25_top"], 6),
        "dense_top_score": round(effective["dense_top"], 6),
        "base_hit_count": len(effective["base_indices"]),
        "selected_chunk_count": len(selected_indices),
        "selected_chunk_indices": selected_indices,
        "base_hits": base_hits,
    }

    return selected_chunks, diagnostics


def extract_json(text: str) -> Dict[str, Any]:
    """Parse JSON from plain text or fenced JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def call_llm_json(client: OpenAI, model: str, prompt: str, instructions: str) -> Dict[str, Any]:
    """Single LLM call with deterministic settings, then strict JSON extraction."""
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=prompt,
        temperature=0,
    )
    return extract_json(response.output_text)


def build_retrieval_config(
    embedding_model_name: str,
    retrieval_top_k: int,
    neighbor_window: int,
    include_scoring_prompt: bool = False,
) -> Dict[str, Any]:
    config = {
        "sparse": "local_bm25",
        "dense": f"sentence_transformers:{embedding_model_name}",
        "fusion": "reciprocal_rank_fusion",
        "chunking": "paragraph_aware_with_heading_context",
        "retrieval_top_k": retrieval_top_k,
        "neighbor_window": neighbor_window,
    }
    if include_scoring_prompt:
        config["scoring_prompt"] = "single_topic_scoring_from_raw_retrieved_evidence"
    return config


def build_retrieval_payload(
    bp_file: str,
    business_plan: str,
    topic_and_assessment_criteria: List[Dict[str, Any]],
    embedding_model_name: str,
    embedding_model: Any,
    retrieval_top_k: int,
    neighbor_window: int,
) -> Dict[str, Any]:
    """Run only local hybrid retrieval and return an inspectable per-criterion payload."""
    chunks = build_contextual_chunks(business_plan)
    retriever = HybridRetrievalIndex(
        chunks,
        embedding_model_name=embedding_model_name,
        embedding_model=embedding_model,
    )
    topic_results: List[Dict[str, Any]] = []

    for topic_block in topic_and_assessment_criteria:
        topic = topic_block.get("Topic", "")
        criteria = topic_block.get("Assessment Criteria", [])
        criterion_retrieval_rows: List[Dict[str, Any]] = []
        criterion_retrieval_diagnostics: List[Dict[str, Any]] = []
        total_retrieved_chunks = 0

        for criterion in criteria:
            retrieved, retrieval_diagnostics = retrieve_question_chunks(
                retriever,
                topic,
                criterion,
                top_k=retrieval_top_k,
                neighbor_window=neighbor_window,
            )
            total_retrieved_chunks += len(retrieved)
            criterion_retrieval_rows.append(
                {
                    "criterion": criterion,
                    "retrieved_chunks": [
                        {
                            "chunk_index": chunk["index"],
                            "heading": chunk.get("heading", "Document"),
                            "text": chunk["text"],
                        }
                        for chunk in retrieved
                    ],
                }
            )
            criterion_retrieval_diagnostics.append(retrieval_diagnostics)

        topic_results.append(
            {
                "Topic": topic,
                "criteria": criteria,
                "retrieved_chunk_count": total_retrieved_chunks,
                "retrieval_diagnostics": criterion_retrieval_diagnostics,
                "stage1_evidence": {"retrieved_evidence": criterion_retrieval_rows},
            }
        )

    return {
        "business_plan_file": os.path.basename(bp_file),
        "method": "question_level_hybrid_retrieval",
        "retrieval_config": build_retrieval_config(
            embedding_model_name,
            retrieval_top_k,
            neighbor_window,
        ),
        "topics": topic_results,
    }


def validate_retrieval_payload(payload: Any, topic_criteria_len: Dict[str, int]) -> Tuple[bool, str]:
    """Validate saved retrieval artifacts before optional LLM scoring."""
    if not isinstance(payload, dict):
        return False, "Retrieval payload is not an object."
    topics = payload.get("topics")
    if not isinstance(topics, list) or not topics:
        return False, "Retrieval payload topics missing or empty."

    expected_topics = set(topic_criteria_len.keys())
    seen_topics = set()
    for topic_result in topics:
        if not isinstance(topic_result, dict):
            return False, "At least one retrieval topic is not an object."
        topic = topic_result.get("Topic")
        if not isinstance(topic, str) or topic not in topic_criteria_len:
            return False, f"Unexpected or missing topic: {topic!r}"
        seen_topics.add(topic)

        criteria = topic_result.get("criteria")
        if not isinstance(criteria, list) or len(criteria) != topic_criteria_len[topic]:
            return False, f"Topic {topic!r} criteria length mismatch."

        stage1 = topic_result.get("stage1_evidence")
        if not isinstance(stage1, dict):
            return False, f"Topic {topic!r} missing stage1_evidence object."
        retrieved_evidence = stage1.get("retrieved_evidence")
        if not isinstance(retrieved_evidence, list) or len(retrieved_evidence) != len(criteria):
            return False, f"Topic {topic!r} retrieved evidence length mismatch."

        for row in retrieved_evidence:
            if not isinstance(row, dict):
                return False, f"Topic {topic!r} has non-object retrieval row."
            if not isinstance(row.get("criterion"), str):
                return False, f"Topic {topic!r} has retrieval row without criterion."
            if not isinstance(row.get("retrieved_chunks"), list):
                return False, f"Topic {topic!r} has retrieval row without retrieved_chunks list."

    if seen_topics != expected_topics:
        missing = sorted(expected_topics - seen_topics)
        extra = sorted(seen_topics - expected_topics)
        return False, f"Topic coverage mismatch. Missing={missing}, Extra={extra}"

    return True, "ok"


def score_retrieval_payload(
    retrieval_payload: Dict[str, Any],
    client: OpenAI,
    model: str,
    topic_criteria_len: Dict[str, int],
    base_name: str,
    topic_delay: float = 2.0,
) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """Call the LLM using previously retrieved chunks and return scored topic results."""
    topic_results: List[Dict[str, Any]] = []

    for retrieval_topic in retrieval_payload["topics"]:
        topic = retrieval_topic.get("Topic", "")
        criteria = retrieval_topic.get("criteria", [])
        stage1_evidence = retrieval_topic.get("stage1_evidence", {})
        criterion_retrieval_rows = stage1_evidence.get("retrieved_evidence", [])

        topic_success = False
        failure_reason = ""
        max_topic_attempts = 3
        for topic_attempt in range(1, max_topic_attempts + 1):
            stage2_prompt = PROMPTS_RAG_QUESTION["score_topic_from_retrieved_evidence"].format(
                topic=topic,
                criteria_json=json.dumps(criteria, ensure_ascii=False, indent=2),
                criterion_retrieval_json=json.dumps(
                    criterion_retrieval_rows,
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            try:
                stage2_result = call_llm_json(
                    client=client,
                    model=model,
                    prompt=stage2_prompt,
                    instructions="You are an expert evaluator in sustainability assessment.",
                )
            except openai.RateLimitError:
                print("Rate limit reached during question-level scoring. Sleeping 60 seconds.")
                time.sleep(60)
                continue
            except Exception as e:
                print(
                    f"Scoring/parsing failed for {base_name} / {topic} "
                    f"(topic attempt {topic_attempt}/{max_topic_attempts}): {e}"
                )
                failure_reason = str(e)
                continue

            topic_result = {
                "Topic": topic,
                "criteria": criteria,
                "retrieved_chunk_count": retrieval_topic.get("retrieved_chunk_count", 0),
                "retrieval_diagnostics": retrieval_topic.get("retrieval_diagnostics", []),
                "stage1_evidence": {"retrieved_evidence": criterion_retrieval_rows},
                "stage2_scoring": stage2_result,
            }
            topic_ok, topic_reason = validate_rag_topic_result(topic_result, topic_criteria_len)
            if not topic_ok:
                print(
                    f"Topic validation failed for {base_name} / {topic} "
                    f"(topic attempt {topic_attempt}/{max_topic_attempts}): {topic_reason}"
                )
                failure_reason = topic_reason
                sleep_if_needed(topic_delay)
                continue

            topic_results.append(topic_result)
            topic_success = True
            sleep_if_needed(topic_delay)
            break

        if not topic_success:
            return None, failure_reason or f"topic_failed:{topic}"

    return topic_results, "ok"


def build_scored_payload(
    retrieval_payload: Dict[str, Any],
    topic_results: List[Dict[str, Any]],
    embedding_model_name: str,
    retrieval_top_k: int,
    neighbor_window: int,
    retrieval_source_file: Optional[str] = None,
) -> Dict[str, Any]:
    retrieval_config = dict(
        retrieval_payload.get("retrieval_config")
        or build_retrieval_config(embedding_model_name, retrieval_top_k, neighbor_window)
    )
    retrieval_config["scoring_prompt"] = "single_topic_scoring_from_raw_retrieved_evidence"
    payload = {
        "business_plan_file": retrieval_payload.get("business_plan_file"),
        "method": "question_level_rag_single_stage_scoring",
        "retrieval_config": retrieval_config,
        "topics": topic_results,
    }
    if retrieval_source_file:
        payload["retrieval_source_file"] = retrieval_source_file
    return payload


def parse_args() -> argparse.Namespace:
    """Separate question-level RAG pipeline with its own folders and selection options."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["full", "retrieve", "score"],
        default="full",
        help=(
            "full runs retrieval and LLM scoring; retrieve saves only hybrid retrieval chunks; "
            "score reads saved retrieval chunks and calls the LLM."
        ),
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-5.2",
        help="OpenRouter model id to use for question-level RAG forecasting.",
    )
    parser.add_argument(
        "--output-subdir",
        default=None,
        help="Subdirectory under private/outputs/ where output JSON files will be written.",
    )
    parser.add_argument(
        "--retrieval-subdir",
        default=None,
        help="Saved retrieval private/outputs/ subdirectory to read when --mode score is used.",
    )
    parser.add_argument(
        "--logs-subdir",
        default=None,
        help="Optional subdirectory under private/results/logs/. Defaults to <model-slug>_question_rag.",
    )
    parser.add_argument(
        "--skip-existing-subdir",
        default=None,
        help="Optional private/outputs/ subdirectory used only to skip files already forecasted earlier.",
    )
    parser.add_argument(
        "--latest-version-only",
        action="store_true",
        help="Keep only the highest _V# file per startup before forecasting.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Local sentence-transformers model used for dense retrieval.",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=5,
        help="Number of fused base hits to keep before neighbor expansion.",
    )
    parser.add_argument(
        "--neighbor-window",
        type=int,
        default=1,
        help="How many adjacent chunks to include around each fused hit.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit for number of business plans to process after selection.",
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=None,
        help=(
            "Optional explicit markdown files to process. Paths may be absolute, "
            "relative to the current working directory, or relative to the repo root."
        ),
    )
    parser.add_argument(
        "--topic-delay",
        type=float,
        default=2.0,
        help="Seconds to sleep between topic scoring calls. Defaults to the batch-run delay.",
    )
    return parser.parse_args()


def resolve_output_subdir(outputs_root: Path, subdir: Optional[str]) -> Optional[Path]:
    """Resolve a subdirectory name relative to private/outputs/, or accept an absolute path."""
    if not subdir:
        return None
    candidate = Path(subdir)
    return candidate if candidate.is_absolute() else outputs_root / candidate


def main() -> None:
    """Run retrieval-only, scoring-only, or end-to-end question-level RAG."""
    args = parse_args()
    if args.topic_delay < 0:
        raise ValueError("--topic-delay must be non-negative.")

    repo_root = find_repo_root(Path(__file__).resolve())
    data_folder = repo_root / "private" / "plans"
    outputs_root = get_outputs_root(repo_root)
    model_slug = slugify_model_name(args.model)
    embedding_slug = slugify_model_name(args.embedding_model)
    if args.mode == "retrieve":
        output_subdir = args.output_subdir or f"forecast_question_retrieval_{embedding_slug}"
        logs_subdir = args.logs_subdir or f"{embedding_slug}_question_retrieval"
    else:
        output_subdir = args.output_subdir or f"forecast_question_rag_{model_slug}"
        logs_subdir = args.logs_subdir or f"{model_slug}_question_rag"
    output_folder = resolve_output_subdir(outputs_root, output_subdir)
    skip_existing_folder = resolve_output_subdir(outputs_root, args.skip_existing_subdir)
    retrieval_folder = resolve_output_subdir(outputs_root, args.retrieval_subdir)
    logs_dir = get_results_root(repo_root) / "logs" / logs_subdir

    if args.mode == "score" and retrieval_folder is None:
        raise ValueError("--retrieval-subdir is required when --mode score is used.")

    bp_files = resolve_input_files(args.input_files, repo_root, data_folder)
    if not bp_files:
        source = "--input-files" if args.input_files else str(data_folder)
        print(f"No business plan files found from {source}")
        return
    if args.latest_version_only:
        bp_files = select_latest_version_files(bp_files)
    selected_files = bp_files[: args.max_files] if args.max_files is not None else bp_files

    questions_path = get_questions_path(repo_root, "sustainability_forecasting_questions.json")
    if not questions_path.exists():
        raise FileNotFoundError(f"Missing questions file: {questions_path}")

    with open(questions_path, "r", encoding="utf-8") as f:
        topic_and_assessment_criteria = json.load(f)
    topic_criteria_len = build_topic_criteria_len(topic_and_assessment_criteria)

    client: Optional[OpenAI] = None
    if args.mode in {"full", "score"}:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set in your environment")
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    embedding_model: Optional[Any] = None
    if args.mode in {"full", "retrieve"}:
        embedding_model = load_embedding_model(args.embedding_model)
    output_folder.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "timestamp_utc": utc_timestamp(),
        "pipeline": "forecast_sroi_rag_question",
        "mode": args.mode,
        "model": args.model,
        "embedding_model": args.embedding_model,
        "input_folder": str(data_folder),
        "input_files": bp_files if args.input_files else None,
        "output_folder": str(output_folder),
        "retrieval_folder": str(retrieval_folder) if retrieval_folder else None,
        "logs_folder": str(logs_dir),
        "skip_existing_folder": str(skip_existing_folder) if skip_existing_folder else None,
        "latest_version_only": args.latest_version_only,
        "topic_delay": args.topic_delay,
        "chunking_strategy": "paragraph_aware_with_heading_context",
        "retrieval_strategy": "question_level_local_bm25_plus_embeddings_with_rrf",
        "scoring_strategy": "single_prompt_per_topic" if args.mode in {"full", "score"} else None,
        "total_files": len(selected_files),
        "processed": 0,
        "succeeded": 0,
        "failed": 0,
        "skipped_existing": 0,
        "records": [],
    }

    for bp_file in selected_files:
        run_manifest["processed"] += 1
        base_name = os.path.splitext(os.path.basename(bp_file))[0]
        json_output_path = output_folder / f"{base_name}.json"
        if json_output_path.exists():
            print(f"Output file {json_output_path} already exists. Skipping {bp_file}.")
            run_manifest["skipped_existing"] += 1
            run_manifest["records"].append(
                {"file": os.path.basename(bp_file), "status": "skipped_existing"}
            )
            continue
        if skip_existing_folder is not None:
            prior_output_path = skip_existing_folder / f"{base_name}.json"
            if prior_output_path.exists():
                print(
                    f"Reference output {prior_output_path} already exists. "
                    f"Skipping {bp_file} for incremental rerun."
                )
                run_manifest["skipped_existing"] += 1
                run_manifest["records"].append(
                    {
                        "file": os.path.basename(bp_file),
                        "status": "skipped_existing_reference",
                        "reference_file": str(prior_output_path),
                    }
                )
                continue

        print(f"Processing file: {bp_file}")

        max_file_attempts = 3
        write_success = False
        failure_reason = ""
        retrieval_payload: Optional[Dict[str, Any]] = None

        if args.mode == "score":
            retrieval_input_path = retrieval_folder / f"{base_name}.json"  # type: ignore[operator]
            if not retrieval_input_path.exists():
                failure_reason = f"missing_retrieval_file:{retrieval_input_path}"
                print(f"Missing saved retrieval file for {base_name}: {retrieval_input_path}")
                run_manifest["failed"] += 1
                run_manifest["records"].append(
                    {
                        "file": os.path.basename(bp_file),
                        "status": "failed",
                        "reason": failure_reason,
                    }
                )
                continue
            with open(retrieval_input_path, "r", encoding="utf-8") as f:
                retrieval_payload = json.load(f)
            retrieval_ok, retrieval_reason = validate_retrieval_payload(
                retrieval_payload,
                topic_criteria_len,
            )
            if not retrieval_ok:
                failure_reason = retrieval_reason
                print(f"Saved retrieval validation failed for {base_name}: {retrieval_reason}")
                run_manifest["failed"] += 1
                run_manifest["records"].append(
                    {
                        "file": os.path.basename(bp_file),
                        "status": "failed",
                        "reason": failure_reason,
                        "retrieval_file": str(retrieval_input_path),
                    }
                )
                continue
        else:
            with open(bp_file, "r", encoding="utf-8") as f:
                business_plan = f.read()

        for file_attempt in range(1, max_file_attempts + 1):
            if args.mode in {"full", "retrieve"}:
                retrieval_payload = build_retrieval_payload(
                    bp_file=bp_file,
                    business_plan=business_plan,
                    topic_and_assessment_criteria=topic_and_assessment_criteria,
                    embedding_model_name=args.embedding_model,
                    embedding_model=embedding_model,
                    retrieval_top_k=args.retrieval_top_k,
                    neighbor_window=args.neighbor_window,
                )
                retrieval_ok, retrieval_reason = validate_retrieval_payload(
                    retrieval_payload,
                    topic_criteria_len,
                )
                if not retrieval_ok:
                    print(
                        f"Retrieval validation failed for {base_name} "
                        f"(attempt {file_attempt}/{max_file_attempts}): {retrieval_reason}"
                    )
                    failure_reason = retrieval_reason
                    if file_attempt < max_file_attempts:
                        time.sleep(5 * file_attempt)
                        continue
                    break

            if args.mode == "retrieve":
                atomic_write_json(json_output_path, retrieval_payload)
                write_success = True
                run_manifest["records"].append(
                    {
                        "file": os.path.basename(bp_file),
                        "status": "succeeded",
                        "attempts": file_attempt,
                        "output_kind": "retrieval",
                    }
                )
                break

            if retrieval_payload is None or client is None:
                failure_reason = "missing_retrieval_payload_or_client"
                break

            scored_topics, score_reason = score_retrieval_payload(
                retrieval_payload=retrieval_payload,
                client=client,
                model=args.model,
                topic_criteria_len=topic_criteria_len,
                base_name=base_name,
                topic_delay=args.topic_delay,
            )
            if scored_topics is None:
                failure_reason = score_reason
                print(
                    f"Failed scoring for {base_name} "
                    f"(attempt {file_attempt}/{max_file_attempts}): {score_reason}"
                )
                if file_attempt < max_file_attempts:
                    time.sleep(5 * file_attempt)
                    continue
                break

            retrieval_source_file = None
            if args.mode == "score":
                retrieval_source_file = str(retrieval_folder / f"{base_name}.json")  # type: ignore[operator]
            output_payload = build_scored_payload(
                retrieval_payload=retrieval_payload,
                topic_results=scored_topics,
                embedding_model_name=args.embedding_model,
                retrieval_top_k=args.retrieval_top_k,
                neighbor_window=args.neighbor_window,
                retrieval_source_file=retrieval_source_file,
            )
            is_valid, reason = validate_rag_payload(output_payload, topic_criteria_len)
            if not is_valid:
                print(
                    f"File validation failed for {base_name} "
                    f"(attempt {file_attempt}/{max_file_attempts}): {reason}"
                )
                failure_reason = reason
                if file_attempt < max_file_attempts:
                    time.sleep(5 * file_attempt)
                    continue
                break

            atomic_write_json(json_output_path, output_payload)
            write_success = True
            run_manifest["records"].append(
                {
                    "file": os.path.basename(bp_file),
                    "status": "succeeded",
                    "attempts": file_attempt,
                    "output_kind": "scored_rag",
                }
            )
            break

        if not write_success:
            if args.mode == "retrieve":
                print(
                    f"Failed to generate complete question-level retrieval output for {base_name} "
                    f"after {max_file_attempts} attempts."
                )
            else:
                print(
                    f"Failed to generate complete question-level RAG output for {base_name} "
                    f"after {max_file_attempts} attempts."
                )
            run_manifest["failed"] += 1
            run_manifest["records"].append(
                {
                    "file": os.path.basename(bp_file),
                    "status": "failed",
                    "attempts": max_file_attempts,
                    "reason": failure_reason,
                }
            )
        else:
            run_manifest["succeeded"] += 1

    manifest_path = write_run_manifest(logs_dir, output_subdir, run_manifest)
    print(f"Run manifest written: {manifest_path}")


if __name__ == "__main__":
    main()
