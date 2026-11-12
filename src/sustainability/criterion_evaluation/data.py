"""Read-only extraction and explicit alignment of criterion-level source ratings.

Workbook columns B/C/D hold criterion text, raw score and confidence. Forecasts
without criterion text use the known topic-local 37/35-item layouts: the legacy
layout omits Q013 and Q019, never the final items of those topics.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import re
from typing import Any
import unicodedata

import openpyxl
import pandas as pd


LEGACY_MISSING = {"Q013", "Q019"}
DIAGNOSTIC_COLUMNS = ["source_file", "startup", "config_id", "question_id", "reason", "detail"]
EXPERT_COLUMNS = [
    "startup", "question_id", "expert_id", "expert_name", "rating", "confidence",
    "source_file", "source_sheet", "source_row", "rating_cell", "confidence_cell",
    "source_question_text", "match_method", "raw_rating", "raw_confidence", "status",
]
MODEL_COLUMNS = [
    "startup", "question_id", "config_id", "model_rating", "confidence", "version",
    "source_file", "source_topic", "source_scoring_topic", "source_index", "mapping_method", "raw_rating",
    "raw_confidence", "status",
]
SELECTION_COLUMNS = ["startup", "config_id", "version", "source_file", "available_versions", "status"]


def normalize_text(value: str) -> str:
    """Normalize only typography and whitespace, preserving wording differences."""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    return re.sub(r"\s+", " ", text).strip()


def _startup_key(value: str) -> str:
    return unicodedata.normalize("NFKC", str(value)).strip().casefold()


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def valid_rating(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and 1 <= number <= 5 else None


def _raw(value: Any) -> str:
    return "" if value is None else str(value)


def _diagnose(diagnostics: list[dict], reason: str, detail: str, **context: Any) -> None:
    row = {key: "" for key in DIAGNOSTIC_COLUMNS}
    row.update(context, reason=reason, detail=detail)
    diagnostics.append(row)


def load_catalog(path: Path) -> pd.DataFrame:
    """Assign stable IDs in rubric order; require the specified 37-item rubric."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for block in raw:
        topic = block["Topic"].strip()
        for index, criterion in enumerate(block["Assessment Criteria"], start=1):
            rows.append({
                "question_id": f"Q{len(rows) + 1:03d}", "topic": topic,
                "category": criterion["category"].strip(),
                "question_text": criterion["question"].strip(), "topic_index": index,
            })
    if len(rows) != 37:
        raise ValueError(f"Expected 37 criteria in {path}; found {len(rows)}")
    if len({normalize_text(row["question_text"]) for row in rows}) != 37:
        raise ValueError("Rubric contains duplicate normalized criterion text")
    return pd.DataFrame(rows)


def question_lookup(catalog: pd.DataFrame, aliases_path: Path | None = None) -> dict[str, tuple[str, str]]:
    """Return explicit normalized text -> (question ID, matching method)."""
    lookup = {normalize_text(row.question_text): (row.question_id, "normalized_exact")
              for row in catalog.itertuples()}
    aliases_path = aliases_path or Path(__file__).with_name("question_aliases.json")
    aliases = json.loads(aliases_path.read_text(encoding="utf-8"))["aliases"]
    ids = set(catalog.question_id)
    for alias in aliases:
        qid = alias["question_id"]
        if qid not in ids:
            raise ValueError(f"Alias targets unknown criterion {qid}")
        key = normalize_text(alias["text"])
        if key in lookup and lookup[key][0] != qid:
            raise ValueError(f"Conflicting criterion alias: {alias['text']}")
        lookup.setdefault(key, (qid, "reviewed_alias"))
    return lookup


def _expert_name(path: Path) -> str:
    stem = re.sub(r"(?i)^completed[_ -]+", "", path.stem)
    stem = re.sub(r"(?i)_?score_card.*$", "", stem)
    return re.sub(r"[_-]+", " ", stem).strip()


def load_expert_ratings(experts_dir: Path, catalog: pd.DataFrame,
                        diagnostics: list[dict] | None = None) -> pd.DataFrame:
    """Retain mapped observations including missing and invalid scores."""
    diagnostics = diagnostics if diagnostics is not None else []
    lookup = question_lookup(catalog)
    rows = []
    workbooks = sorted(path for path in Path(experts_dir).glob("*.xlsx") if not path.name.startswith("~$"))
    if not workbooks:
        raise ValueError(f"No expert workbooks found in {experts_dir}")
    for path in workbooks:
        name = _expert_name(path)
        # Keep the workbook filename as identity; display-name parsing can merge
        # distinct people or workbook versions with similar filenames.
        expert_id = path.name
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            for sheet in workbook:
                startup = sheet.title.strip()
                for row_index, cells in enumerate(sheet.iter_rows(min_col=2, max_col=4, values_only=True), start=1):
                    question, raw_rating, raw_confidence = cells
                    if question is None:
                        continue
                    text = str(question).strip()
                    match = lookup.get(normalize_text(text))
                    context = {"source_file": str(path), "startup": startup}
                    if match is None:
                        # Headers and prose without numeric scores are not observations.
                        if _number(raw_rating) is not None:
                            _diagnose(diagnostics, "question_outside_rubric",
                                      f"{sheet.title}!B{row_index}: {text}; rating={raw_rating}", **context)
                        continue
                    qid, method = match
                    rating = valid_rating(raw_rating)
                    status = "available" if rating is not None else (
                        "missing_expert_rating" if raw_rating is None or str(raw_rating).strip() == ""
                        else "invalid_expert_rating")
                    if status != "available":
                        _diagnose(diagnostics, status, f"{sheet.title}!C{row_index}: {raw_rating!r}",
                                  question_id=qid, **context)
                    rows.append({
                        "startup": startup, "question_id": qid, "expert_id": expert_id,
                        "expert_name": name, "rating": rating, "confidence": _number(raw_confidence),
                        "source_file": str(path), "source_sheet": sheet.title, "source_row": row_index,
                        "rating_cell": f"C{row_index}", "confidence_cell": f"D{row_index}",
                        "source_question_text": text, "match_method": method,
                        "raw_rating": _raw(raw_rating), "raw_confidence": _raw(raw_confidence), "status": status,
                    })
        finally:
            workbook.close()
    frame = pd.DataFrame(rows, columns=EXPERT_COLUMNS)
    if frame.empty:
        raise ValueError(f"No expert observations match the rubric in {experts_dir}")
    duplicate_keys = frame.assign(startup_key=frame.startup.map(_startup_key)).duplicated(
        ["startup_key", "question_id", "expert_id"], keep=False)
    for row in frame.loc[duplicate_keys].itertuples():
        _diagnose(diagnostics, "duplicate_expert_observation",
                  f"Repeated expert {row.expert_id}; all observations excluded, including {row.source_sheet}!{row.rating_cell}",
                  source_file=row.source_file, startup=row.startup, question_id=row.question_id)
    frame.loc[duplicate_keys, "rating"] = float("nan")
    frame.loc[duplicate_keys, "status"] = "duplicate_expert_observation"
    return frame


def select_latest_forecasts(input_dir: Path, config_id: str,
                            diagnostics: list[dict] | None = None) -> pd.DataFrame:
    """Select highest version even if malformed; never silently fall back."""
    diagnostics = diagnostics if diagnostics is not None else []
    if not input_dir.is_dir():
        raise ValueError(f"Forecast directory does not exist: {input_dir}")
    groups: dict[str, list[tuple[int, str, Path]]] = defaultdict(list)
    for path in sorted(input_dir.glob("*.json")):
        match = re.fullmatch(r"(.+)_V(\d+)", path.stem)
        if match is None:
            _diagnose(diagnostics, "unrecognized_forecast_filename", "Expected <startup>_V<number>.json",
                      source_file=str(path), config_id=config_id)
            continue
        startup, version = match.group(1).strip(), int(match.group(2))
        groups[_startup_key(startup)].append((version, startup, path))
    rows = []
    for key in sorted(groups):
        available = groups[key]
        highest = max(item[0] for item in available)
        candidates = [item for item in available if item[0] == highest]
        _, startup, _ = candidates[0]
        source_file = ";".join(str(item[2]) for item in candidates)
        status = "selected" if len(candidates) == 1 else "duplicate_latest_version"
        if status != "selected":
            _diagnose(diagnostics, status, "Multiple files identify the same latest startup/version; all excluded",
                      source_file=source_file, startup=startup, config_id=config_id)
        rows.append({"startup": startup, "config_id": config_id, "version": highest,
                     "source_file": source_file, "available_versions": json.dumps(sorted({item[0] for item in available})),
                     "status": status})
    return pd.DataFrame(rows, columns=SELECTION_COLUMNS)


def load_model_ratings(selections: pd.DataFrame, catalog: pd.DataFrame,
                       diagnostics: list[dict] | None = None) -> pd.DataFrame:
    """Read both raw forecast formats and quarantine only ambiguous topics."""
    diagnostics = diagnostics if diagnostics is not None else []
    lookup = question_lookup(catalog)
    topic_ids = {topic: group.question_id.tolist() for topic, group in catalog.groupby("topic", sort=False)}
    topics_by_key = {normalize_text(topic): topic for topic in topic_ids}
    rows = []
    for selection in selections.to_dict("records"):
        context = {key: selection[key] for key in ("source_file", "startup", "config_id")}
        prepared = {row.question_id: {
            **context, "question_id": row.question_id, "model_rating": None, "confidence": None,
            "version": selection["version"], "source_topic": row.topic, "source_scoring_topic": "", "source_index": None,
            "mapping_method": "", "raw_rating": "", "raw_confidence": "", "status": "missing_topic",
        } for row in catalog.itertuples()}
        if selection["status"] != "selected":
            for row in prepared.values():
                row["status"] = selection["status"]
            rows.extend(prepared.values())
            continue
        try:
            payload = json.loads(Path(selection["source_file"]).read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                business_plan_file = payload.get("business_plan_file")
                if business_plan_file and Path(business_plan_file).stem != Path(selection["source_file"]).stem:
                    raise ValueError(f"business_plan_file does not match forecast filename: {business_plan_file}")
                blocks = payload.get("topics")
            else:
                blocks = payload
            if not isinstance(blocks, list) or not all(isinstance(block, dict) for block in blocks):
                raise ValueError("Expected topic list or an object containing a topics list")
        except (OSError, ValueError, TypeError) as error:
            _diagnose(diagnostics, "invalid_forecast", str(error), **context)
            for row in prepared.values():
                row["status"] = "invalid_forecast"
            rows.extend(prepared.values())
            continue
        grouped: dict[str, list[dict]] = defaultdict(list)
        for block in blocks:
            topic = topics_by_key.get(normalize_text(block.get("Topic", "")))
            if topic is None:
                _diagnose(diagnostics, "topic_outside_rubric", str(block.get("Topic", "")), **context)
            else:
                grouped[topic].append(block)
        for topic, expected in topic_ids.items():
            def quarantine(reason: str, detail: str) -> None:
                _diagnose(diagnostics, reason, f"{topic}: {detail}", **context)
                for qid in expected:
                    prepared[qid]["status"] = reason

            candidates = grouped.get(topic, [])
            if not candidates:
                quarantine("missing_topic", "No ratings provided")
                continue
            if len(candidates) != 1:
                quarantine("duplicate_topic", "Multiple topic blocks; all topic ratings excluded")
                continue
            block = candidates[0]
            scoring = block.get("stage2_scoring", block)
            if not isinstance(scoring, dict):
                quarantine("invalid_scoring_block", "Expected a scoring object")
                continue
            mismatched_label = "Topic" in scoring and normalize_text(scoring["Topic"]) != normalize_text(topic)
            ratings = scoring.get("ratings")
            if not isinstance(ratings, list):
                quarantine("malformed_topic_length", "ratings is not an array")
                continue
            legacy = [qid for qid in expected if qid not in LEGACY_MISSING]
            criteria = block.get("criteria")
            if criteria is not None:
                if not isinstance(criteria, list) or len(criteria) != len(ratings):
                    quarantine("malformed_topic_length", "criteria and ratings arrays have different lengths")
                    continue
                mapped = [lookup.get(normalize_text(text), (None, ""))[0] for text in criteria]
                if (None in mapped or len(set(mapped)) != len(mapped)
                        or set(mapped) not in (set(expected), set(legacy))):
                    quarantine("ambiguous_criterion_text", "Criterion texts must uniquely identify the complete current or legacy topic")
                    continue
                mapping_method = "explicit_criterion_text"
            elif len(ratings) == len(expected):
                mapped = expected
                mapping_method = "current_topic_order"
            elif len(legacy) != len(expected) and len(ratings) == len(legacy):
                mapped = legacy
                mapping_method = "legacy35_topic_order"
            else:
                quarantine("malformed_topic_length", f"Found {len(ratings)} ratings; expected {len(expected)}"
                           + (f" or legacy {len(legacy)}" if len(legacy) != len(expected) else ""))
                continue
            if mismatched_label:
                if criteria is None:
                    quarantine("scoring_topic_mismatch", str(scoring["Topic"]))
                    continue
                # The generated scoring label can be a classification answer such
                # as "Economic". Validated outer criterion texts establish the
                # rating identity without relying on this generated label.
                _diagnose(diagnostics, "scoring_topic_label_ignored",
                          f"{topic}: generated label {scoring['Topic']!r}; validated outer criteria used; ratings retained",
                          **context)
            for qid in set(expected) - set(mapped):
                prepared[qid]["status"] = "criterion_absent_from_legacy_rubric"
                prepared[qid]["mapping_method"] = mapping_method
                _diagnose(diagnostics, "criterion_absent_from_legacy_rubric", topic, question_id=qid, **context)
            confidences = scoring.get("confidence_scores")
            if confidences is None:
                confidences = [None] * len(ratings)
            elif not isinstance(confidences, list) or len(confidences) != len(ratings):
                _diagnose(diagnostics, "malformed_confidence_length", f"{topic}: confidence metadata excluded", **context)
                confidences = [None] * len(ratings)
            for index, (qid, raw_rating, raw_confidence) in enumerate(zip(mapped, ratings, confidences), start=1):
                rating = valid_rating(raw_rating)
                status = "available" if rating is not None else "invalid_model_rating"
                if rating is None:
                    _diagnose(diagnostics, status, f"{topic} rating {index}: {raw_rating!r}", question_id=qid, **context)
                prepared[qid].update(model_rating=rating, confidence=_number(raw_confidence),
                                     source_scoring_topic=str(scoring.get("Topic", "")),
                                     source_index=index, mapping_method=mapping_method,
                                     raw_rating=_raw(raw_rating), raw_confidence=_raw(raw_confidence), status=status)
        rows.extend(prepared.values())
    return pd.DataFrame(rows, columns=MODEL_COLUMNS)


def align_records(catalog: pd.DataFrame, experts: pd.DataFrame, models: pd.DataFrame,
                  configurations: list[dict], diagnostics: list[dict] | None = None) -> pd.DataFrame:
    """Create the full startup/criterion/configuration grid with two-rater eligibility."""
    diagnostics = diagnostics if diagnostics is not None else []
    grouped_experts = {(startup, qid): group for (startup, qid), group in
                       experts.groupby(["startup", "question_id"], sort=False)}
    model_lookup = {(row["startup"], row["question_id"], row["config_id"]): row
                    for row in models.to_dict("records")}
    # Assign consistent E1/E2 identities per startup, even when a question is missing.
    expert_ids = {startup: sorted(group.expert_id.unique()) for startup, group in experts.groupby("startup")}
    startups = sorted(set(experts.startup) | set(models.startup), key=_startup_key)
    rows = []
    for startup in startups:
        identities = expert_ids.get(startup, [])
        if len(identities) > 2:
            _diagnose(diagnostics, "ambiguous_expert_count", f"Expected two experts, found {identities}", startup=startup)
        for criterion in catalog.to_dict("records"):
            qid = criterion["question_id"]
            group = grouped_experts.get((startup, qid))
            expert_values: dict[str, Any] = {}
            expert_reasons = []
            if len(identities) > 2:
                expert_reasons.append("ambiguous_expert_count")
            for index in (1, 2):
                identity = identities[index - 1] if index <= len(identities) else ""
                observations = group.loc[group.expert_id == identity] if group is not None else None
                observation = observations.iloc[0] if observations is not None and len(observations) == 1 else None
                score = (observation.rating if observation is not None and len(identities) <= 2
                         and pd.notna(observation.rating) else float("nan"))
                expert_values[f"expert_{index}"] = score
                expert_values[f"expert_{index}_id"] = identity
                for field in ("source_file", "source_sheet", "rating_cell", "confidence", "status"):
                    expert_values[f"expert_{index}_{field}"] = observation[field] if observation is not None else ""
                if observation is None or pd.isna(score):
                    reason = "duplicate_expert_observation" if observations is not None and len(observations) > 1 else (
                        observation.status if observation is not None else "missing_expert_rating")
                    if reason != "available":
                        expert_reasons.append(reason)
            for configuration in configurations:
                model = model_lookup.get((startup, qid, configuration["id"]))
                reasons = list(expert_reasons)
                if model is None:
                    reasons.append("missing_model_forecast")
                elif model["status"] != "available":
                    reasons.append(model["status"])
                model_rating = model["model_rating"] if model else None
                eligible = not reasons and all(pd.notna(expert_values[f"expert_{i}"]) for i in (1, 2)) and pd.notna(model_rating)
                rows.append({
                    "startup": startup, **criterion, "config_id": configuration["id"],
                    "config_label": configuration["label"], **expert_values,
                    "expert_consensus": (expert_values["expert_1"] + expert_values["expert_2"]) / 2,
                    "model_rating": model_rating, "model_confidence": model["confidence"] if model else None,
                    "version": model["version"] if model else None,
                    "model_source_file": model["source_file"] if model else "",
                    "model_status": model["status"] if model else "missing_model_forecast",
                    "eligible": bool(eligible), "exclusion_reason": ";".join(dict.fromkeys(reasons)),
                })
    return pd.DataFrame(rows)


def prepare_data(repo_root: Path, questions_path: Path, experts_dir: Path,
                 config: dict) -> dict[str, pd.DataFrame]:
    """Extract and align inputs entirely in memory; no source files are modified."""
    diagnostics: list[dict] = []
    catalog = load_catalog(questions_path)
    experts = load_expert_ratings(experts_dir, catalog, diagnostics)
    selections = pd.concat([
        select_latest_forecasts(Path(repo_root) / configuration["input_dir"], configuration["id"], diagnostics)
        for configuration in config["configurations"]
    ], ignore_index=True)
    models = load_model_ratings(selections, catalog, diagnostics)
    # Match startup case/Unicode consistently while preferring workbook display names.
    display_names = {}
    for frame in (experts, models, selections):
        for startup in frame.startup:
            display_names.setdefault(_startup_key(startup), startup)
    for frame in (experts, models, selections):
        frame["startup"] = frame.startup.map(lambda value: display_names[_startup_key(value)])
    for diagnostic in diagnostics:
        value = diagnostic["startup"]
        if value:
            diagnostic["startup"] = display_names.get(_startup_key(value), value)
    aligned = align_records(catalog, experts, models, config["configurations"], diagnostics)
    return {
        "catalog": catalog, "expert_ratings": experts, "model_ratings": models,
        "aligned_records": aligned, "diagnostics": pd.DataFrame(diagnostics, columns=DIAGNOSTIC_COLUMNS),
        "selections": selections,
    }
