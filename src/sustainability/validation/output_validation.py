import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def build_topic_criteria_len(topic_and_assessment_criteria: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        t.get("Topic", ""): len(t.get("Assessment Criteria", []))
        for t in topic_and_assessment_criteria
        if isinstance(t, dict) and isinstance(t.get("Topic"), str)
    }


def validate_sroi_payload(payload: Any, topic_criteria_len: Dict[str, int]) -> Tuple[bool, str]:
    if not isinstance(payload, list):
        return False, "Payload is not a list."
    if not payload:
        return False, "Payload list is empty."

    expected_topics = set(topic_criteria_len.keys())
    seen_topics = set()

    for item in payload:
        if not isinstance(item, dict):
            return False, "At least one topic item is not an object."

        topic = item.get("Topic")
        if not isinstance(topic, str) or topic not in topic_criteria_len:
            return False, f"Unexpected or missing topic: {topic!r}"
        seen_topics.add(topic)

        ratings = item.get("ratings")
        conf = item.get("confidence_scores")
        if not isinstance(ratings, list) or not ratings:
            return False, f"Topic {topic!r} missing or empty ratings."
        if not isinstance(conf, list) or not conf:
            return False, f"Topic {topic!r} missing or empty confidence_scores."
        if len(ratings) != len(conf):
            return False, f"Topic {topic!r} ratings/confidence length mismatch."
        if len(ratings) != topic_criteria_len[topic]:
            return False, f"Topic {topic!r} ratings length != criteria length."

    if seen_topics != expected_topics:
        missing = sorted(expected_topics - seen_topics)
        extra = sorted(seen_topics - expected_topics)
        return False, f"Topic coverage mismatch. Missing={missing}, Extra={extra}"

    return True, "ok"


def validate_rag_topic_result(topic_result: Any, topic_criteria_len: Dict[str, int]) -> Tuple[bool, str]:
    if not isinstance(topic_result, dict):
        return False, "Topic result is not an object."

    topic = topic_result.get("Topic")
    if not isinstance(topic, str) or topic not in topic_criteria_len:
        return False, f"Unexpected or missing topic: {topic!r}"

    if not isinstance(topic_result.get("stage1_evidence"), dict):
        return False, f"Topic {topic!r} missing stage1_evidence object."

    stage2 = topic_result.get("stage2_scoring")
    if not isinstance(stage2, dict):
        return False, f"Topic {topic!r} missing stage2_scoring object."

    ratings = stage2.get("ratings")
    conf = stage2.get("confidence_scores")
    if not isinstance(ratings, list) or not ratings:
        return False, f"Topic {topic!r} missing or empty stage2 ratings."
    if not isinstance(conf, list) or not conf:
        return False, f"Topic {topic!r} missing or empty stage2 confidence_scores."
    if len(ratings) != len(conf):
        return False, f"Topic {topic!r} stage2 ratings/confidence length mismatch."
    if len(ratings) != topic_criteria_len[topic]:
        return False, f"Topic {topic!r} stage2 ratings length != criteria length."

    return True, "ok"


def validate_rag_payload(payload: Any, topic_criteria_len: Dict[str, int]) -> Tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "RAG payload is not an object."
    topics = payload.get("topics")
    if not isinstance(topics, list) or not topics:
        return False, "RAG payload topics missing or empty."

    expected_topics = set(topic_criteria_len.keys())
    seen_topics = set()
    for topic_result in topics:
        ok, msg = validate_rag_topic_result(topic_result, topic_criteria_len)
        if not ok:
            return False, msg
        seen_topics.add(topic_result["Topic"])

    if seen_topics != expected_topics:
        missing = sorted(expected_topics - seen_topics)
        extra = sorted(seen_topics - expected_topics)
        return False, f"Topic coverage mismatch. Missing={missing}, Extra={extra}"

    return True, "ok"


def atomic_write_json(path: Path, payload: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)
