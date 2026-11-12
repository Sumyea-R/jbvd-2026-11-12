PROMPTS_RAG_QUESTION = {
    "score_topic_from_retrieved_evidence": """
Your task is to evaluate how well a startup addresses a sustainability topic based only on retrieved criterion-level evidence.

Topic:
{topic}

Assessment criteria:
{criteria_json}

Retrieved evidence per criterion:
{criterion_retrieval_json}

Scoring rules:
- Based on the evidence, give a rating from 1 to 5 for each criterion:
  - 1 = Not addressed at all
  - 2 = Minimally addressed
  - 3 = Somewhat addressed
  - 4 = Well addressed
  - 5 = Fully addressed
- For each rating, provide a confidence score from 1 to 5 based on how confident you are in that rating:
  - 1 = Very low confidence
  - 2 = Low confidence
  - 3 = Moderate confidence
  - 4 = High confidence
  - 5 = Very high confidence
- Use only the provided evidence.
- Lower confidence when retrieval is weak, sparse, or indirect.
- List any criteria for which evidence is missing.
- Write a short narrative grounded only in the evidence.

Return STRICTLY as JSON:
```json
{{
  "Topic": "",
  "ratings": [1, 4, 3, ...],
  "confidence_scores": [5, 3, 4, ...],
  "evidence": ["quoted statements from the plan..."],
  "missing_criteria": ["criterion_2", "criterion_4"],
  "narrative": "..."
}}
```
""",
}
