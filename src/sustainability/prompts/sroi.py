PROMPTS = {
    "sroi": """
Your task is to evaluate how well the following business plan focuses on sustainability.
You will assess it based on several topics, each with specific assessment criteria.

Below are the topics and their criteria:
{topic_and_assessment_criteria}

Follow these steps carefully:

1. For each topic, find text or quotes in the business plan that show evidence related to each sustainability criterion.
2. Based on the evidence, give a rating from 1 to 5 for each criterion:
   - 1 = Not addressed at all
   - 2 = Minimally addressed
   - 3 = Somewhat addressed
   - 4 = Well addressed
   - 5 = Fully addressed
3. Write a short explanation (narrative) summarizing your reasoning for the ratings.
4. List any criteria that are missing or not mentioned in the business plan.

Return your answer **strictly** in the following JSON format:

```json
{{
  "Topic": "",
  "ratings": [1, 4, 3, ...],
  "evidence": ["quoted statements from the plan..."],
  "missing_criteria": ["criterion_2", "criterion_4"],
  "narrative": "..."
}}
```

Now read this business plan and answer based on it:
{business_plan}
""",
    "sroi_with_confidence": """
Your task is to evaluate how well the following business plan focuses on sustainability.
You will assess it based on several topics, each with specific assessment criteria.

Below are the topics and their criteria:
{topic_and_assessment_criteria}

Follow these steps carefully:

1. For each topic, find text or quotes in the business plan that show evidence related to each sustainability criterion.
2. Based on the evidence, give a rating from 1 to 5 for each criterion:
   - 1 = Not addressed at all
   - 2 = Minimally addressed
   - 3 = Somewhat addressed
   - 4 = Well addressed
   - 5 = Fully addressed
3. For each rating, provide a confidence score from 1 to 5 based on how confident you are in that rating:
   - 1 = Very low confidence
   - 2 = Low confidence
   - 3 = Moderate confidence
   - 4 = High confidence
   - 5 = Very high confidence
4. Write a short explanation (narrative) summarizing your reasoning for the ratings.
5. List any criteria that are missing or not mentioned in the business plan.

Return your answer **strictly** in the following JSON format:

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

Now read this business plan and answer based on it:
{business_plan}
""",
}
