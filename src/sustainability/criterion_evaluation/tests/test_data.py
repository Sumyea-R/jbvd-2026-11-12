"""Source-format and alignment tests; all venture and expert names are synthetic."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import openpyxl
import pandas as pd

from sustainability.criterion_evaluation.data import (
    LEGACY_MISSING, align_records, load_catalog, load_expert_ratings,
    load_model_ratings, prepare_data, question_lookup, select_latest_forecasts, valid_rating,
)


RUBRIC = Path(__file__).resolve().parents[4] / "src/sustainability/config/sustainability_forecasting_questions_category.json"


class DataPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.catalog = load_catalog(RUBRIC)
        self.experts_dir = self.root / "experts"
        self.experts_dir.mkdir()
        self.forecasts_dir = self.root / "forecasts"
        self.forecasts_dir.mkdir()
        self.configuration = {"id": "test", "label": "Test", "provider": "test",
                              "method": "full_context", "input_dir": "forecasts"}

    def tearDown(self):
        self.temporary.cleanup()

    def forecast(self, legacy=False):
        return [{"Topic": topic,
                 "ratings": [1 + index % 5 for index, qid in enumerate(group.question_id)
                             if not legacy or qid not in LEGACY_MISSING],
                 "confidence_scores": [5 for qid in group.question_id if not legacy or qid not in LEGACY_MISSING]}
                for topic, group in self.catalog.groupby("topic", sort=False)]

    def write_model(self, payload, name="SyntheticVenture_V1.json"):
        path = self.forecasts_dir / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def load_models(self):
        diagnostics = []
        selected = select_latest_forecasts(self.forecasts_dir, "test", diagnostics)
        return load_model_ratings(selected, self.catalog, diagnostics), diagnostics

    def workbook(self, name, questions, startup="SyntheticVenture"):
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = startup
        sheet.append(["Topic", "Questions", "Score", "Confidence"])
        for question, score in questions:
            sheet.append([None, question, score, 4])
        path = self.experts_dir / name
        workbook.save(path)
        workbook.close()
        return path

    def test_catalog_and_reviewed_aliases(self):
        self.assertEqual(self.catalog.question_id.tolist(), [f"Q{i:03d}" for i in range(1, 38)])
        lookup = question_lookup(self.catalog)
        self.assertEqual(lookup["the venture prioritzes creating value for the environment."], ("Q009", "reviewed_alias"))
        self.assertNotIn("the team has sufficient knowledge to establish a sustainable business model.", lookup)

    def test_rating_validation(self):
        for value in (0, 6, True, "nan", "inf", None, "no score"):
            self.assertIsNone(valid_rating(value))
        self.assertEqual(valid_rating("2.5"), 2.5)
        self.assertEqual(valid_rating(5), 5.0)

    def test_legacy_ratings_keep_correct_later_positions(self):
        self.write_model(self.forecast(legacy=True))
        rows, diagnostics = self.load_models()
        indexed = rows.set_index("question_id")
        self.assertEqual(len(rows), 37)
        self.assertTrue(pd.isna(indexed.loc["Q013", "model_rating"]))
        self.assertTrue(pd.isna(indexed.loc["Q019", "model_rating"]))
        self.assertEqual(indexed.loc["Q014", "model_rating"], 5)
        self.assertEqual(indexed.loc["Q015", "model_rating"], 1)
        self.assertEqual(indexed.loc["Q020", "model_rating"], 5)
        self.assertEqual(indexed.loc["Q021", "model_rating"], 1)
        self.assertEqual(indexed.loc["Q022", "model_rating"], 1)
        self.assertEqual(sum(row["reason"] == "criterion_absent_from_legacy_rubric" for row in diagnostics), 2)

    def test_current_format_and_confidence_preserved_without_weighting(self):
        self.write_model(self.forecast())
        rows, diagnostics = self.load_models()
        self.assertEqual(rows.model_rating.notna().sum(), 37)
        self.assertEqual(rows.iloc[0].model_rating, 1)
        self.assertEqual(rows.iloc[0].confidence, 5)
        self.assertEqual(diagnostics, [])

    def test_rag_criteria_text_controls_order(self):
        blocks = []
        for block, (_, group) in zip(self.forecast(), self.catalog.groupby("topic", sort=False)):
            blocks.append({"Topic": block["Topic"], "criteria": list(reversed(group.question_text.tolist())),
                           "stage2_scoring": block})
        self.write_model({"business_plan_file": "SyntheticVenture_V1.md", "topics": blocks})
        rows, _ = self.load_models()
        indexed = rows.set_index("question_id")
        self.assertEqual(indexed.loc["Q003", "model_rating"], 1)
        self.assertEqual(indexed.loc["Q001", "model_rating"], 3)
        self.assertEqual(indexed.loc["Q001", "mapping_method"], "explicit_criterion_text")

    def test_malformed_topic_is_quarantined_without_discarding_other_topics(self):
        blocks = self.forecast(legacy=True)
        blocks[-1]["ratings"].append(4)
        self.write_model(blocks, "ExampleVenture_V1.json")
        rows, diagnostics = self.load_models()
        team = rows.loc[rows.source_topic == "Team Competencies"]
        self.assertTrue(team.model_rating.isna().all())
        self.assertTrue(team.status.eq("malformed_topic_length").all())
        self.assertEqual(rows.model_rating.notna().sum(), 29)
        self.assertTrue(any("Found 7 ratings" in item["detail"] for item in diagnostics))

    def test_generated_rag_classification_label_does_not_override_criteria(self):
        blocks = self.forecast()
        group = self.catalog.loc[self.catalog.topic == blocks[1]["Topic"]]
        scoring = dict(blocks[1], Topic="Economic")
        blocks[1] = {"Topic": group.iloc[0].topic, "criteria": group.question_text.tolist(),
                     "stage2_scoring": scoring}
        self.write_model({"topics": blocks})
        rows, diagnostics = self.load_models()
        self.assertEqual(rows.model_rating.notna().sum(), 37)
        self.assertEqual(rows.set_index("question_id").loc["Q004", "source_scoring_topic"], "Economic")
        self.assertTrue(any(item["reason"] == "scoring_topic_label_ignored" for item in diagnostics))

    def test_ambiguous_explicit_criteria_and_invalid_scores(self):
        blocks = self.forecast()
        blocks[0]["criteria"] = ["Unrecognized criterion"] * 3
        blocks[1]["ratings"][0] = 6
        self.write_model({"topics": blocks})
        rows, _ = self.load_models()
        indexed = rows.set_index("question_id")
        self.assertEqual(indexed.loc["Q001", "status"], "ambiguous_criterion_text")
        self.assertEqual(indexed.loc["Q004", "status"], "invalid_model_rating")
        self.assertEqual(indexed.loc["Q005", "status"], "available")

    def test_latest_version_and_invalid_latest_do_not_fall_back(self):
        self.write_model(self.forecast(), "SyntheticVenture_V1.json")
        self.write_model({"bad": "format"}, "SyntheticVenture_V2.json")
        selected = select_latest_forecasts(self.forecasts_dir, "test")
        self.assertEqual(selected.iloc[0].version, 2)
        self.assertEqual(json.loads(selected.iloc[0].available_versions), [1, 2])
        rows, _ = self.load_models()
        self.assertTrue(rows.status.eq("invalid_forecast").all())
        self.assertTrue(rows.version.eq(2).all())

    def test_duplicate_latest_version_is_not_arbitrarily_selected(self):
        self.write_model(self.forecast(), "SyntheticVenture_V1.json")
        self.write_model(self.forecast(), "SyntheticVenture_V01.json")
        rows, diagnostics = self.load_models()
        self.assertTrue(rows.model_rating.isna().all())
        self.assertTrue(rows.status.eq("duplicate_latest_version").all())
        self.assertEqual(diagnostics[0]["reason"], "duplicate_latest_version")

    def test_workbook_alias_outside_question_missing_rating_and_provenance(self):
        questions = [("The venture prioritzes creating value for the environment.", 4),
                     (self.catalog.iloc[0].question_text, None),
                     ("The team has sufficient knowledge to establish a sustainable business model.", 4)]
        self.workbook("Completed_ExpertA_Score_Card.xlsx", questions)
        diagnostics = []
        rows = load_expert_ratings(self.experts_dir, self.catalog, diagnostics)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows.iloc[0].question_id, "Q009")
        self.assertEqual(rows.iloc[0].expert_id, "Completed_ExpertA_Score_Card.xlsx")
        self.assertEqual(rows.iloc[0].expert_name, "ExpertA")
        self.assertEqual(rows.iloc[0].rating_cell, "C2")
        self.assertEqual(rows.iloc[0].match_method, "reviewed_alias")
        self.assertEqual(rows.iloc[1].status, "missing_expert_rating")
        self.assertEqual({row["reason"] for row in diagnostics}, {"question_outside_rubric", "missing_expert_rating"})

    def test_duplicate_observations_are_excluded_even_if_equal(self):
        text = self.catalog.iloc[0].question_text
        self.workbook("ExpertA_Score_Card.xlsx", [(text, 3), (text, 3)])
        rows = load_expert_ratings(self.experts_dir, self.catalog)
        self.assertTrue(rows.rating.isna().all())
        self.assertTrue(rows.status.eq("duplicate_expert_observation").all())

    def test_prepare_data_keeps_union_and_requires_two_complete_experts(self):
        first, second = self.catalog.iloc[:2].question_text
        self.workbook("ExpertA_Score_Card.xlsx", [(first, 2), (second, None)])
        self.workbook("ExpertB_Score_Card.xlsx", [(first, 4), (second, 5)], startup="syntheticventure")
        self.write_model(self.forecast(), "SYNTHETICVENTURE_V2.json")
        self.write_model(self.forecast(), "ModelOnly_V1.json")
        frames = prepare_data(self.root, RUBRIC, self.experts_dir, {"configurations": [self.configuration]})
        aligned = frames["aligned_records"]
        self.assertEqual(len(aligned), 74)
        eligible = aligned.loc[aligned.eligible]
        self.assertEqual(len(eligible), 1)
        self.assertEqual(eligible.iloc[0].startup, "SyntheticVenture")
        self.assertEqual(eligible.iloc[0].expert_consensus, 3)
        self.assertEqual(eligible.iloc[0].version, 2)
        self.assertEqual(eligible.iloc[0].expert_1_id, "ExpertA_Score_Card.xlsx")
        self.assertEqual(eligible.iloc[0].expert_2_id, "ExpertB_Score_Card.xlsx")
        self.assertIn("missing_expert_rating", aligned.loc[aligned.question_id == "Q002"].iloc[0].exclusion_reason)

    def test_more_than_two_experts_is_ambiguous_not_truncated(self):
        text = self.catalog.iloc[0].question_text
        for name in ("ExpertA", "ExpertB", "ExpertC"):
            self.workbook(f"{name}_Score_Card.xlsx", [(text, 3)])
        self.write_model(self.forecast())
        frames = prepare_data(self.root, RUBRIC, self.experts_dir, {"configurations": [self.configuration]})
        self.assertFalse(frames["aligned_records"].eligible.any())
        self.assertTrue(frames["diagnostics"].reason.eq("ambiguous_expert_count").any())


if __name__ == "__main__":
    unittest.main()
