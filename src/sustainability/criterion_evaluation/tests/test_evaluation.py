"""Tests of statistical definitions, startup cohorts, and paired resampling."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from sustainability.criterion_evaluation.evaluation import METRICS, compute_metrics, evaluate


def catalog(count: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "question_id": f"Q{index:03d}",
                "topic": "Test topic",
                "category": "Test category",
                "question_text": f"Criterion {index}?",
            }
            for index in range(1, count + 1)
        ]
    )


def configuration(four: bool = False) -> dict:
    providers = ["gpt", "gemini"] if four else ["gpt"]
    return {
        "configurations": [
            {
                "id": f"{provider}_{method}",
                "label": f"{provider} {method}",
                "provider": provider,
                "method": method,
            }
            for provider in providers
            for method in ["full_context", "rag"]
        ],
        "comparisons": [
            {
                "id": provider,
                "label": provider.upper(),
                "full_context": f"{provider}_full_context",
                "rag": f"{provider}_rag",
            }
            for provider in providers
        ],
    }


def records(
    e1=(1, 2, 3, 4, 5),
    e2=(2, 2, 3, 4, 4),
    full=(2, 2, 4, 4, 4),
    rag=(1, 3, 3, 5, 5),
    question_id="Q001",
) -> pd.DataFrame:
    rows = []
    for index, (first, second, full_rating, rag_rating) in enumerate(zip(e1, e2, full, rag)):
        for method, model_rating in [("full_context", full_rating), ("rag", rag_rating)]:
            rows.append(
                {
                    "startup": f"startup_{index:03d}",
                    "question_id": question_id,
                    "config_id": f"gpt_{method}",
                    "expert_1": first,
                    "expert_2": second,
                    "model_rating": model_rating,
                    "version": 1,
                    "eligible": True,
                    "exclusion_reason": "",
                }
            )
    return pd.DataFrame(rows)


class MetricDefinitionsTests(unittest.TestCase):
    def test_hand_calculated_eight_metrics_and_tied_rank_correlation(self):
        actual = compute_metrics([1, 2, 4], [3, 2, 2], [2, 3, 1])
        expected = {
            "pearson": -np.sqrt(3) / 2,
            "spearman": -np.sqrt(3) / 2,
            "mae": 1,
            "rmse": np.sqrt(5 / 3),
            "bias": -1 / 3,
            "human_human_mae": 4 / 3,
            "model_human_mae": 4 / 3,
            "hrmae": 1,
        }
        self.assertEqual(set(actual), set(METRICS))
        for metric, value in expected.items():
            with self.subTest(metric=metric):
                self.assertAlmostEqual(actual[metric], value)

    def test_hrmae_uses_individual_experts_and_correct_direction(self):
        actual = compute_metrics([1, 2], [3, 4], [2, 3])
        self.assertEqual(actual["mae"], 0)
        self.assertEqual(actual["model_human_mae"], 1)
        self.assertEqual(actual["human_human_mae"], 2)
        self.assertEqual(actual["hrmae"], 2)
        example = compute_metrics([1] * 5, [2, 2, 2, 1, 1], [1, 1, 1, 2, 1])
        self.assertAlmostEqual(example["hrmae"], 1.2)
        self.assertAlmostEqual(100 * (1 - 1 / example["hrmae"]), 100 / 6)

    def test_constant_inputs_and_zero_denominator_are_undefined(self):
        perfect = compute_metrics([3] * 5, [3] * 5, [3] * 5)
        self.assertTrue(np.isnan(perfect["pearson"]))
        self.assertTrue(np.isnan(perfect["spearman"]))
        self.assertTrue(np.isnan(perfect["hrmae"]))
        self.assertEqual(perfect["mae"], 0)
        disagreement = compute_metrics([3] * 5, [3] * 5, [4] * 5)
        self.assertEqual(disagreement["hrmae"], 0)

    def test_empty_and_one_startup_inputs(self):
        self.assertTrue(all(np.isnan(value) for value in compute_metrics([], [], []).values()))
        one = compute_metrics([1], [3], [3])
        self.assertTrue(np.isnan(one["pearson"]))
        self.assertEqual(one["mae"], 1)

    def test_invalid_complete_metric_inputs_raise(self):
        for e1, e2, model in [([1], [1, 2], [1]), ([1], [np.nan], [2]), ([0], [1], [2])]:
            with self.subTest(inputs=(e1, e2, model)), self.assertRaises(ValueError):
                compute_metrics(e1, e2, model)


class EvaluationCohortTests(unittest.TestCase):
    def test_all_37_criteria_and_four_configurations_remain_in_output(self):
        results, comparisons = evaluate(records(), catalog(37), configuration(True), bootstrap=20)
        self.assertEqual(len(results), 148)
        self.assertEqual(len(comparisons), 74)
        self.assertEqual(comparisons.groupby("comparison_id").size().to_dict(), {"gemini": 37, "gpt": 37})
        absent = results.loc[results["question_id"].eq("Q037")]
        self.assertTrue(absent["n"].eq(0).all())
        self.assertTrue(absent["status"].eq("no_eligible_startups").all())
        self.assertTrue(absent[list(METRICS)].isna().all().all())
        for metric in METRICS:
            self.assertTrue(absent[f"{metric}_n_boot_valid"].eq(0).all())

    def test_missing_experts_invalid_ratings_and_unknown_eligibility_are_excluded(self):
        frame = records()
        frame.loc[0, "expert_2"] = np.nan
        frame.loc[2, "model_rating"] = 6
        frame.loc[4, "eligible"] = False
        frame["eligible"] = frame["eligible"].astype(object)
        frame.loc[6, "eligible"] = "True"
        results, comparisons = evaluate(frame, catalog(), configuration(), bootstrap=30)
        full = results.loc[results["config_id"].eq("gpt_full_context")].iloc[0]
        self.assertEqual(full["n"], 1)
        self.assertEqual(full["status"], "insufficient_startups")
        self.assertEqual(full["mae"], 0.5)
        self.assertEqual(full["human_human_mae"], 1)
        self.assertTrue(np.isnan(full["pearson"]))
        for metric in METRICS:
            self.assertEqual(full[f"{metric}_n_boot_valid"], 0)
            self.assertTrue(np.isnan(full[f"{metric}_ci_low"]))
        self.assertEqual(comparisons.iloc[0]["n"], 1)

    def test_confidence_intervals_require_at_least_five_startups(self):
        four = records().loc[lambda frame: frame["startup"].ne("startup_004")]
        results, comparisons = evaluate(four, catalog(), configuration(), bootstrap=50)
        self.assertTrue(results["n"].eq(4).all())
        self.assertTrue(results["mae"].notna().all())
        self.assertTrue(results["mae_ci_low"].isna().all())
        self.assertTrue(results["mae_n_boot_valid"].eq(0).all())
        self.assertEqual(comparisons.iloc[0]["delta_mae_n_boot_valid"], 0)
        five, pairs = evaluate(records(), catalog(), configuration(), bootstrap=50)
        self.assertTrue(five["mae_n_boot_valid"].eq(50).all())
        self.assertEqual(pairs.iloc[0]["delta_mae_n_boot_valid"], 50)

    def test_invalid_unknown_and_inconsistent_versions_are_excluded_from_pairing(self):
        frame = records()
        frame.loc[1, "version"] = 2
        frame.loc[3, "version"] = np.nan
        frame.loc[5, "expert_1"] = 1
        # Integral float and integer version representations refer to the same plan.
        frame["version"] = frame["version"].astype(object)
        frame.loc[7, "version"] = "1.0"
        results, comparisons = evaluate(frame, catalog(), configuration(), bootstrap=20)
        self.assertTrue(results["n"].eq(5).all())
        pair = comparisons.iloc[0]
        self.assertEqual(pair["n"], 2)
        self.assertEqual(pair["n_common_startups"], 5)
        self.assertEqual(pair["n_version_mismatch"], 2)
        self.assertEqual(pair["n_expert_mismatch"], 1)

    def test_unequal_coverage_does_not_change_paired_maes(self):
        common = records(e1=[2] * 5, e2=[2] * 5, full=[2] * 5, rag=[3] * 5)
        extra = common.iloc[:1].copy()
        extra["startup"] = "only_in_full"
        extra["model_rating"] = 5
        frame = pd.concat([common, extra], ignore_index=True)
        results, comparisons = evaluate(frame, catalog(), configuration(), bootstrap=100)
        full_result = results.loc[results["config_id"].eq("gpt_full_context")].iloc[0]
        self.assertEqual(full_result["mae"], 0.5)
        pair = comparisons.iloc[0]
        self.assertEqual(pair["n_full_eligible"], 6)
        self.assertEqual(pair["n_rag_eligible"], 5)
        self.assertEqual(pair["n"], 5)
        self.assertEqual(pair["full_mae"], 0)
        self.assertEqual(pair["rag_mae"], 1)
        self.assertEqual(pair["delta_mae"], 1)
        self.assertEqual(pair["delta_mae_ci_low"], 1)
        self.assertEqual(pair["delta_mae_ci_high"], 1)

    def test_duplicate_startup_records_are_an_explicit_error(self):
        frame = records()
        duplicate = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate(duplicate, catalog(), configuration())

    def test_ranks_are_descending_ties_share_rank_and_missing_rank_is_na(self):
        first = records(e1=[2] * 5, e2=[2] * 5, full=[2] * 5, rag=[3] * 5)
        second = first.assign(question_id="Q002")
        third = records(e1=[2] * 5, e2=[2] * 5, full=[3] * 5, rag=[2] * 5, question_id="Q003")
        _, comparisons = evaluate(pd.concat([first, second, third]), catalog(4), configuration(), bootstrap=10)
        self.assertEqual(comparisons["question_id"].tolist(), ["Q001", "Q002", "Q003", "Q004"])
        self.assertEqual(comparisons["rank"].iloc[:3].tolist(), [1, 1, 3])
        self.assertTrue(pd.isna(comparisons["rank"].iloc[3]))
        self.assertEqual(comparisons.iloc[0]["interpretation"], "full_context_lower_mae")
        self.assertEqual(comparisons.iloc[2]["interpretation"], "rag_lower_mae")

    def test_equal_deltas_with_different_baseline_errors_share_rank(self):
        first = records(e1=[1] * 5, e2=[1] * 5,
                        full=[2, 2, 1, 1, 1], rag=[2, 2, 2, 1, 1])
        second = records(e1=[1] * 5, e2=[1] * 5,
                         full=[2, 2, 2, 1, 1], rag=[2, 2, 2, 2, 1], question_id="Q002")
        _, comparisons = evaluate(pd.concat([first, second]), catalog(2), configuration(), bootstrap=10)
        self.assertEqual(comparisons["delta_mae"].tolist(), [0.2, 0.2])
        self.assertEqual(comparisons["rank"].tolist(), [1, 1])


class BootstrapTests(unittest.TestCase):
    def test_reproducible_and_invariant_to_input_and_catalog_order(self):
        frame = pd.concat([records(), records(question_id="Q002")], ignore_index=True)
        first = evaluate(frame, catalog(2), configuration(), bootstrap=200, seed=42)
        changed_order = evaluate(
            frame.sample(frac=1, random_state=37),
            catalog(2).iloc[::-1],
            configuration(),
            bootstrap=200,
            seed=42,
        )
        for expected, actual, keys in zip(
            first, changed_order, [["config_id", "question_id"], ["comparison_id", "question_id"]]
        ):
            pd.testing.assert_frame_equal(
                expected.sort_values(keys).reset_index(drop=True),
                actual.sort_values(keys).reset_index(drop=True),
            )

    def test_pairing_preserves_constant_per_startup_delta_despite_variable_errors(self):
        frame = records(e1=[2] * 5, e2=[2] * 5, full=[2, 3, 2, 3, 2], rag=[3, 4, 3, 4, 3])
        _, comparisons = evaluate(frame, catalog(), configuration(), bootstrap=1000)
        pair = comparisons.iloc[0]
        self.assertAlmostEqual(pair["delta_mae"], 1)
        self.assertEqual(pair["delta_mae_ci_low"], 1)
        self.assertEqual(pair["delta_mae_ci_high"], 1)
        self.assertEqual(pair["delta_mae_n_boot_valid"], 1000)

    def test_single_configuration_resamples_experts_and_model_together(self):
        # Model equals consensus for every startup; independently resampling
        # columns would introduce error and produce a nonzero interval.
        frame = records(e1=[1, 2, 3, 4, 5], e2=[1, 2, 3, 4, 5], full=[1, 2, 3, 4, 5])
        results, _ = evaluate(frame, catalog(), configuration(), bootstrap=200)
        full = results.loc[results["config_id"].eq("gpt_full_context")].iloc[0]
        for name in ["mae", "rmse", "bias", "human_human_mae", "model_human_mae"]:
            self.assertEqual(full[name], 0)
            self.assertEqual(full[f"{name}_ci_low"], 0)
            self.assertEqual(full[f"{name}_ci_high"], 0)
            self.assertEqual(full[f"{name}_n_boot_valid"], 200)
        self.assertEqual(full["hrmae_n_boot_valid"], 0)
        self.assertTrue(np.isnan(full["hrmae_ci_low"]))
        self.assertAlmostEqual(full["pearson_ci_low"], 1)
        self.assertAlmostEqual(full["spearman_ci_high"], 1)

    def test_undefined_bootstrap_correlations_are_counted_and_excluded(self):
        frame = records(e1=[1, 1, 1, 1, 2], e2=[1, 1, 1, 1, 2], full=[1, 1, 1, 1, 2])
        results, _ = evaluate(frame, catalog(), configuration(), bootstrap=300)
        full = results.loc[results["config_id"].eq("gpt_full_context")].iloc[0]
        self.assertGreater(full["pearson_n_boot_valid"], 0)
        self.assertLess(full["pearson_n_boot_valid"], 300)
        self.assertEqual(full["spearman_n_boot_valid"], full["pearson_n_boot_valid"])
        self.assertEqual(full["mae_n_boot_valid"], 300)
        self.assertAlmostEqual(full["pearson_ci_low"], 1)

    def test_bootstrap_zero_hrmae_denominators_are_counted_separately(self):
        frame = records(e1=[1] * 5, e2=[1, 1, 1, 1, 2], full=[1] * 5)
        results, _ = evaluate(frame, catalog(), configuration(), bootstrap=300)
        full = results.loc[results["config_id"].eq("gpt_full_context")].iloc[0]
        self.assertEqual(full["hrmae"], 2)
        self.assertGreater(full["hrmae_n_boot_valid"], 0)
        self.assertLess(full["hrmae_n_boot_valid"], 300)
        self.assertEqual(full["hrmae_ci_low"], 2)
        self.assertEqual(full["hrmae_ci_high"], 2)
        self.assertEqual(full["human_human_mae_n_boot_valid"], 300)

    def test_zero_bootstrap_preserves_points_and_disables_intervals(self):
        results, comparisons = evaluate(records(), catalog(), configuration(), bootstrap=0)
        self.assertTrue(results["mae"].notna().all())
        self.assertTrue(results["mae_ci_low"].isna().all())
        self.assertTrue(results["mae_n_boot_valid"].eq(0).all())
        self.assertEqual(comparisons.iloc[0]["delta_mae_n_boot_valid"], 0)

    def test_invalid_bootstrap_parameters_raise(self):
        for kwargs in [{"bootstrap": -1}, {"bootstrap": 1.5}, {"bootstrap": True}, {"seed": -1}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                evaluate(records(), catalog(), configuration(), **kwargs)


if __name__ == "__main__":
    unittest.main()
