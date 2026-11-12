"""Configuration and destination guards for the independent command."""

import json
from pathlib import Path
import tempfile
import unittest

from sustainability.criterion_evaluation.__main__ import (
    WORKFLOW, check_output_directory, load_configuration, parse_args,
)


class RunnerTest(unittest.TestCase):
    def test_rejects_unrelated_nonempty_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            protected = output / "old_results.csv"
            protected.write_text("unchanged", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unrelated"):
                check_output_directory(output)
            self.assertEqual(protected.read_text(), "unchanged")
            (output / "run_manifest.json").write_text(json.dumps({"workflow": WORKFLOW}))
            check_output_directory(output)

    def test_rejects_invalid_method_pair_reference(self):
        config = {
            "configurations": [{"id": "full", "label": "Full", "provider": "Test",
                                "method": "full_context", "input_dir": "outputs/test"}],
            "comparisons": [{"id": "test", "label": "Test", "full_context": "full", "rag": "full"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "Invalid rag reference"):
                load_configuration(path)

    def test_alias_arguments_and_defaults(self):
        args = parse_args(["--experts-dir", "experts", "--rubric", "questions.json"])
        self.assertEqual(args.expert_dir, Path("experts"))
        self.assertEqual(args.questions, Path("questions.json"))
        self.assertEqual(args.bootstrap, 2000)
        self.assertEqual(args.seed, 42)


if __name__ == "__main__":
    unittest.main()
