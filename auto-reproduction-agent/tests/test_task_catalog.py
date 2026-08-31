from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from reproducer.agent.prompts import SYSTEM_PROMPT


class MainExperimentTaskContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[2]
        cls.tasks_root = cls.project_root / "tasks"
        cls.catalog = json.loads(
            (cls.tasks_root / "catalog.json").read_text(encoding="utf-8")
        )

    def test_catalog_paths_and_ids_are_consistent(self) -> None:
        for entry in self.catalog["tasks"]:
            task_path = self.tasks_root / entry["path"]
            self.assertTrue(task_path.is_file(), task_path)
            task = json.loads(task_path.read_text(encoding="utf-8"))
            self.assertEqual(entry["task_id"], task["task_id"])

    def test_active_tasks_define_paper_aligned_local_experiments(self) -> None:
        active_suites = {"main_experiments", "main_protocol_adaptations"}
        allowed_scopes = {
            "full_main_evaluation",
            "main_table_subset",
            "main_experiment_protocol",
        }
        for entry in self.catalog["tasks"]:
            if entry.get("suite") not in active_suites:
                continue
            task = json.loads(
                (self.tasks_root / entry["path"]).read_text(encoding="utf-8")
            )
            self.assertTrue(task["task_class"].startswith("paper_main_"))
            self.assertIn(task["paper_alignment"]["scope"], allowed_scopes)
            self.assertRegex(
                task["paper_alignment"]["paper_location"],
                re.compile(r"(Table|Section|Figure)", re.IGNORECASE),
            )
            self.assertEqual(task["model_execution"]["mode"], "local_weights_only")
            self.assertIs(
                task["model_execution"]["external_model_api_allowed_for_experiment"],
                False,
            )
            self.assertTrue(task["claims"])
            for claim in task["claims"]:
                self.assertRegex(
                    claim["source"],
                    re.compile(r"(Table|Section|Figure)", re.IGNORECASE),
                )

    def test_active_inputs_are_immutable(self) -> None:
        active_suites = {"main_experiments", "main_protocol_adaptations"}
        for entry in self.catalog["tasks"]:
            if entry.get("suite") not in active_suites:
                continue
            task = json.loads(
                (self.tasks_root / entry["path"]).read_text(encoding="utf-8")
            )
            self.assertRegex(task["repository"]["commit"], r"^[0-9a-f]{40}$")
            self.assertRegex(task["paper"]["sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue(task["paper"]["url"].startswith("https://"))
            self.assertTrue(task["repository"]["url"].startswith("https://"))

    def test_agent_prompt_preserves_scope_and_credentials(self) -> None:
        self.assertIn("do not reduce a declared", SYSTEM_PROMPT)
        self.assertIn("Never embed a credential", SYSTEM_PROMPT)
        self.assertIn("local-only must not call an external model API", SYSTEM_PROMPT)

    def test_agent_prompt_treats_author_environment_as_protocol(self) -> None:
        self.assertIn("author-declared software versions", SYSTEM_PROMPT)
        self.assertIn("task-local isolated environment", SYSTEM_PROMPT)
        self.assertIn("Install exact author-declared versions first", SYSTEM_PROMPT)
        self.assertIn("latest dependency versions by default", SYSTEM_PROMPT)
        self.assertIn("compatibility deviation, and report", SYSTEM_PROMPT)

    def test_streamingllm_pins_model_and_author_environment(self) -> None:
        task = json.loads(
            (self.tasks_root / "streamingllm" / "task.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            task["environment_protocol"]["author_baseline"]["transformers"],
            "4.33.0",
        )
        self.assertRegex(
            task["model_execution"]["models"][0]["revision"],
            r"^[0-9a-f]{40}$",
        )
        self.assertIn(
            "Do not silently upgrade Transformers",
            "\n".join(task["environment_protocol"]["prohibited_shortcuts"]),
        )


if __name__ == "__main__":
    unittest.main()
