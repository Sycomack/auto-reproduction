import json
import tempfile
import unittest
from pathlib import Path

import fitz

from reproducer.runtime.paper_evidence import prepare_paper_evidence
from reproducer.task import TaskSpec


class FakeVisionClient:
    def __init__(self, invalid_coordinates: bool = False) -> None:
        self.calls = []
        self.invalid_coordinates = invalid_coordinates

    def analyze(self, image_path, prompt, detail="high"):
        self.calls.append((image_path, prompt, detail))
        if len(self.calls) == 1:
            content = json.dumps(
                {
                    "found": True,
                    "bbox": [0.08, 0.08, 0.92, 0.62],
                    "caption_bbox": [0.1, 0.63, 0.9, 0.7],
                    "confidence": 0.95,
                }
            )
        elif len(self.calls) == 2:
            content = json.dumps(
                {
                    "found": True,
                    "bbox": [0.02, 0.02, 0.27, 0.36],
                    "confidence": 0.96,
                }
            )
        else:
            x_values = [100, 80] if self.invalid_coordinates else [100, 60]
            content = json.dumps(
                {
                    "figure_label": "Figure 4",
                    "panel_count": 1,
                    "panels": [
                        {
                            "panel_title": "XSUM, LLaMA-7B",
                            "dataset": "XSUM",
                            "model": "LLaMA-7B",
                            "metric": "ROUGE-2",
                            "x_axis": "KV Cache Budget (%)",
                            "y_axis": "ROUGE-2",
                            "series": [
                                {
                                    "name": "H2O",
                                    "points": [
                                        {
                                            "x": x_value,
                                            "y": 12.0,
                                            "uncertainty": "estimated",
                                        }
                                        for x_value in x_values
                                    ],
                                }
                            ],
                        }
                    ],
                    "uncertainties": [],
                }
            )
        return {"model": "fake-vision", "content": content, "usage": {}}


class PaperEvidenceTests(unittest.TestCase):
    def test_prepare_finds_caption_renders_crop_and_writes_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "paper.pdf"
            document = fitz.open()
            page = document.new_page()
            page.draw_rect(fitz.Rect(60, 60, 530, 410))
            page.insert_text((70, 440), "Figure 4: Main experimental result")
            document.save(pdf_path)
            document.close()
            repository = root / "repository"
            repository.mkdir()
            task_file = root / "task.json"
            task_file.write_text(
                json.dumps(
                    {
                        "task_id": "visual-demo",
                        "title": "Visual demo",
                        "paper": {"path": "paper.pdf"},
                        "repository": {"path": "repository"},
                        "claims": [{"claim_id": "C1", "statement": "Claim"}],
                        "visual_inputs": [
                            {
                                "id": "figure_4",
                                "figure_label": "Figure 4",
                                "purpose": "main result",
                                "focus": "top-left panel",
                            }
                        ],
                        "reproduction_protocol": {
                            "reported_results": {
                                "absolute_digitization_tolerance": 1.0,
                                "series": {
                                    "H2O": [
                                        {"budget_percent": 100, "rouge_2": 12.2},
                                        {"budget_percent": 60, "rouge_2": 12.1},
                                    ]
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            task = TaskSpec.load(task_file)
            workspace = root / "workspace"
            workspace.mkdir()
            client = FakeVisionClient()

            evidence_path = prepare_paper_evidence(
                task, pdf_path, workspace, vision_client=client
            )

            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            record = evidence["visual_inputs"][0]
            self.assertEqual(record["status"], "analyzed")
            self.assertEqual(record["page"], 1)
            self.assertEqual(record["localization_method"], "vision_bbox")
            self.assertEqual(record["focus_localization_method"], "vision_bbox")
            self.assertEqual(record["analysis"]["panel_count"], 1)
            self.assertEqual(record["analysis_validation"]["status"], "passed")
            self.assertEqual(record["focus"], "top-left panel")
            self.assertTrue((workspace / record["assets"]["figure_crop"]).is_file())
            self.assertTrue((workspace / record["assets"]["focus_crop"]).is_file())
            self.assertEqual(len(client.calls), 3)
            self.assertIn("complete figure", client.calls[0][1])
            self.assertIn("top-left panel", client.calls[1][1])
            self.assertIn("[100.0, 60.0]", client.calls[2][1])
            self.assertEqual(client.calls[2][0].name, "figure_4.png")

    def test_unexpected_axis_ticks_invalidate_numeric_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "paper.pdf"
            document = fitz.open()
            page = document.new_page()
            page.insert_text((70, 440), "Figure 4: Main experimental result")
            document.save(pdf_path)
            document.close()
            repository = root / "repository"
            repository.mkdir()
            task_file = root / "task.json"
            task_file.write_text(
                json.dumps(
                    {
                        "task_id": "invalid-visual-demo",
                        "title": "Invalid visual demo",
                        "paper": {"path": "paper.pdf"},
                        "repository": {"path": "repository"},
                        "claims": [{"claim_id": "C1", "statement": "Claim"}],
                        "visual_inputs": [
                            {
                                "id": "figure_4",
                                "figure_label": "Figure 4",
                                "focus": "top-left panel",
                            }
                        ],
                        "reproduction_protocol": {
                            "reported_results": {
                                "absolute_digitization_tolerance": 1.0,
                                "series": {
                                    "H2O": [
                                        {"budget_percent": 100, "rouge_2": 12.2},
                                        {"budget_percent": 60, "rouge_2": 12.1},
                                    ]
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            workspace = root / "workspace"
            workspace.mkdir()

            evidence_path = prepare_paper_evidence(
                TaskSpec.load(task_file),
                pdf_path,
                workspace,
                vision_client=FakeVisionClient(invalid_coordinates=True),
            )

            record = json.loads(evidence_path.read_text(encoding="utf-8"))[
                "visual_inputs"
            ][0]
            self.assertEqual(record["status"], "analysis_invalid")
            validation = record["analysis_validation"]
            self.assertEqual(validation["status"], "failed")
            self.assertEqual(validation["series"][0]["missing_x"], [60.0])
            self.assertEqual(validation["series"][0]["unexpected_x"], [80.0])


if __name__ == "__main__":
    unittest.main()
