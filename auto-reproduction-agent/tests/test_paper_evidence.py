import json
import tempfile
import unittest
from pathlib import Path

import fitz

from reproducer.runtime.paper_evidence import prepare_paper_evidence
from reproducer.task import TaskSpec


class FakeVisionClient:
    def __init__(self) -> None:
        self.calls = []

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
        else:
            content = json.dumps(
                {
                    "figure_label": "Figure 4",
                    "panel_count": 2,
                    "panels": [],
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
            self.assertEqual(record["analysis"]["panel_count"], 2)
            self.assertEqual(record["focus"], "top-left panel")
            self.assertTrue((workspace / record["assets"]["figure_crop"]).is_file())
            self.assertEqual(len(client.calls), 2)
            self.assertIn("top-left panel", client.calls[0][1])
            self.assertIn("points", client.calls[1][1])


if __name__ == "__main__":
    unittest.main()
