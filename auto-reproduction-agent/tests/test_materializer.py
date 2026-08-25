import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from reproducer.runtime.materializer import materialize_task


@unittest.skipUnless(shutil.which("git"), "git is required")
class MaterializerTests(unittest.TestCase):
    def test_downloads_paper_and_clones_pinned_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            upstream = root / "upstream"
            upstream.mkdir()
            self._git(upstream, "init")
            self._git(upstream, "config", "user.email", "test@example.com")
            self._git(upstream, "config", "user.name", "Test User")
            (upstream / "README.md").write_text("upstream", encoding="utf-8")
            self._git(upstream, "add", "README.md")
            self._git(upstream, "commit", "-m", "initial")
            commit = self._git(upstream, "rev-parse", "HEAD")

            source_pdf = root / "source.pdf"
            source_pdf.write_bytes(b"%PDF-test-materializer")
            paper_hash = hashlib.sha256(source_pdf.read_bytes()).hexdigest()
            task_dir = root / "tasks" / "demo"
            task_dir.mkdir(parents=True)
            task_file = task_dir / "task.json"
            task_file.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "task_id": "demo",
                        "title": "Demo",
                        "paper": {
                            "path": "paper.pdf",
                            "url": source_pdf.as_uri(),
                            "sha256": paper_hash,
                        },
                        "repository": {
                            "path": "repository",
                            "url": str(upstream),
                            "commit": commit,
                        },
                        "claims": [{"claim_id": "C1", "statement": "Claim"}],
                    }
                ),
                encoding="utf-8",
            )

            first = materialize_task(task_file)
            second = materialize_task(task_file)

            self.assertEqual(first.paper_status, "downloaded")
            self.assertEqual(first.repository_status, "cloned")
            self.assertEqual(second.paper_status, "reused")
            self.assertEqual(second.repository_status, "reused")
            self.assertTrue(first.task.paper_path.is_file())
            self.assertEqual(
                self._git(first.task.repository_path, "rev-parse", "HEAD"), commit
            )

    @staticmethod
    def _git(cwd: Path, *arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=True,
        )
        return completed.stdout.strip()


if __name__ == "__main__":
    unittest.main()
