"""Submission snapshot isolation, without SLURM or model dependencies."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class LauncherTests(unittest.TestCase):
    def test_dry_run_uses_frozen_commit_and_rejects_tracked_edits(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            (repo / "snellius").mkdir(parents=True)
            (repo / "scripts").mkdir()
            for name in ("submit_r9.sh", "job_r9.sbatch"):
                shutil.copy(root / "snellius" / name, repo / "snellius" / name)
            runner = repo / "scripts" / "exp9_reserve.py"
            runner.write_text("original commit\n")

            def git(*args):
                return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)

            git("init")
            git("add", ".")
            git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.com",
                "commit", "-m", "fixture")
            commit = git("rev-parse", "HEAD").stdout.strip()
            out = Path(directory) / "results"
            env = os.environ | {"DRY_RUN": "1", "HSPEC_OUT": str(out)}
            command = ["bash", "snellius/submit_r9.sh", "both"]
            result = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True, check=True)
            snapshot, = list(out.glob("source_*"))
            self.assertEqual((snapshot / ".hspec_commit").read_text().strip(), commit)
            self.assertIn(str(snapshot / "snellius/job_r9.sbatch"), result.stdout)
            self.assertIn("--gpus-per-node=2", result.stdout)
            self.assertIn("--gpus-per-node=3", result.stdout)
            runner.write_text("changed checkout\n")
            self.assertEqual((snapshot / "scripts/exp9_reserve.py").read_text(), "original commit\n")
            failed = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("Commit tracked changes", failed.stderr)


if __name__ == "__main__":
    unittest.main()
