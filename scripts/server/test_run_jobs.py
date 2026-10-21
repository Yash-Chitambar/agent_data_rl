"""Capacity checks must never launch or mutate a run on a busy shared GPU."""
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.server import run_jobs as jobs


class CapacityTests(unittest.TestCase):
    def check(self, output):
        with patch.object(jobs.subprocess, "check_output", side_effect=output), \
                patch.object(jobs.time, "sleep"):
            return jobs.gpu_idle("1")

    def test_idle_requires_three_samples(self):
        self.assertTrue(self.check(["40, 0", ""] * 3)[0])

    def test_busy_memory_without_a_visible_process(self):
        self.assertFalse(self.check(["1024, 0", ""])[0])

    def test_compute_or_process_blocks_launch(self):
        for outputs in (["40, 2", ""], ["40, 0", "12345"]):
            with self.subTest(outputs=outputs):
                self.assertFalse(self.check(outputs)[0])

    def test_later_sample_becomes_busy(self):
        self.assertFalse(self.check(["40, 0", "", "2048, 0", "12345"])[0])

    def test_unreadable_capacity_fails_closed(self):
        for output in (["N/A, N/A"], [OSError("nvidia-smi unavailable")]):
            with self.subTest(output=output):
                self.assertFalse(self.check(output)[0])

    def test_wait_keeps_run_intact_and_does_not_count_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "jobs.txt"
            manifest.write_text("python -m blind_insertion.train --name pilot\n")
            run = root / "runs" / "pilot"
            run.mkdir(parents=True)
            (run / "last.pt").write_bytes(b"existing checkpoint")
            args = argparse.Namespace(state_dir=str(root / "state"), runs_dir=str(root / "runs"),
                                      gpus="0", per_gpu=1, require_idle_gpus=True,
                                      dry_run=False, force=False, retries=0, manifests=[manifest],
                                      only="", skip_tag=[], max_jobs=1, watch=False, poll=1)
            runner = jobs.Runner(args)
            # The GPU becomes busy between the repeated check and the launch check.
            with patch.object(jobs, "gpu_idle", side_effect=[(True, "idle"), (False, "busy")]), \
                    patch.object(jobs.subprocess, "Popen") as launch, \
                    patch.object(jobs.time, "sleep", side_effect=InterruptedError("end test")):
                with self.assertRaises(InterruptedError):
                    runner.run()  # Must wait, rather than exit or discard the queued job.
                launch.assert_not_called()
            self.assertEqual((run / "last.pt").read_bytes(), b"existing checkpoint")
            self.assertEqual(runner.attempts, {})
            self.assertEqual(runner.running, {})


if __name__ == "__main__":
    unittest.main()
