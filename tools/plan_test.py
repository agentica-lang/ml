"""The plan of CI (tools/plan.py) on the real agentica.toml files of this repository."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plan

ROOT = Path(__file__).resolve().parent.parent
ALL = plan.packages(ROOT)


class DependencyWalk(unittest.TestCase):
    def test_a_leaf_brings_only_itself(self):
        deps = plan.dependencies(ROOT)
        self.assertEqual(plan.closure(deps, {"vision"}), {"vision"})
        self.assertEqual(plan.closure(deps, {"whisper"}), {"whisper"})

    def test_dependents_come_in_through_others(self):
        deps = plan.dependencies(ROOT)
        # server/ reads transformers/, which reads nn/, so a change to nn/ runs it too.
        self.assertEqual(
            plan.closure(deps, {"nn"}), {"nn", "hub", "asr", "train", "transformers", "vision", "server"}
        )
        self.assertEqual(plan.closure(deps, {"tensor"}), set(ALL) - {"whisper", "agc-ir"})
        self.assertEqual(plan.closure(deps, {"agc-ir"}), set(ALL) - {"whisper"})


class Selection(unittest.TestCase):
    def select(self, event, *changed):
        return plan.select(ROOT, event, list(changed))

    def test_a_push_runs_everything(self):
        out = self.select("push", "vision/vit.agc")
        self.assertEqual(out["packages"], ALL)
        self.assertEqual([out[k] for k in ("xla", "torch", "ggml_arm64", "vulkan")], ["true"] * 4)

    def test_a_file_outside_the_packages_runs_everything(self):
        for path in (".github/workflows/ci.yml", "tools/plan.py", "third_party/ggml"):
            self.assertEqual(self.select("pull_request", path)["packages"], ALL)

    def test_one_leaf_runs_only_its_job(self):
        out = self.select("pull_request", "vision/vit.agc")
        self.assertEqual(out["packages"], ["vision"])
        self.assertEqual([out[k] for k in ("xla", "torch", "ggml_arm64", "vulkan")], ["false"] * 4)

    def test_no_changed_file_runs_everything(self):
        self.assertEqual(self.select("pull_request")["packages"], ALL)

    def test_the_files_the_workflow_ignores_do_not_count(self):
        out = self.select("pull_request", "vision/vit.agc", "README.md", "site/content/x.txt")
        self.assertEqual(out["packages"], ["vision"])

    def test_the_vulkan_job_runs_for_transformers_only(self):
        self.assertEqual(self.select("pull_request", "transformers/qwen3.agc")["vulkan"], "true")
        # tensor/ brings transformers/ in (it depends on it), so a package that does not is the case that leaves the job out.
        self.assertEqual(self.select("pull_request", "vision/vit.agc")["vulkan"], "false")

    def test_the_arm_job_runs_for_transformers(self):
        out = self.select("pull_request", "transformers/qwen3.agc")
        # server/ depends on transformers/ directly, so the change runs its job beside transformers/ and vision/.
        self.assertEqual(out["packages"], ["server", "transformers", "vision"])
        self.assertEqual([out[k] for k in ("xla", "torch", "ggml_arm64", "vulkan")], ["false", "false", "true", "true"])


if __name__ == "__main__":
    unittest.main()
