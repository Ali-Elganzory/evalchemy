"""Tests for the two evalchemy features added in this change:

1. Universal response logging: ``--log_samples`` records prompts/outputs for
   *every* chat benchmark via the shared ``BaseBenchmark.compute`` funnel
   (previously HumanEval and similar benchmarks discarded their generations),
   and ``DCEvaluationTracker.save_results_samples`` persists them.
2. ``--max_tokens`` / ``--n_repeat`` are configurable through the CLI and are
   threaded down to every benchmark that supports them.

These tests use stdlib ``unittest`` only (no pytest dependency) but are named
``test_*`` so they are also discoverable by pytest.

Run from the evalchemy repo root with an env that has lm_eval installed, e.g.:

    PYTHONPATH=. python -m unittest discover -s tests -v
"""

import json
import os
import tempfile
import unittest

from lm_eval.api.instance import Instance

from eval.task import BaseBenchmark, TaskManager
from eval.eval import setup_custom_parser
from eval.eval_tracker import DCEvaluationTracker


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class FakeLM:
    """Minimal language model standing in for a real LM in ``compute``."""

    def __init__(self, world_size: int = 1, rank: int = 0):
        self.world_size = world_size
        self.rank = rank

    def generate_until(self, prompts):
        # Echo back a deterministic, identifiable completion per prompt.
        return [f"completion::{p[0]}" for p in (inst.args for inst in prompts)]


class _ConcreteBenchmark(BaseBenchmark):
    """A concrete BaseBenchmark so we can exercise the shared machinery."""

    def generate_responses(self, model):  # pragma: no cover - not used directly
        return {}

    def evaluate_responses(self, results):  # pragma: no cover - not used directly
        return {}


def _make_instances(prompts, docs=None):
    docs = docs or [{"answer": f"gold-{i}"} for i in range(len(prompts))]
    instances = []
    for idx, (prompt, doc) in enumerate(zip(prompts, docs)):
        instances.append(
            Instance(
                "generate_until",
                doc,
                (prompt, {"max_new_tokens": 123, "temperature": 0.0}),
                idx,
            )
        )
    return instances


class _RepeatBenchmark(BaseBenchmark):
    """Benchmark that hardcodes ``self.n_repeat`` like AIME24/HMMT/etc."""

    def __init__(self, logger=None, system_instruction=None):
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.n_repeat = 1

    def generate_responses(self, model):  # pragma: no cover
        return {}

    def evaluate_responses(self, results):  # pragma: no cover
        return {}


class _NoRepeatBenchmark(BaseBenchmark):
    """Benchmark with no ``n_repeat`` (e.g. HumanEval-style)."""

    def __init__(self, logger=None, system_instruction=None):
        super().__init__(logger=logger, system_instruction=system_instruction)

    def generate_responses(self, model):  # pragma: no cover
        return {}

    def evaluate_responses(self, results):  # pragma: no cover
        return {}


# --------------------------------------------------------------------------- #
# 1. Universal sample logging via BaseBenchmark.compute
# --------------------------------------------------------------------------- #
class TestComputeSampleLogging(unittest.TestCase):
    def test_records_samples_when_enabled(self):
        bench = _ConcreteBenchmark()
        bench.log_samples = True
        model = FakeLM()
        instances = _make_instances(["q0", "q1", "q2"])

        outputs = bench.compute(model, instances)

        self.assertEqual(outputs, ["completion::q0", "completion::q1", "completion::q2"])
        samples = bench.get_logged_samples()
        self.assertEqual(len(samples), 3)

        s0 = samples[0]
        # The output is captured for post-hoc analysis.
        self.assertEqual(s0["resps"], ["completion::q0"])
        self.assertEqual(s0["filtered_resps"], ["completion::q0"])
        self.assertEqual(s0["prompt"], "q0")
        self.assertEqual(s0["doc"], {"answer": "gold-0"})
        self.assertEqual(s0["target"], "gold-0")
        self.assertEqual(s0["doc_id"], 0)
        # task_name is stamped onto every instance by compute().
        self.assertEqual(s0["task_name"], "_Concrete")
        # Hash fields required by DCEvaluationTracker.save_results_aggregated.
        for key in ("doc_hash", "prompt_hash", "target_hash"):
            self.assertIn(key, s0)
            self.assertIsInstance(s0[key], str)
            self.assertTrue(s0[key])

    def test_no_samples_recorded_when_disabled(self):
        bench = _ConcreteBenchmark()
        self.assertFalse(bench.log_samples)  # default off
        bench.compute(FakeLM(), _make_instances(["a", "b"]))
        self.assertEqual(bench.get_logged_samples(), [])

    def test_humaneval_style_multi_call_accumulates(self):
        """HumanEval calls compute() once per language; samples must accumulate.

        This is exactly the case that previously produced no logged responses.
        """
        bench = _ConcreteBenchmark()
        bench.log_samples = True
        model = FakeLM()

        bench.compute(model, _make_instances(["py0", "py1"]))  # e.g. python
        bench.compute(model, _make_instances(["sh0"]))  # e.g. sh

        samples = bench.get_logged_samples()
        self.assertEqual(len(samples), 3)
        self.assertEqual(
            [s["prompt"] for s in samples],
            ["py0", "py1", "sh0"],
        )

    def test_non_primary_rank_does_not_record(self):
        bench = _ConcreteBenchmark()
        bench.log_samples = True
        # rank 1 with world_size 1 -> still rank != 0, must not record.
        bench.compute(FakeLM(world_size=1, rank=1), _make_instances(["x"]))
        self.assertEqual(bench.get_logged_samples(), [])


# --------------------------------------------------------------------------- #
# 2. TaskManager threads log_samples + n_repeat to every benchmark
# --------------------------------------------------------------------------- #
class TestTaskManagerOverrides(unittest.TestCase):
    def _manager(self, **kwargs):
        # task_list with a non-existent task makes _load_benchmarks skip every
        # real benchmark (fast, no heavy imports) while still running the real
        # __init__ path so benchmark_kwargs are wired exactly as in production.
        return TaskManager(task_list=["__none__"], **kwargs)

    def test_overrides_applied_to_repeat_benchmark(self):
        tm = self._manager(log_samples=True, n_repeat=7)
        tm._register_benchmark("RepeatBench", _RepeatBenchmark)

        inst = tm.get_benchmark("RepeatBench")
        self.assertIsNotNone(inst)
        self.assertTrue(inst.log_samples)
        self.assertEqual(inst.n_repeat, 7)

    def test_n_repeat_ignored_when_unsupported(self):
        tm = self._manager(log_samples=True, n_repeat=5)
        tm._register_benchmark("NoRepeatBench", _NoRepeatBenchmark)

        inst = tm.get_benchmark("NoRepeatBench")
        self.assertIsNotNone(inst)
        self.assertTrue(inst.log_samples)
        # No n_repeat attribute was created out of thin air.
        self.assertFalse(hasattr(inst, "n_repeat"))

    def test_defaults_preserved_when_flags_absent(self):
        tm = self._manager()  # no log_samples / n_repeat
        tm._register_benchmark("RepeatBench", _RepeatBenchmark)

        inst = tm.get_benchmark("RepeatBench")
        self.assertFalse(inst.log_samples)  # default
        self.assertEqual(inst.n_repeat, 1)  # benchmark's own default


# --------------------------------------------------------------------------- #
# 3. DCEvaluationTracker.save_results_samples writes per-task JSONL
# --------------------------------------------------------------------------- #
class TestSaveResultsSamples(unittest.TestCase):
    def test_writes_jsonl_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = DCEvaluationTracker(output_path=tmp, use_database=False)
            tracker.general_config_tracker.model_name = "org/test-model"
            tracker.general_config_tracker.model_name_sanitized = "org__test-model"
            tracker.date_id = "2026-05-18T00-00-00"

            samples = [
                {"doc_id": 0, "prompt": "q0", "resps": ["a0"], "target": "t0"},
                {"doc_id": 1, "prompt": "q1", "resps": ["a1"], "target": "t1"},
            ]
            tracker.save_results_samples(task_name="HumanEval", samples=samples)

            expected = os.path.join(
                tmp, "org__test-model", "samples_HumanEval_2026-05-18T00-00-00.jsonl"
            )
            self.assertTrue(os.path.exists(expected), f"missing {expected}")
            with open(expected) as f:
                lines = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]["prompt"], "q0")
            self.assertEqual(lines[1]["resps"], ["a1"])

    def test_no_output_path_is_noop(self):
        tracker = DCEvaluationTracker(output_path=None, use_database=False)
        # Should not raise.
        tracker.save_results_samples(task_name="AIME24", samples=[{"a": 1}])

    def test_logged_samples_hashes_are_aggregation_compatible(self):
        """End-to-end: samples from compute() must satisfy the hash contract
        used by save_results_aggregated (doc_hash/prompt_hash/target_hash)."""
        bench = _ConcreteBenchmark()
        bench.log_samples = True
        bench.compute(FakeLM(), _make_instances(["q0", "q1"]))
        samples = {"DemoTask": bench.get_logged_samples()}

        with tempfile.TemporaryDirectory() as tmp:
            tracker = DCEvaluationTracker(output_path=tmp, use_database=False)
            tracker.general_config_tracker.model_name = "m"
            tracker.general_config_tracker.model_name_sanitized = "m"
            results = {"results": {"DemoTask": {"acc": 1.0}}}
            # Must not raise KeyError on s["doc_hash"]+s["prompt_hash"]+...
            tracker.save_results_aggregated(results=results, samples=samples)
            self.assertIn("DemoTask", results["task_hashes"])


# --------------------------------------------------------------------------- #
# 4. CLI exposes --n_repeat and --max_tokens / --log_samples
# --------------------------------------------------------------------------- #
class TestCLIArgs(unittest.TestCase):
    def test_n_repeat_and_max_tokens_parsed(self):
        parser = setup_custom_parser()
        args = parser.parse_args(
            [
                "--tasks",
                "AIME24",
                "--max_tokens",
                "32768",
                "--n_repeat",
                "8",
                "--log_samples",
            ]
        )
        self.assertEqual(args.n_repeat, 8)
        self.assertEqual(args.max_tokens, "32768")
        self.assertTrue(args.log_samples)

    def test_n_repeat_defaults_to_none(self):
        parser = setup_custom_parser()
        args = parser.parse_args(["--tasks", "AIME24"])
        self.assertIsNone(args.n_repeat)
        self.assertFalse(args.log_samples)


if __name__ == "__main__":
    unittest.main()
