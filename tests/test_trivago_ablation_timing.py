"""CPU coverage of timed checkpoint recovery and Databricks notebook orchestration."""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from src import experiment_runner
from src.experiment_reporting import export_results

try:
    import torch
    from src import train_trivago_top5 as trainer
except (ImportError, OSError):
    torch = None

if __package__:
    from .test_trivago_top5_all_events import fixture
else:
    from test_trivago_top5_all_events import fixture


@unittest.skipIf(torch is None, "PyTorch unavailable")
class AblationTimingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run_dir = self.root / "run"
        self.latest = self.run_dir / "result_latest.pth"
        self.best = self.run_dir / "result_best.pth"
        self.history = self.run_dir / "metric_history.json"
        self.status = self.run_dir / "run_status.json"
        self.prepared = self.make_prepared(18)

    def make_prepared(self, fields, folder=None):
        folder = folder or self.root / str(fields)
        folder.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            prepared, config = fixture(folder, full_metadata=True, field_budget=fields - 3,
                                       action_scope="item_event", with_prices=True)
        self.assertEqual(len(config["field_names"]), fields)
        return prepared

    @staticmethod
    def invoke(*args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(sys, "argv", ["trainer", *args]), \
                patch.object(torch.cuda, "is_available", return_value=False):
            trainer.main()
        return output.getvalue()

    def train(self, *args, output=None, prepared=None):
        return self.invoke("--prepared-dir", str(prepared or self.prepared),
                           "--output-dir", str(output or self.run_dir), "--epochs", "2", "--batch-size", "1",
                           "--emb-size", "4", "--num-heads", "2", "--sample-order", "2",
                           "--metadata-projection", "fieldwise", "--checkpoint-every", "1", "--max-hours", "0", *args)

    def test_mid_epoch_resume_and_untimed_compatibility(self):
        plain = self.root / "plain"
        self.train("--max-batches", "2", output=plain)
        legacy_path = plain / self.latest.name
        legacy_checkpoint = legacy_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "fresh timed run directory"):
            self.train("--resume", "--track-time", output=plain)
        self.assertEqual(legacy_path.read_bytes(), legacy_checkpoint)
        with patch.object(trainer, "timed_now", side_effect=AssertionError("Untimed run used clock")):
            self.train("--resume", output=plain)
        untimed = trainer.load_checkpoint(legacy_path)
        self.assertNotIn("timing_state", untimed)
        self.assertNotIn("timing_environment", untimed)

        with patch.dict(os.environ, DATABRICKS_RUNTIME_VERSION="fixture-a"):
            self.train("--track-time", "--max-batches", "2")
        partial = trainer.load_checkpoint(self.latest)
        self.assertEqual((partial["epoch"], partial["global_step"], partial["next_offset"]), (0, 2, 2))
        first_time = partial["timing_state"]["train_time_seconds"]
        self.assertGreater(first_time, 0)
        saved = self.latest.read_bytes()
        with self.assertRaisesRegex(ValueError, "Keep --track-time"):
            self.train("--resume")
        self.assertEqual(self.latest.read_bytes(), saved)

        rebuilt = self.make_prepared(18, self.root / "rebuilt")
        with patch.dict(os.environ, DATABRICKS_RUNTIME_VERSION="fixture-b"):
            output = self.train("--resume", "--track-time", "--max-batches", "4", prepared=rebuilt)
        partial = trainer.load_checkpoint(self.latest)
        self.assertEqual((partial["epoch"], partial["global_step"], partial["next_offset"]), (0, 4, 4))
        self.assertIn("Timing environment changed", output)
        self.assertEqual([entry["databricks_runtime"] for entry in partial["timing_environment"]],
                         ["fixture-a", "fixture-b"])
        resumed_time = partial["timing_state"]["train_time_seconds"]
        self.assertGreater(resumed_time, first_time)
        self.train("--resume", "--track-time")
        timed = trainer.load_checkpoint(self.latest)
        self.assertEqual(timed["timing_state"], dict(train_time_seconds=0, val_retry_seconds=0))
        rows = trainer.timed_epochs(timed["metric_history"], 2)
        self.assertGreater(rows[0]["train_time_seconds"], resumed_time)
        self.assertGreater(rows[1]["cumulative_val_time_seconds"], rows[0]["cumulative_val_time_seconds"])
        self.assertEqual([{k: v for k, v in row.items() if k not in trainer.EPOCH_TIMING_FIELDS}
                          for row in rows], untimed["metric_history"])
        for key in ("epoch", "next_offset", "global_step", "best_epoch", "best_ndcg_at_5",
                    "scheduler", "early_stopping", "training_policy", "model_arguments"):
            self.assertEqual(timed[key], untimed[key], key)
        for key in timed["model"]:
            torch.testing.assert_close(timed["model"][key], untimed["model"][key], rtol=0, atol=0)

    def test_validation_retry_resumes_at_end_of_train(self):
        with patch.object(trainer, "evaluate", side_effect=trainer.TimeBudgetExceeded):
            self.train("--track-time", "--epochs", "1")
        checkpoint = trainer.load_checkpoint(self.latest)
        self.assertEqual((checkpoint["epoch"], checkpoint["next_offset"], checkpoint["global_step"]), (0, 6, 6))
        retry = checkpoint["timing_state"]["val_retry_seconds"]
        self.assertGreater(retry, 0)
        self.assertEqual(trainer.read_json(self.status, dict)["status"], "PAUSED_DURING_VAL")
        original_forward = trainer.forward

        def evaluation_only(model, batch):
            self.assertFalse(model.training, "Resume repeated training batches")
            return original_forward(model, batch)

        with patch.object(trainer, "forward", side_effect=evaluation_only):
            self.train("--resume", "--track-time", "--epochs", "1")
        row, = trainer.read_json(self.history, list)
        self.assertEqual(row["val_retry_seconds"], retry)
        self.assertEqual(row["train_time_seconds"], checkpoint["timing_state"]["train_time_seconds"])
        self.assertEqual(row["epoch_time_seconds"], row["train_time_seconds"] + row["val_time_seconds"])
        self.assertGreater(row["val_time_seconds"], 0)

    def test_training_time_budget_pause_keeps_timing_and_cursor(self):
        with patch.object(trainer, "check_time", side_effect=trainer.TimeBudgetExceeded):
            self.train("--track-time", "--checkpoint-every", "1000")
        checkpoint = trainer.load_checkpoint(self.latest)
        self.assertEqual((checkpoint["epoch"], checkpoint["next_offset"], checkpoint["global_step"]), (0, 1, 1))
        self.assertGreater(checkpoint["timing_state"]["train_time_seconds"], 0)
        self.assertEqual(trainer.read_json(self.status, dict)["status"], "PAUSED")

    def test_recovers_interrupted_best_publication_and_stale_reports(self):
        original_publish = trainer.publish_best
        val = dict(recall_at_5=.5, examples=2, ce_loss=1., metadata_loss=.2, total_loss=1.2)

        def publish_first(latest, best):
            if not best.exists():
                return original_publish(latest, best)
            raise OSError("interrupted before replacing best")

        with patch.object(trainer, "evaluate", side_effect=[dict(val, ndcg_at_5=.3), dict(val, ndcg_at_5=.8)]), \
                patch.object(trainer, "publish_best", side_effect=publish_first):
            with self.assertRaisesRegex(OSError, "interrupted"):
                self.train("--track-time")
        self.assertEqual(trainer.load_checkpoint(self.best)["epoch"], 1)
        self.assertEqual(trainer.load_checkpoint(self.latest)["epoch"], 2)
        self.assertEqual(len(trainer.read_json(self.history, list)), 1)
        self.status.unlink()
        with patch.object(trainer, "evaluate", side_effect=AssertionError("Completed run evaluated")):
            self.train("--resume", "--track-time")
        self.assertEqual(self.best.read_bytes(), self.latest.read_bytes())
        self.assertEqual(len(trainer.read_json(self.history, list)), 2)
        self.assertEqual(trainer.read_json(self.status, dict)["status"], "COMPLETE")

    def test_best_only_corrupt_checkpoint_and_signature_never_restart(self):
        self.train("--track-time", "--epochs", "1")
        self.latest.unlink()
        best_content = self.best.read_bytes()
        for options in ((), ("--resume",), ("--eval-only", "test")):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "checkpoint|run state"):
                self.train("--track-time", *options)
        self.assertEqual(self.best.read_bytes(), best_content)
        self.latest.write_bytes(b"invalid payload")
        with self.assertRaisesRegex(ValueError, "Cannot load checkpoint"):
            self.train("--resume", "--track-time")
        self.assertEqual(self.latest.read_bytes(), b"invalid payload")
        self.latest.write_bytes(best_content)
        checkpoint = trainer.load_checkpoint(self.latest)
        checkpoint["signature"] = ()
        trainer.atomic_save(checkpoint, self.latest)
        bad_signature = self.latest.read_bytes()
        with self.assertRaisesRegex(ValueError, "does not match prepared data"):
            self.train("--resume", "--track-time")
        self.assertEqual(self.latest.read_bytes(), bad_signature)

    def test_bad_timing_history_and_training_policy_are_rejected(self):
        self.train("--track-time", "--max-batches", "7")
        saved = self.latest.read_bytes()
        with self.assertRaisesRegex(ValueError, "Resume training policy differs"):
            self.train("--track-time", "--resume", "--lr-factor", ".7")
        self.assertEqual(self.latest.read_bytes(), saved)
        for value in (None, -1, float("inf"), float("nan")):
            checkpoint = trainer.load_checkpoint(self.latest)
            checkpoint["metric_history"][0]["train_time_seconds"] = value
            trainer.atomic_save(checkpoint, self.latest)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "invalid timing"):
                self.train("--track-time", "--resume")
        checkpoint["metric_history"].append(checkpoint["metric_history"][0])
        trainer.atomic_save(checkpoint, self.latest)
        with self.assertRaisesRegex(ValueError, "consecutive completed epochs"):
            self.train("--track-time", "--resume")

    def test_atomic_write_failure_preserves_previous_checkpoint(self):
        self.train("--track-time", "--max-batches", "2")
        saved = self.latest.read_bytes()

        def fail_save(payload, temporary):
            temporary.write_bytes(b"partial")
            raise OSError("disk full")

        with patch.object(torch, "save", side_effect=fail_save), \
                self.assertRaisesRegex(RuntimeError, "free space"):
            self.train("--track-time", "--resume", "--max-batches", "4")
        self.assertEqual(self.latest.read_bytes(), saved)
        self.assertEqual(self.latest.with_suffix(".pth.tmp").read_bytes(), b"partial")
        self.train("--track-time", "--resume", "--max-batches", "4")
        self.assertEqual(trainer.load_checkpoint(self.latest)["global_step"], 4)

    def test_output_lock_rejects_a_second_writer(self):
        with trainer.lock_output(self.run_dir):
            with self.assertRaisesRegex(RuntimeError, "Cannot lock"):
                self.train("--track-time", "--max-batches", "2")
        self.train("--track-time", "--max-batches", "2")
        self.assertTrue(self.latest.is_file())

    def test_forward_timing_weights_the_short_final_batch_by_examples(self):
        model = torch.nn.Linear(1, 1)
        model.action_mode = "none"

        def forward(model, batch):
            return torch.ones(len(batch), dtype=torch.long), torch.zeros(len(batch), 3), 0

        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(trainer, "batch_at", side_effect=lambda data, indices: indices), \
                patch.object(trainer, "forward", side_effect=forward), \
                patch.object(trainer, "timed_now", side_effect=[1, 1.5, 5, 6.5]):
            metrics = trainer.evaluate(model, list(range(3)), [[1, 2]] * 3, 2, track_inference=True)
        self.assertEqual(metrics["examples"], 3)
        self.assertEqual(metrics["inference_total_seconds"], 2)
        self.assertEqual(metrics["inference_samples_per_second"], 1.5)
        self.assertEqual(metrics["inference_ms_per_sample"], 2000 / 3)
        self.assertEqual((metrics["ndcg_at_5"], metrics["recall_at_5"]), (1, 1))

    def test_test_evaluates_the_best_model_without_writing_checkpoints(self):
        val = dict(recall_at_5=.5, examples=2, ce_loss=1., metadata_loss=.2, total_loss=1.2)
        with patch.object(trainer, "evaluate", side_effect=[dict(val, ndcg_at_5=.8), dict(val, ndcg_at_5=.3)]):
            self.train("--track-time")
        best = trainer.load_checkpoint(self.best)
        self.assertEqual(best["epoch"], 1)
        self.assertEqual(trainer.load_checkpoint(self.latest)["epoch"], 2)
        before = [path.read_bytes() for path in (self.latest, self.best)]
        original_status = self.status.read_bytes()
        validation_output = self.train("--track-time", "--eval-only", "val")
        self.assertIn('"val_time_seconds"', validation_output)
        self.assertNotIn('"test_time_seconds"', validation_output)
        self.assertEqual(self.status.read_bytes(), original_status)
        evaluate = trainer.evaluate

        def check_model(model, *args, **kwargs):
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, best["model"][name], atol=0, rtol=0)
            metrics = evaluate(model, *args)
            timed = evaluate(model, *args, **kwargs)
            self.assertEqual(metrics, {key: timed[key] for key in metrics})
            return timed

        with patch.object(trainer, "evaluate", side_effect=check_model):
            self.train("--track-time", "--eval-only", "test")
        status = trainer.read_json(self.status, dict)
        self.assertEqual((status["best_epoch"], status["epochs"]), (1, 2))
        self.assertTrue(trainer.valid_test_status(status, True))
        self.assertEqual([path.read_bytes() for path in (self.latest, self.best)], before)

    def notebook(self, prepared):
        path = Path(__file__).resolve().parents[1] / "databricks_mleup_ablation.py"
        cells = path.read_text(encoding="utf-8").split("# COMMAND ----------")[1:]
        namespace = {}
        exec(compile(cells[0], str(path), "exec"), namespace)
        namespace.update(RUNS_DIR=self.root, prepared=prepared)
        namespace["TRAINING"].update(epochs=1, batch_size=1, emb_size=4, num_heads=2, sample_order=2,
                                     checkpoint_every=1, max_hours=0)
        exec(compile(cells[3], str(path), "exec"), namespace)
        return namespace

    def test_notebook_runs_all_four_variants_and_exports_consistent_csv(self):
        notebook = self.notebook({18: self.prepared, 53: self.make_prepared(53)})
        experiments = notebook["EXPERIMENTS"]
        models = list(experiments)
        commands = []

        def run_script(repo, *args):
            commands.append(args)
            self.invoke(*args[args.index("--prepared-dir"):])

        run = notebook["run_experiment"]
        completed = self.root / models[0]
        with contextlib.redirect_stdout(io.StringIO()), patch.object(experiment_runner, "run_script") as process:
            process.side_effect = lambda repo, *args: run_script(repo, *args, "--max-batches", "2")
            run(models[0])
            self.assertFalse((completed / self.best.name).exists())
            (completed / self.status.name).write_text("{", encoding="utf-8")
            process.side_effect = run_script
            for model in models:
                run(model)
            self.assertIn("--resume", commands[1])
            self.assertIn("--eval-only", commands[2])
            self.assertTrue(all("--track-time" in cmd for cmd in commands))
            self.assertEqual(Path(commands[0][commands[0].index("--output-dir") + 1]), completed)
            checkpoint_bytes = {path: path.read_bytes() for path in completed.glob("*.pth")}
            commands.clear()
            run(models[0])
            self.assertEqual(len(commands), 1)
            self.assertIn("--resume", commands[0])
            status = trainer.read_json(completed / self.status.name, dict)
            status["best_checkpoint_id"] = "stale-result"
            trainer.atomic_json(completed / self.status.name, status)
            run(models[0])
            self.assertIn("--eval-only", commands[-1])
            for path, content in checkpoint_bytes.items():
                self.assertEqual(path.read_bytes(), content)
            report, summary, _, _ = export_results(self.root, experiments)
            self.assertEqual(list(summary["model"]), models)
            for model in models:
                status = trainer.read_json(self.root / model / self.status.name, dict)
                self.assertTrue(trainer.valid_test_status(status, True))
                self.assertEqual(status["metrics"]["examples"], 2)
                self.assertEqual(report["experiments"][model]["hyperparameters"]["batch_size"], 1)
            timeline = pd.read_csv(self.root / "epoch_timing.csv")
            self.assertEqual(len(timeline), 4)
            first = timeline.iloc[0]
            self.assertAlmostEqual(summary.iloc[0]["time_to_best_min"],
                                   (first["epoch_time_seconds"] + first["val_retry_seconds"]) / 60)
            for model in models:
                (self.root / model / self.status.name).unlink()
            self.assertTrue(export_results(self.root, experiments)[1].empty)
            self.assertTrue(pd.read_csv(self.root / "experiment_summary.csv").empty)
            self.assertEqual(len(pd.read_csv(self.root / "epoch_timing.csv")), 4)

    def test_untimed_summary_preserves_accuracy_columns(self):
        run = self.root / "02_baseline_18"
        self.train("--epochs", "1", output=run)
        self.train("--epochs", "1", "--eval-only", "test", output=run)
        with contextlib.redirect_stdout(io.StringIO()):
            summary = export_results(self.root, self.notebook({})["EXPERIMENTS"])[1]
        self.assertEqual(list(summary["model"]), ["02_baseline_18"])
        self.assertIn("ndcg_at_5", summary.columns)
        self.assertTrue(summary["test_time_min"].isna().all())


if __name__ == "__main__":
    unittest.main()
