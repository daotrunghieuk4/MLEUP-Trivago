"""Reporting fixtures use JSON and tiny checkpoint placeholders, without PyTorch."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import experiment_reporting as reporting
from src.run_metadata import atomic_json, file_reference


def notebook_cells():
    path = Path(__file__).resolve().parents[1] / "databricks_mleup_ablation.py"
    return [compile(cell, str(path), "exec") for cell in
            path.read_text(encoding="utf-8").split("# COMMAND ----------")[1:]]


def notebook_configuration():
    configuration = {}
    exec(notebook_cells()[0], configuration)
    return configuration


class ExperimentReportingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.experiments = notebook_configuration()["EXPERIMENTS"]
        self.names = list(self.experiments)

    def save_run(self, index, label="EVALUATED", *, timed=True, ndcg=.6, recall=.8, device="cpu",
                 seconds=60, legacy=False):
        name = self.names[index]
        experiment = self.experiments[name]
        folder = self.root / name
        folder.mkdir(exist_ok=True)
        latest, best = (folder / f"result_{kind}.pth" for kind in ("latest", "best"))
        latest.write_bytes(b"latest-known-epoch-2")
        best.write_bytes(b"best-known-epoch-1")
        history = []
        for epoch, score in [(1, .7), (2, .65)]:
            row = dict(epoch=epoch, learning_rate=.001 / epoch, next_learning_rate=.001 / epoch,
                       train=dict(ce_loss=epoch, metadata_loss=.2, total_loss=epoch + .2),
                       val=dict(ce_loss=2 * epoch, metadata_loss=.3, total_loss=2 * epoch + .3,
                                ndcg_at_5=score, recall_at_5=score + .1))
            if timed:
                row.update(train_time_seconds=seconds, val_time_seconds=15, epoch_time_seconds=seconds + 15,
                           cumulative_train_time_seconds=seconds * epoch, cumulative_val_time_seconds=15 * epoch,
                           val_retry_seconds=5 if epoch == 1 else 10)
            history.append(row)
        history_path = folder / "metric_history.json"
        atomic_json(history_path, history)
        environment = dict(device=device, pytorch="fixture", cuda=None, batch_size=96, databricks_runtime=None)
        finished = label in ("COMPLETE", "EARLY_STOPPED", "EVALUATED")
        status = dict(status=label, epochs=2, global_step=100, selection_metric="ndcg_at_5",
                      training_finished=finished, best_epoch=1, best_val_ndcg_at_5=.7,
                      early_stopping=dict(stopped=label == "EARLY_STOPPED", reason="patience" if label == "EARLY_STOPPED" else None))
        if not legacy:
            status.update(
                report_metadata_version=1,
                metric_history_sha256=hashlib.sha256(history_path.read_bytes()).hexdigest(),
                checkpoint_files=dict(best=file_reference(best), latest=file_reference(latest),
                                      best_identifier=f"best-{name}" if timed else None),
                run_metadata=dict(hyperparameters=dict(batch_size=96, track_time=timed, seed=2023),
                                  model_configuration=dict(metadata_fields=experiment["fields"],
                                                           action_mode=experiment["action_mode"],
                                                           price_mode=experiment["price_mode"]),
                                  reproducibility=dict(git_commit_sha=None)),
            )
        if timed:
            status["timing_environment"] = [environment]
        if label == "EVALUATED":
            status.update(split="test", evaluated_checkpoint=str(best), test_environment=environment,
                          metrics=dict(ndcg_at_5=ndcg, recall_at_5=recall, examples=100))
            if timed:
                status.update(best_checkpoint_id=f"best-{name}")
                status["metrics"].update(test_time_seconds=12, inference_total_seconds=10,
                                         inference_ms_per_sample=100, inference_samples_per_second=10)
        atomic_json(folder / "run_status.json", status)
        return folder, history, status

    def export(self):
        return reporting.export_results(self.root, self.experiments)

    def test_four_statuses_and_best_epoch_distinct_from_total(self):
        folder, _, _ = self.save_run(0, "EVALUATED")
        self.save_run(1, "EARLY_STOPPED")
        self.save_run(2, "RUNNING")
        report, summary, timeline, ablation = self.export()
        results = report["experiments"]
        self.assertEqual(list(results), self.names)
        self.assertEqual([row["status"] for row in results.values()],
                         ["EVALUATED", "EARLY_STOPPED", "RUNNING", "NOT_STARTED"])
        result = results[self.names[0]]
        self.assertEqual(result["hyperparameters"]["batch_size"], 96)
        self.assertEqual(result["training_summary"]["epochs_completed"], 2)
        self.assertEqual(result["best_epoch"]["epoch"], 1)
        self.assertEqual(result["best_epoch"]["train"]["total_loss"], 1.2)
        self.assertEqual(result["best_epoch"]["validation"]["total_loss"], 2.3)
        self.assertEqual(result["best_epoch"]["validation"]["ndcg_at_5"], .7)
        self.assertEqual(result["efficiency"]["total_train_time_seconds"], 120)
        self.assertEqual(result["efficiency"]["total_val_retry_seconds"], 15)
        self.assertEqual(result["best_epoch"]["timing"]["time_to_best_seconds"], 80)
        self.assertEqual(result["checkpoint"]["best_checkpoint_identifier"], f"best-{self.names[0]}")
        self.assertIsNone(result["reproducibility"]["git_commit_sha"])
        self.assertEqual(list(summary["model"]), self.names[:1])
        self.assertEqual(summary.iloc[0]["avg_train_time_min"], 1)
        self.assertAlmostEqual(summary.iloc[0]["time_to_best_min"], 80 / 60)
        self.assertEqual(len(timeline), 6)
        self.assertTrue(ablation.empty)
        self.assertIsNone(results[self.names[2]]["test_results"])
        self.assertEqual(results[self.names[2]]["best_epoch"]["epoch"], 1)
        self.assertIsNone(results[self.names[3]]["hyperparameters"]["seed"])
        self.assertIsNone(results[self.names[3]]["test_results"])
        self.assertEqual(sorted(path.name for path in folder.glob("*.pth")), ["result_best.pth", "result_latest.pth"])
        self.assertEqual(json.loads((self.root / "best_experiments.json").read_text(encoding="utf-8")), report)

    def test_ties_ablation_deltas_zero_denominators_and_environment(self):
        self.save_run(0, ndcg=0, recall=.4, seconds=0)
        self.save_run(1, ndcg=.5, recall=.6, seconds=60)
        self.save_run(2, ndcg=.75, recall=.6, seconds=90)
        self.save_run(3, ndcg=.75, recall=.8, seconds=45, device="fixture-GPU")
        report, summary, _, ablation = self.export()
        self.assertEqual(list(summary.columns), list(reporting.SUMMARY_COLUMNS))
        self.assertEqual(list(ablation.columns), list(reporting.ABLATION_COLUMNS))
        a, b, c = report["ablation_comparison"]
        self.assertEqual(a["ndcg_absolute_change"], .5)
        self.assertIsNone(a["ndcg_relative_change_pct"])
        self.assertIsNone(a["train_time_change_pct"])
        self.assertEqual(b["train_time_before_min"], 1)
        self.assertEqual(b["train_time_after_min"], 1.5)
        self.assertEqual(b["train_time_change_pct"], 50)
        self.assertEqual(b["ndcg_relative_change_pct"], 50)
        self.assertIsNone(c["inference_change_pct"])
        self.assertTrue(any("environments differ" in warning for warning in report["warnings"]))
        self.assertEqual(report["comparison"]["highest_test_ndcg_at_5"],
                         dict(experiment_ids=self.names[2:], value=.75))

    def test_legacy_untimed_and_incomplete_timing_keep_accuracy(self):
        self.save_run(0, timed=False, legacy=True)
        folder, history, status = self.save_run(1)
        history[-1].pop("train_time_seconds")
        history[-1]["val"]["total_loss"] = float("nan")
        status["metrics"]["inference_samples_per_second"] = float("inf")
        self.rewrite_history(folder, history, status)
        report, summary, _, _ = self.export()
        self.assertEqual(list(summary["model"]), self.names[:2])
        self.assertTrue(summary["avg_epoch_time_min"].isna().all())
        self.assertTrue(summary["inference_samples_per_second"].isna().all())
        self.assertTrue(any("Timing totals" in text for text in report["warnings"]))
        self.assertTrue(any("legacy status" in text for text in report["warnings"]))
        raw = (self.root / "best_experiments.json").read_text(encoding="utf-8")
        self.assertNotIn("NaN", raw)
        self.assertNotIn("Infinity", raw)
        self.assertIsNone(report["experiments"][self.names[0]]["hyperparameters"]["batch_size"])

    def rewrite_history(self, folder, history, status):
        path = folder / "metric_history.json"
        atomic_json(path, history)
        status["metric_history_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        atomic_json(folder / "run_status.json", status)

    def test_stale_mismatched_and_invalid_artifacts_do_not_publish_results(self):
        for index, fault in enumerate(("wrong-best", "duplicate-epoch", "bad-json", "checkpoint-replaced")):
            folder, history, status = self.save_run(index)
            if fault == "wrong-best":
                status["best_epoch"] = 2
                atomic_json(folder / "run_status.json", status)
            elif fault == "duplicate-epoch":
                history[-1]["epoch"] = 1
                self.rewrite_history(folder, history, status)
            elif fault == "bad-json":
                (folder / "metric_history.json").write_text("{", encoding="utf-8")
            elif fault == "checkpoint-replaced":
                (folder / "result_best.pth").write_bytes(b"unrelated")
        report, summary, _, _ = self.export()
        self.assertTrue(summary.empty)
        self.assertTrue(all(item["status"] == "CHECK_STATUS" for item in report["experiments"].values()))
        self.assertTrue(all(item["test_results"] is None for item in report["experiments"].values()))

    def test_later_report_keeps_completed_models_and_atomic_json(self):
        self.save_run(0)
        self.save_run(1, "COMPLETE")
        first, _, _, _ = self.export()
        self.assertEqual(first["experiments"][self.names[1]]["status"], "COMPLETE")
        self.assertIsNone(first["experiments"][self.names[1]]["test_results"])
        self.save_run(1)
        report, _, _, _ = self.export()
        self.assertEqual(first["experiments"][self.names[0]], report["experiments"][self.names[0]])
        target = self.root / "best_experiments.json"
        previous = target.read_bytes()
        with patch.object(reporting.json, "dump", side_effect=OSError("write interrupted")):
            with self.assertRaisesRegex(OSError, "interrupted"):
                self.export()
        self.assertEqual(target.read_bytes(), previous)

    def test_configuration_and_summary_cells_alone(self):
        self.save_run(2, "PAUSED")
        cells, shown = notebook_cells(), []
        self.assertEqual(len(cells), 9)
        namespace = {"display": shown.append}
        exec(cells[0], namespace)
        namespace["RUNS_DIR"] = self.root
        with patch.dict("sys.modules", {"torch": None, "src.model": None}):
            exec(cells[-1], namespace)
        self.assertEqual(len(shown), 3)
        self.assertEqual(namespace["report"]["experiments"][self.names[2]]["status"], "PAUSED")
        self.assertNotIn("prepared", namespace)


if __name__ == "__main__":
    unittest.main()
