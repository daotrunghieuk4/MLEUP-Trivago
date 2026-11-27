"""Verify configuration, shared preprocessing and the command boundary."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import experiment_runner as runner
from src import run_metadata

if __package__:
    from .test_experiment_reporting import notebook_configuration
    from .test_trivago_top5_all_events import fixture
else:
    from test_experiment_reporting import notebook_configuration
    from test_trivago_top5_all_events import fixture


class ExperimentRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.notebook = notebook_configuration()
        self.experiments = self.notebook["EXPERIMENTS"]
        self.training = self.notebook["TRAINING"]

    def test_command_matches_pre_refactor_effective_settings(self):
        defaults = {
            "--epochs": "50", "--batch-size": "128", "--emb-size": "16", "--num-heads": "4",
            "--sample-order": "3", "--si-dc": "2.0", "--beta": "0.3", "--alpha": "0.2",
            "--lr": "0.001", "--lr-factor": "0.5", "--lr-patience": "1", "--min-lr": "1e-05",
            "--patience": "5", "--min-delta": "0.0001", "--loss-patience": "50",
            "--loss-min-delta": "0.001", "--metadata-projection": "fieldwise",
            "--checkpoint-every": "1000", "--max-hours": "8.5",
        }
        fields_modes = [(53, "none", "off"), (18, "none", "off"), (18, "event_graph_candidate", "off"),
                        (18, "event_graph_candidate", "relative_mlp_v1")]
        self.assertEqual(self.notebook["SEED"], 2023)
        self.assertEqual(self.notebook["RUNS_DIR"].name, "seed2023_e50_b128_timed_v1")
        runner.validate_configuration(self.experiments, self.training, self.root)
        for (name, experiment), (fields, action, price) in zip(self.experiments.items(), fields_modes, strict=True):
            prepared = self.root / f"prepared_{fields}"
            output = self.root / name
            command = runner.build_train_command(experiment, self.training, prepared, output)
            tokens = command[command.index("--prepared-dir"):-1]
            self.assertEqual(command[-1], "--track-time")
            self.assertEqual(dict(zip(tokens[::2], tokens[1::2], strict=True)), {
                **defaults, "--prepared-dir": str(prepared), "--output-dir": str(output),
                "--action-mode": action, "--price-mode": price,
            })
            self.assertEqual(experiment["fields"], fields)

    def test_invalid_settings_and_output_alias_are_rejected(self):
        for settings in (dict(epochs=0), dict(batch_size=0), dict(checkpoint_every=0),
                         dict(emb_size=15), dict(lr=float("nan")), dict(min_lr=.1)):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                runner.validate_configuration(self.experiments, self.training | settings, self.root)
        experiment = next(iter(self.experiments.values()))
        for name, settings in (("../escape", experiment), ("price", experiment | {"price_mode": "relative_mlp_v1"}),
                               ("action", experiment | {"action_mode": "unknown"})):
            with self.subTest(name=name), self.assertRaises(ValueError):
                runner.validate_configuration({name: settings}, self.training, self.root)

    def test_shared_prepared_datasets_reuse_and_verify_artifacts(self):
        from src.trivago_data_parts import pack

        raw = self.root / "original"
        raw.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            fixture(raw, full_metadata=True, field_budget=50, action_scope="item_event", with_prices=True)
        workspace, scratch = self.root / "workspace", self.root / "scratch"
        workspace.mkdir()
        scratch.mkdir()
        pack(raw, workspace / "data_parts", 1)
        actual_repo = Path(__file__).resolve().parents[1]
        execute = runner.run_script

        def prepare():
            return runner.prepare_datasets(workspace, raw, self.root / "unneeded", scratch,
                                           self.experiments, self.notebook["SEED"])

        with contextlib.redirect_stdout(io.StringIO()), patch.object(runner, "check_repo"), \
                patch.object(runner, "unpack", wraps=runner.unpack) as unpack, \
                patch.object(runner, "run_script", side_effect=lambda repo, *args: execute(actual_repo, *args)) as run:
            prepared = prepare()
            self.assertEqual(list(prepared), [53, 18])
            self.assertEqual(prepare(), prepared)
            self.assertEqual(run.call_count, 2)
            unpack.assert_called_once()
            candidate = prepared[18] / "test_candidates.txt"
            candidate.write_bytes(b"corrupted artifact")
            with self.assertRaisesRegex(ValueError, "SHA-256.*test_candidates.txt"):
                prepare()
            self.assertEqual(candidate.read_bytes(), b"corrupted artifact")
            self.assertEqual(run.call_count, 2)

    def test_missing_latest_is_flagged_without_loading_weights(self):
        name = next(iter(self.experiments))
        output = self.root / name
        output.mkdir()
        (output / "result_best.pth").write_bytes(b"not a checkpoint")
        self.assertEqual(runner.inspect_run_status(self.root, self.experiments)[0]["status"], "CHECK_STATUS")
        with patch.object(runner, "run_script") as process, self.assertRaisesRegex(RuntimeError, "result_latest"):
            runner.run_experiment(name, repo=self.root, runs_dir=self.root, experiments=self.experiments,
                                  training=self.training, prepared={}, seed=2023)
        process.assert_not_called()

    def test_atomic_replace_retries_a_windows_sharing_error(self):
        pending, final = self.root / "pending", self.root / "final"
        pending.write_bytes(b"complete payload")
        final.write_bytes(b"prior payload")
        sharing = PermissionError("temporarily locked")
        sharing.winerror = 32
        replace = run_metadata.os.replace

        def locked_once(source, target):
            if operation.call_count == 1:
                raise sharing
            replace(source, target)

        with patch.object(run_metadata.os, "replace", side_effect=locked_once) as operation, \
                patch.object(run_metadata.time, "sleep") as sleep:
            run_metadata.replace_file(pending, final)
        sleep.assert_called_once()
        self.assertEqual(final.read_bytes(), b"complete payload")


if __name__ == "__main__":
    unittest.main()
