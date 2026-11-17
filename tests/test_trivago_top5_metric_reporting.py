"""Validation NDCG selection, dual stopping, loss reports and safe resume."""
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch
from trivago_top5_checkpoint import new_early_stopping, update_early_stopping


class DualStoppingPolicyTests(unittest.TestCase):
    def test_exact_relative_loss_threshold_counts_at_different_scales(self):
        for scale in (1e-9, .01, 1., 1000., 1e9):
            with self.subTest(scale=scale):
                state = new_early_stopping()
                train = val = scale
                update_early_stopping(state, .6, train, val)
                for epoch in range(1, 4):
                    train *= .999
                    val *= 1.001
                    stopped = update_early_stopping(state, .6 + .01 * epoch, train, val)
                    self.assertEqual(state['loss_bad_epochs'], epoch)
                    self.assertEqual(stopped, epoch == 3)
                self.assertEqual(state['reason'], 'ce_loss_divergence')

    def test_below_threshold_on_either_side_breaks_loss_streak(self):
        for train_delta, val_delta in ((.0009999, .002), (.002, .0009999),
                                      (.00001, .00001), (0., 0.)):
            with self.subTest(train_delta=train_delta, val_delta=val_delta):
                state = new_early_stopping()
                update_early_stopping(state, .6, 1., 1.)
                update_early_stopping(state, .61, .998, 1.002)
                self.assertEqual(state['loss_bad_epochs'], 1)
                self.assertFalse(update_early_stopping(
                    state, .62, .998 * (1 - train_delta), 1.002 * (1 + val_delta)))
                self.assertEqual(state['loss_bad_epochs'], 0)

    def test_loss_can_stop_while_ndcg_keeps_improving(self):
        state = new_early_stopping()
        for train, val in [(4., 1.), (3., 2.), (2., 3.)]:
            self.assertFalse(update_early_stopping(state, val / 10, train, val))
        self.assertTrue(update_early_stopping(state, .4, 1., 4.))
        self.assertEqual(state['reason'], 'ce_loss_divergence')
        self.assertEqual(state['ndcg_bad_epochs'], 0)

    def test_ndcg_priority_when_both_trigger_and_ties_count_bad(self):
        state = new_early_stopping()
        update_early_stopping(state, .6, 4., 1.)
        for train, val in [(3., 2.), (2., 3.), (1., 4.)]:
            update_early_stopping(state, .6, train, val)
        self.assertEqual(state['reason'], 'val_ndcg_no_improvement')
        self.assertEqual(state['triggered'], ['val_ndcg_no_improvement', 'ce_loss_divergence'])

    def test_improvement_resets_ndcg_and_broken_trend_resets_loss(self):
        state = new_early_stopping()
        update_early_stopping(state, .6, 4., 1.)
        update_early_stopping(state, .6, 3., 2.)
        self.assertEqual((state['ndcg_bad_epochs'], state['loss_bad_epochs']), (1, 1))
        update_early_stopping(state, .7, 3., 2.)
        self.assertEqual((state['ndcg_bad_epochs'], state['loss_bad_epochs']), (0, 0))

    def test_validation_loss_alone_does_not_indicate_divergence(self):
        state = new_early_stopping()
        for value in range(5):
            self.assertFalse(update_early_stopping(state, value / 10, float(value), float(value)))

    def test_missing_legacy_train_loss_breaks_streak(self):
        state = new_early_stopping()
        update_early_stopping(state, .1, 4., 1.)
        update_early_stopping(state, .2, 3., 2.)
        update_early_stopping(state, .3, None, 3.)
        self.assertFalse(update_early_stopping(state, .4, 2., 4.))
        self.assertEqual(state['loss_bad_epochs'], 0)

    def test_small_gains_accumulate_against_anchor_not_raw_best(self):
        state = new_early_stopping()
        for score in (.6, .60004, .60008):
            self.assertFalse(update_early_stopping(state, score, 1., 1.))
        self.assertEqual((state['ndcg_bad_epochs'], state['ndcg_anchor']), (2, .6))
        self.assertFalse(update_early_stopping(state, .60012, 1., 1.))
        self.assertEqual((state['ndcg_bad_epochs'], state['ndcg_anchor']), (0, .60012))

    def test_exact_delta_and_nonfinite_monitoring_values(self):
        state = new_early_stopping(ndcg_anchor=.6)
        update_early_stopping(state, .6 + state['min_delta'], 1., 1.)
        self.assertEqual(state['ndcg_bad_epochs'], 1)
        before = dict(state)
        with self.assertRaises(FloatingPointError):
            update_early_stopping(state, float('nan'), 1., 1.)
        self.assertEqual(state, before)

try:
    import torch
    import train_trivago_top5 as trainer
except (ImportError, OSError):
    torch = None


@unittest.skipIf(torch is None, 'PyTorch unavailable')
class ValidationMetricTrainingTests(unittest.TestCase):
    def command(self, prepared, run, epochs=2, extra=()):
        return ['trainer', '--prepared-dir', str(prepared), '--output-dir', str(run),
                '--epochs', str(epochs), '--batch-size', '2', '--emb-size', '4', '--num-heads', '2',
                '--sample-order', '2', '--metadata-projection', 'fieldwise', '--action-mode', 'all_event',
                '--checkpoint-every', '1', '--max-hours', '0', *extra]

    def invoke(self, argv):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(sys, 'argv', argv), \
                patch.object(torch.cuda, 'is_available', return_value=False):
            trainer.main()
        return output.getvalue()

    def prepared(self, folder):
        if __package__:
            from .test_trivago_top5_all_events import fixture
        else:
            from test_trivago_top5_all_events import fixture
        return fixture(folder, full_metadata=True)[0]

    @staticmethod
    def legacy_scheduler(payload):
        payload.pop('training_policy', None)
        payload['scheduler'] = dict(step_size=3, gamma=.1, last_epoch=payload['epoch'])
        return payload

    @staticmethod
    def val(ndcg, recall=.5, ce=1.):
        return {'ndcg_at_5': ndcg, 'recall_at_5': recall, 'examples': 2,
                'ce_loss': ce, 'metadata_loss': .2, 'total_loss': ce + .2}

    def assert_loss_reporting(self, output, payload):
        self.assertIn('ce_loss', output)
        self.assertIn('early_stopping', payload)
        self.assertEqual(payload['selection_metric'], 'ndcg_at_5')
        for row in payload['metric_history']:
            if 'train' in row:
                self.assertAlmostEqual(row['train']['total_loss'],
                                       row['train']['ce_loss'] + row['train']['metadata_loss'])

    def test_real_epochs_report_both_val_metrics_losses_and_final_test(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            folder = Path(temporary)
            prepared, run = self.prepared(folder), folder / 'run'
            output = self.invoke(self.command(prepared, run))
            records = [json.loads(line) for line in output.splitlines() if line.startswith('{"epoch"')]
            self.assertEqual([row['epoch'] for row in records], [1, 2])
            self.assertTrue(all(set(row['val']) == {'ndcg_at_5', 'recall_at_5', 'examples',
                                                   'ce_loss', 'metadata_loss', 'total_loss'} for row in records))
            self.assertTrue(all(row['train']['complete'] and row['train']['examples'] == 6 for row in records))
            latest = torch.load(run / 'result_latest.pth', weights_only=False)
            self.assert_loss_reporting(output, latest)
            self.assertTrue(latest['training_finished'])
            self.assertEqual(latest['global_step'], 6)
            self.assertFalse((run / 'loss_history.json').exists())
            self.assertIn('"test"', self.invoke(self.command(prepared, run, extra=['--eval-only', 'test'])))

    def test_ndcg_stops_after_three_bad_epochs_even_without_loss_divergence(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            folder = Path(temporary)
            prepared, run = self.prepared(folder), folder / 'run'
            with patch.object(trainer, 'evaluate', side_effect=[self.val(n) for n in [.3, .9, .2, .2, .1, .1]]):
                output = self.invoke(self.command(prepared, run, epochs=6))
            latest = torch.load(run / 'result_latest.pth', weights_only=False)
            best = torch.load(run / 'result_best.pth', weights_only=False)
            self.assertEqual(latest['epoch'], 5)
            self.assertEqual(latest['global_step'], 15)
            self.assertEqual((best['epoch'], best['best_ndcg_at_5']), (2, .9))
            self.assertEqual(best['scheduler']['last_epoch'], 2)
            self.assert_loss_reporting(output, latest)
            self.assertEqual(latest['early_stopping']['reason'], 'val_ndcg_no_improvement')
            self.assertEqual(latest['early_stopping']['loss_bad_epochs'], 0)
            self.assertEqual(json.loads((run / 'run_status.json').read_text())['status'], 'EARLY_STOPPED')
            digest = hashlib.sha256((run / 'result_latest.pth').read_bytes()).digest()
            self.invoke(self.command(prepared, run, epochs=20, extra=['--resume']))
            self.assertEqual(hashlib.sha256((run / 'result_latest.pth').read_bytes()).digest(), digest)
            self.assertIn('"test"', self.invoke(self.command(prepared, run, extra=['--eval-only', 'test'])))

    def test_partial_epoch_resume_retains_offsets_optimizer_and_metric_history(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            folder = Path(temporary)
            prepared, run = self.prepared(folder), folder / 'run'
            self.invoke(self.command(prepared, run, extra=['--max-batches', '1']))
            partial = torch.load(run / 'result_latest.pth', weights_only=False)
            self.assertEqual((partial['epoch'], partial['next_offset']), (0, 2))
            self.assertEqual(partial['train_epoch_totals']['examples'], 2)
            self.invoke(self.command(prepared, run, extra=['--resume']))
            latest = torch.load(run / 'result_latest.pth', weights_only=False)
            self.assertEqual(latest['global_step'], 6)
            self.assertEqual([row['epoch'] for row in latest['metric_history']], [1, 2])
            self.assertEqual(latest['metric_history'][0]['train']['examples'], 6)
            full_run = folder / 'uninterrupted'
            self.invoke(self.command(prepared, full_run))
            full = torch.load(full_run / 'result_latest.pth', weights_only=False)
            self.assertEqual(full['metric_history'], latest['metric_history'])
            self.assertEqual(full['early_stopping'], latest['early_stopping'])
            for key in full['model']:
                self.assertTrue(torch.equal(full['model'][key], latest['model'][key]), key)

    def test_loss_branch_stops_with_increasing_ndcg_and_saves_current_best(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            prepared, run = self.prepared(folder), folder / 'run'
            calls = []

            def controlled_forward(model, batch):
                targets = batch[3].long()
                scores = torch.zeros((len(targets), 6))
                scores.scatter_(1, targets[:, None], 1. + len(calls) // 3)
                scores.requires_grad_(True)
                calls.append(True)
                return targets, scores, torch.tensor(0.)

            with patch.object(trainer, 'forward', side_effect=controlled_forward), \
                    patch.object(trainer, 'evaluate', side_effect=[self.val(.1 * i, ce=float(i)) for i in range(1, 7)]):
                self.invoke(self.command(prepared, run, epochs=6))
            latest = torch.load(run / 'result_latest.pth', weights_only=False)
            best = torch.load(run / 'result_best.pth', weights_only=False)
            self.assertEqual(latest['epoch'], 4)
            self.assertEqual(latest['early_stopping']['reason'], 'ce_loss_divergence')
            self.assertEqual(best['epoch'], 4)

    def test_ndcg_patience_survives_resume_and_policy_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            prepared, run = self.prepared(folder), folder / 'run'
            with patch.object(trainer, 'evaluate', side_effect=[self.val(.9), self.val(.8)]):
                self.invoke(self.command(prepared, run, epochs=2))
            with self.assertRaisesRegex(ValueError, 'training policy differs'):
                self.invoke(self.command(prepared, run, epochs=6, extra=['--resume', '--patience', '4']))
            with patch.object(trainer, 'evaluate', side_effect=[self.val(.7), self.val(.6)]):
                self.invoke(self.command(prepared, run, epochs=6, extra=['--resume']))
            latest = torch.load(run / 'result_latest.pth', weights_only=False)
            self.assertEqual(latest['epoch'], 4)
            self.assertEqual(latest['early_stopping']['ndcg_bad_epochs'], 3)

    def test_plateau_reduces_before_stop_and_best_keeps_subthreshold_gains(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            prepared, run = self.prepared(folder), folder / 'run'
            with patch.object(trainer, 'evaluate', side_effect=[self.val(n) for n in (.6, .60001, .60002, .60003)]):
                self.invoke(self.command(prepared, run, epochs=10))
            latest = torch.load(run / 'result_latest.pth', weights_only=False)
            best = torch.load(run / 'result_best.pth', weights_only=False)
            rows = latest['metric_history']
            self.assertEqual(latest['epoch'], 4)
            self.assertEqual([row['learning_rate'] for row in rows], [.001, .001, .001, .0005])
            self.assertEqual([row['lr_reduced'] for row in rows], [False, False, True, False])
            self.assertEqual(best['best_ndcg_at_5'], .60003)
            self.assertEqual(best['best_epoch'], 4)
            self.assertEqual(best['early_stopping']['ndcg_anchor'], .6)

    def test_plateau_lr_floor_survives_repeated_bad_epochs(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            prepared, run = self.prepared(folder), folder / 'run'
            with patch.object(trainer, 'evaluate', return_value=self.val(.6)):
                self.invoke(self.command(prepared, run, epochs=7,
                                         extra=['--lr', '0.00002', '--patience', '10']))
            latest = torch.load(run / 'result_latest.pth', weights_only=False)
            rates = [row['next_learning_rate'] for row in latest['metric_history']]
            self.assertEqual(rates, [2e-5, 2e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5])

    def test_resume_restores_plateau_counters_and_accumulated_ndcg_anchor(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            prepared, run = self.prepared(folder), folder / 'resumed'
            metrics = [.6, .60004, .60008, .60012, .60013]
            with patch.object(trainer, 'evaluate', side_effect=[self.val(n) for n in metrics[:2]]):
                self.invoke(self.command(prepared, run, epochs=2))
            with patch.object(trainer, 'evaluate', side_effect=[self.val(n) for n in metrics[2:]]):
                self.invoke(self.command(prepared, run, epochs=5, extra=['--resume']))
            with patch.object(trainer, 'evaluate', side_effect=[self.val(n) for n in metrics]):
                self.invoke(self.command(prepared, folder / 'full', epochs=5))
            resumed = torch.load(run / 'result_latest.pth', weights_only=False)
            full = torch.load(folder / 'full/result_latest.pth', weights_only=False)
            for key in ('scheduler', 'metric_history', 'early_stopping', 'training_policy'):
                self.assertEqual(resumed[key], full[key], key)
            self.assertEqual(resumed['early_stopping']['ndcg_anchor'], .60012)
            self.assertEqual(resumed['scheduler']['best'], .60012)
            with self.assertRaisesRegex(ValueError, 'training policy differs'):
                self.invoke(self.command(prepared, run, epochs=6, extra=['--resume', '--min-delta', '.001']))

    def test_v1_migration_backs_up_checkpoint_keeps_optimizer_and_clamps_lr(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            prepared, run = self.prepared(folder), folder / 'run'
            self.invoke(self.command(prepared, run, extra=['--max-batches', '1']))
            legacy = torch.load(run / 'result_latest.pth', weights_only=False)
            self.legacy_scheduler(legacy)
            legacy['early_stopping'].update(version='val_ndcg_or_ce_divergence_v1',
                                            previous_train_ce=2., previous_val_ce=1., loss_bad_epochs=2)
            legacy['early_stopping'].pop('min_delta')
            legacy['early_stopping'].pop('ndcg_anchor')
            legacy['early_stopping'].pop('loss_min_delta')
            legacy['optimizer']['param_groups'][0]['lr'] = 1e-9
            trainer.atomic_save(legacy, run / 'result_latest.pth')
            digest = hashlib.sha256((run / 'result_latest.pth').read_bytes()).digest()
            # Pause on validation, after the already completed epoch's train batches.
            with patch.object(trainer, 'evaluate', side_effect=trainer.TimeBudgetExceeded):
                output = self.invoke(self.command(prepared, run, extra=['--resume']))
            backups = list((run / 'training_policy_backups').glob('*/result_latest.pth'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(hashlib.sha256(backups[0].read_bytes()).digest(), digest)
            new = torch.load(run / 'result_latest.pth', weights_only=False)
            self.assertEqual(new['optimizer']['param_groups'][0]['lr'], 1e-5)
            self.assertEqual(new['global_step'], 3)
            self.assertEqual(new['train_epoch_totals']['examples'], 6)
            state = new['early_stopping']
            self.assertEqual(state['loss_bad_epochs'], 0)
            self.assertEqual((state['previous_train_ce'], state['previous_val_ce']), (2., 1.))
            self.assertFalse(update_early_stopping(state, .6, 1.8, 1.1))
            self.assertFalse(update_early_stopping(state, .7, 1.6, 1.2))
            self.assertTrue(update_early_stopping(state, .8, 1.4, 1.3))
            self.assertEqual(state['reason'], 'ce_loss_divergence')
            self.assertEqual(new['scheduler']['last_epoch'], 0)
            self.assertNotIn('step_size', new['scheduler'])
            self.assertIn('Migrated StepLR to NDCG plateau', output)
            first_key = next(iter(legacy['optimizer']['state']))
            self.assertGreater(new['optimizer']['state'][first_key]['step'],
                               legacy['optimizer']['state'][first_key]['step'])

    def test_migration_does_not_restart_a_legacy_stopped_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            prepared, run = self.prepared(folder), folder / 'run'
            self.invoke(self.command(prepared, run, epochs=1))
            legacy = torch.load(run / 'result_latest.pth', weights_only=False)
            self.legacy_scheduler(legacy)
            legacy['early_stopping'].update(
                version='val_ndcg_or_ce_divergence_v1', loss_bad_epochs=3,
                stopped=True, reason='ce_loss_divergence', triggered=['ce_loss_divergence'])
            legacy['early_stopping'].pop('loss_min_delta')
            trainer.atomic_save(legacy, run / 'result_latest.pth')
            with patch.object(trainer, 'forward', side_effect=AssertionError('Stopped run trained')):
                self.invoke(self.command(prepared, run, epochs=5, extra=['--resume']))
            migrated = torch.load(run / 'result_latest.pth', weights_only=False)
            self.assertEqual(migrated['global_step'], legacy['global_step'])
            self.assertTrue(migrated['early_stopping']['stopped'])
            self.assertEqual(migrated['early_stopping']['loss_bad_epochs'], 0)
            self.assertEqual(json.loads((run / 'run_status.json').read_text())['status'], 'EARLY_STOPPED')

    def test_legacy_partial_epoch_does_not_invent_missing_train_loss(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            prepared, run = self.prepared(folder), folder / 'run'
            self.invoke(self.command(prepared, run, extra=['--max-batches', '1']))
            partial = torch.load(run / 'result_latest.pth', weights_only=False)
            partial.pop('early_stopping')
            partial.pop('train_epoch_totals')
            self.legacy_scheduler(partial)
            trainer.atomic_save(partial, run / 'result_latest.pth')
            self.invoke(self.command(prepared, run, extra=['--resume']))
            latest = torch.load(run / 'result_latest.pth', weights_only=False)
            self.assertFalse(latest['metric_history'][0]['train']['complete'])
            self.assertEqual(latest['metric_history'][0]['train']['examples'], 4)
            self.assertEqual(latest['early_stopping']['loss_bad_epochs'], 0)

    def test_real_event_graph_training_loss_and_partial_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            if __package__:
                from . import test_trivago_top5_all_events as fixtures
            else:
                import test_trivago_top5_all_events as fixtures
            original_prepare = fixtures.prepare

            def item_event_prepare(args):
                args.action_scope = 'item_event'
                return original_prepare(args)

            with patch.object(fixtures, 'prepare', side_effect=item_event_prepare):
                prepared = self.prepared(folder)
            run = folder / 'graph'
            graph = ['--action-mode', 'event_graph_candidate']
            self.invoke(self.command(prepared, run, extra=graph + ['--max-batches', '1']))
            self.invoke(self.command(prepared, run, extra=graph + ['--resume']))
            latest = torch.load(run / 'result_latest.pth', weights_only=False)
            self.assertEqual((latest['epoch'], latest['global_step']), (2, 6))
            for row in latest['metric_history']:
                self.assertTrue(row['train']['complete'])
                for split in ('train', 'val'):
                    for name in ('ce_loss', 'metadata_loss', 'total_loss'):
                        self.assertTrue(torch.isfinite(torch.tensor(row[split][name])))
            self.assertIn('"test"', self.invoke(self.command(prepared, run, extra=graph + ['--eval-only', 'test'])))

    def test_legacy_completed_epoch_one_can_extend_without_retraining_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            prepared, run = self.prepared(folder), folder / 'run'
            self.invoke(self.command(prepared, run, epochs=1))
            legacy = torch.load(run / 'result_latest.pth', weights_only=False)
            legacy.pop('early_stopping')
            legacy.pop('train_epoch_totals')
            self.legacy_scheduler(legacy)
            for record in legacy['metric_history']:
                record.pop('train')
                record.pop('early_stopping')
                for key in ('ce_loss', 'metadata_loss', 'total_loss'):
                    record['val'].pop(key)
            trainer.atomic_save(legacy, run / 'result_latest.pth')
            output = self.invoke(self.command(prepared, run, epochs=2, extra=['--resume']))
            self.assertIn('new NDCG patience starts at zero', output)
            latest = torch.load(run / 'result_latest.pth', weights_only=False)
            self.assertEqual((latest['epoch'], latest['global_step']), (2, 6))
            self.assertNotIn('train', latest['metric_history'][0])
            self.assertEqual(latest['metric_history'][1]['train']['examples'], 6)

    def test_pause_during_val_does_not_repeat_train_and_test_remains_final_only(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            folder = Path(temporary)
            prepared, run = self.prepared(folder), folder / 'run'
            with patch.object(trainer, 'evaluate', side_effect=trainer.TimeBudgetExceeded):
                self.invoke(self.command(prepared, run))
            with self.assertRaisesRegex(ValueError, 'final-only'):
                self.invoke(self.command(prepared, run, extra=['--eval-only', 'test']))
            self.invoke(self.command(prepared, run, extra=['--resume']))
            latest = torch.load(run / 'result_latest.pth', weights_only=False)
            self.assertEqual(latest['global_step'], 6)
            self.assertEqual(len(latest['metric_history']), 2)

    def test_old_loss_selected_checkpoint_resumes_weights_but_reinitializes_metric_best(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            folder = Path(temporary)
            prepared, run = self.prepared(folder), folder / 'run'
            self.invoke(self.command(prepared, run, epochs=1))
            latest = torch.load(run / 'result_latest.pth', weights_only=False)
            latest.update(selection_metric='val_loss', training_finished=True,
                          early_stopping={'patience': 3, 'bad_epochs': 3}, best_val_loss=.01)
            self.legacy_scheduler(latest)
            for key in ('best_ndcg_at_5', 'metric_history', 'best_epoch'):
                latest.pop(key, None)
            trainer.atomic_save(latest, run / 'result_latest.pth')
            with patch.object(trainer, 'evaluate', side_effect=[self.val(.8), self.val(.5)]) as evaluate:
                output = self.invoke(self.command(prepared, run, epochs=2, extra=['--resume']))
            self.assertEqual(evaluate.call_count, 2)
            new = torch.load(run / 'result_latest.pth', weights_only=False)
            best = torch.load(run / 'result_best.pth', weights_only=False)
            self.assertEqual((new['epoch'], new['global_step']), (2, 6))
            self.assertEqual((best['epoch'], best['best_ndcg_at_5']), (1, .8))
            self.assertEqual(new['metric_history'][0]['phase'], 'resume_baseline')
            self.assert_loss_reporting(output, new)

    def test_eval_only_val_does_not_rewrite_run_status_or_checkpoints(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            folder = Path(temporary)
            prepared, run = self.prepared(folder), folder / 'run'
            self.invoke(self.command(prepared, run, extra=['--max-batches', '1']))
            before = {p.name: hashlib.sha256(p.read_bytes()).digest() for p in run.iterdir() if p.is_file()}
            output = self.invoke(self.command(prepared, run, extra=['--eval-only', 'val']))
            after = {p.name: hashlib.sha256(p.read_bytes()).digest() for p in run.iterdir() if p.is_file()}
            self.assertEqual(before, after)
            self.assertIn('"val"', output)

    def test_completed_resume_does_not_update_model_and_recovers_metric_report(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            folder = Path(temporary)
            prepared, run = self.prepared(folder), folder / 'run'
            self.invoke(self.command(prepared, run))
            digest = hashlib.sha256((run / 'result_latest.pth').read_bytes()).digest()
            (run / 'metric_history.json').unlink()
            with patch.object(trainer, 'evaluate', side_effect=AssertionError('Already complete')):
                self.invoke(self.command(prepared, run, extra=['--resume']))
            self.assertEqual(hashlib.sha256((run / 'result_latest.pth').read_bytes()).digest(), digest)
            self.assertEqual(len(json.loads((run / 'metric_history.json').read_text())), 2)

    def test_eval_loads_on_cpu_releases_payload_and_does_not_create_optimizer(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            folder = Path(temporary)
            prepared, run = self.prepared(folder), folder / 'run'
            self.invoke(self.command(prepared, run, epochs=1))
            original_load, original_evaluate = torch.load, trainer.evaluate
            refs = []

            def tracked_load(*args, **kwargs):
                self.assertEqual(kwargs['map_location'], 'cpu')
                payload = original_load(*args, **kwargs)
                refs.append(weakref.ref(payload['model']['embedding_i.weight']))
                return payload

            def tracked_evaluate(*args, **kwargs):
                self.assertTrue(refs)
                self.assertIsNone(refs[-1](), 'Serialized model tensors must be released before evaluation')
                return original_evaluate(*args, **kwargs)

            with patch.object(torch, 'load', side_effect=tracked_load), \
                    patch.object(trainer, 'evaluate', side_effect=tracked_evaluate), \
                    patch.object(torch.optim, 'Adam', side_effect=AssertionError('Evaluation created optimizer')):
                self.invoke(self.command(prepared, run, epochs=1, extra=['--eval-only', 'val']))

    def test_eval_does_not_require_optimizer_or_scheduler_payload(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            folder = Path(temporary)
            prepared, run = self.prepared(folder), folder / 'run'
            self.invoke(self.command(prepared, run, epochs=1))
            best = torch.load(run / 'result_best.pth', weights_only=False)
            best.pop('optimizer')
            best.pop('scheduler')
            trainer.atomic_save(best, run / 'result_best.pth')
            self.assertIn('"val"', self.invoke(self.command(prepared, run, extra=['--eval-only', 'val'])))

    def test_gradients_are_cleared_before_every_training_forward(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            folder = Path(temporary)
            prepared, run = self.prepared(folder), folder / 'run'
            original_forward = trainer.forward
            seen = []

            def checked_forward(model, data):
                if model.training:
                    self.assertTrue(all(p.grad is None for p in model.parameters()))
                    seen.append(True)
                return original_forward(model, data)

            with patch.object(trainer, 'forward', side_effect=checked_forward):
                self.invoke(self.command(prepared, run, epochs=1))
            self.assertEqual(len(seen), 3)

    def test_nonfinite_objective_never_steps_optimizer_or_saves_bad_best(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            folder = Path(temporary)
            prepared, run = self.prepared(folder), folder / 'run'
            output = (torch.tensor([1, 1]), torch.full((2, 6), float('nan'), requires_grad=True), 0.)
            with patch.object(trainer, 'forward', return_value=output):
                with self.assertRaisesRegex(FloatingPointError, 'Non-finite objective'):
                    self.invoke(self.command(prepared, run, epochs=1))
            self.assertFalse((run / 'result_latest.pth').exists())
            self.assertFalse((run / 'result_best.pth').exists())

    def test_nonfinite_validation_does_not_commit_metrics_or_best(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            folder = Path(temporary)
            prepared, run = self.prepared(folder), folder / 'run'
            original_forward = trainer.forward

            def nan_on_eval(model, data):
                targets, scores, auxiliary = original_forward(model, data)
                if not model.training:
                    scores.fill_(float('nan'))
                return targets, scores, auxiliary

            with patch.object(trainer, 'forward', side_effect=nan_on_eval):
                with self.assertRaisesRegex(FloatingPointError, 'Non-finite impression'):
                    self.invoke(self.command(prepared, run, epochs=1))
            self.assertFalse((run / 'result_best.pth').exists())
            latest = torch.load(run / 'result_latest.pth', weights_only=False)
            self.assertEqual(latest['metric_history'], [])


if __name__ == '__main__':
    unittest.main()
