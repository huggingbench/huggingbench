import unittest
from unittest.mock import MagicMock

from client.runner import Runner, RunnerConfig


class RunnerTestCase(unittest.TestCase):
    def test_run(self):
        # Mock dependencies
        mock_client = MagicMock()
        mock_dataset = MagicMock()

        # Create a sample dataset
        sample_dataset = [1, 2, 3, 4, 5]

        # Configure the mock objects
        mock_client.infer_batch_async.return_value = MagicMock()
        mock_client.infer_batch.return_value = MagicMock()
        mock_dataset.__iter__.return_value = iter(sample_dataset)
        mock_dataset.__len__.return_value = len(sample_dataset)

        # Testing no batching
        config = RunnerConfig(batch_size=1, async_req=False, workers=1)
        runner = Runner(config, mock_client, mock_dataset)
        execution_times, _, failure_rate, total, success_count = runner.run()
        self.assertEqual(len(execution_times), len(sample_dataset))
        self.assertEqual(failure_rate, 0.0)
        self.assertEqual(total, len(sample_dataset))
        self.assertEqual(success_count, len(sample_dataset))

        # Testing with batching
        config = RunnerConfig(batch_size=2, async_req=False, workers=1)
        runner = Runner(config, mock_client, mock_dataset)
        execution_times, _, failure_rate, total, success_count = runner.run()
        total_batches = len(sample_dataset) // 2 if len(sample_dataset) % 2 == 0 else len(sample_dataset) // 2 + 1
        self.assertEqual(len(execution_times), total_batches)
        self.assertEqual(failure_rate, 0.0)
        self.assertEqual(total, len(sample_dataset))
        self.assertEqual(success_count, total_batches)

        # Testing with batching and multiple workers
        config = RunnerConfig(batch_size=2, async_req=False, workers=2)
        runner = Runner(config, mock_client, mock_dataset)
        execution_times, _, failure_rate, total, success_count = runner.run()
        total_batches = len(sample_dataset) // 2 if len(sample_dataset) % 2 == 0 else len(sample_dataset) // 2 + 1
        self.assertEqual(len(execution_times), total_batches)
        self.assertEqual(failure_rate, 0.0)
        self.assertEqual(total, len(sample_dataset))
        self.assertEqual(success_count, total_batches)

        # Testing async requests
        config = RunnerConfig(batch_size=1, async_req=True, workers=1)
        runner = Runner(config, mock_client, mock_dataset)
        execution_times, _, failure_rate, total, success_count = runner.run()
        self.assertEqual(len(execution_times), len(sample_dataset))
        self.assertEqual(failure_rate, 0.0)
        self.assertEqual(total, len(sample_dataset))
        self.assertEqual(success_count, len(sample_dataset))
