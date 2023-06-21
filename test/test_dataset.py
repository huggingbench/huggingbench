import unittest

from client.base import DatasetGen, DatasetIterator
from bench.config import Input


class TestBase(unittest.TestCase):
    """Test dataset iterator"""

    def test_dataset_iterator(self):
        datasetGen = DatasetGen([Input(name="test", dtype="FP32", dims=[1, 1, 1])], 200)
        self.assertEqual(len(datasetGen.get_dataset()), 200)
        it = DatasetIterator(datasetGen.get_dataset(), False)
        counter = 0
        for sample in it:
            self.assertIsNotNone(sample["test"])
            counter += 1
        self.assertEqual(counter, 200)
