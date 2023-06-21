import gevent.monkey

gevent.monkey.patch_all()
import unittest
from client.bert import BertDataset, BertGenDataset


class TestBert(unittest.TestCase):
    """Test if we can load Bert Squad dataset"""

    def test_load_and_tokenize_dataset(self):
        """Test loading and tokenization of Bert Squad dataset"""
        bert = BertDataset()

        self.assertTrue(len(bert.get_dataset()) > 1)
        sample = bert.get_dataset()[0]
        for input in BertGenDataset.inputs:
            self.assertIsNotNone(sample[input.name])
