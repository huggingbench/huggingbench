import unittest

from client.resnet import ResnetDataset, ResnetGenDataset


class TestResnet(unittest.TestCase):
    """Test if we can load Resnet dataset"""

    def test_resnet_load(self):
        resnet = ResnetDataset()
        self.assertTrue(len(resnet.get_dataset()) == 1)  # only one cat image :)
        sample = resnet.get_dataset()[0]
        for input in ResnetGenDataset.inputs:
            self.assertIsNotNone(sample[input.name])
            self.assertTrue(len(sample[input.name]) > 0)
