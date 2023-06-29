import unittest
import numpy as np

from plugins.triton.triton_client import TritonClient, INPUT_KEY_DATATYPE
from client.bert import BertGenDataset


class TritonClientTests(unittest.TestCase):
    def test_prepare_input(self):
        client = TritonClient(triton_url=None, model_name=None)
        bertGen = BertGenDataset()
        client.inputs = {mm.name: mm.__dict__ for mm in bertGen.inputs}  # manually set inputs
        for input_name in client.inputs:
            """Set the datatype to dtype for the test"""
            client.inputs[input_name][INPUT_KEY_DATATYPE] = client.inputs[input_name]["dtype"]
        samples = [bertGen.get_dataset()[0]]
        infer_inputs = client._prepare_infer_inputs(samples)

        self.assertEqual(len(infer_inputs), len(bertGen.inputs))
        for i in range(len(infer_inputs)):
            """Test that name and datatype are set correctly"""
            self.assertEqual(infer_inputs[i].name(), bertGen.inputs[i].name)
            self.assertEqual(infer_inputs[i].datatype(), bertGen.inputs[i].dtype)

        for i in range(len(infer_inputs)):
            """Test that raw data is set correctly"""
            self.assertEqual(infer_inputs[i]._raw_data, samples[0][bertGen.inputs[i].name].tobytes())

        """Test that batched data is set correctly"""
        samples = [bertGen.get_dataset()[0], bertGen.get_dataset()[1]]
        infer_inputs = client._prepare_infer_inputs(samples)
        for i in range(len(infer_inputs)):
            self.assertEqual(infer_inputs[i].shape()[0], len(samples))
            batched_data = np.stack([sample[bertGen.inputs[i].name] for sample in samples], 0)
            self.assertEqual(infer_inputs[i]._raw_data, batched_data.tobytes())

    def test_prepare_output(self):
        client = TritonClient(triton_url=None, model_name=None)
        test_outputs = ["output1", "output2"]
        infer_output = client._prepare_infer_outputs(test_outputs)
        for i in range(len(infer_output)):
            self.assertEqual(infer_output[i].name(), test_outputs[i])
