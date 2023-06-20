import logging

from datasets import load_dataset

from transformers import AutoConfig, AutoImageProcessor
from transformers.models.resnet.configuration_resnet import ResNetOnnxConfig

from bench.config import Input
from client.base import BaseDataset, DatasetGen

MODEL_NAME = "microsoft/resnet-50"
MODEL_VERSION = "1"
DATASET_NAME = "huggingface/cats-image"
DATASET_COLUMN_NAME = "image"

log = logging.getLogger(__name__)


class ResnetDataset(BaseDataset):
    def __init__(self, dataset_name: str = DATASET_NAME) -> None:
        self.dataset_name = dataset_name
        dataset = load_dataset(self.dataset_name, split="test")
        model_config = AutoConfig.from_pretrained(MODEL_NAME)
        self.image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        onnx_config = ResNetOnnxConfig(model_config)
        input_names = tuple(onnx_config.inputs.keys())  # pixel_values
        self.input_names = input_names
        self.dataset = dataset.with_transform(self.transform)

    def transform(self, dataset):
        # We need to convert the image to numpy arrays
        dataset[self.input_names[0]] = [
            self.image_processor(image, return_tensors="np", do_resize=True) for image in dataset[DATASET_COLUMN_NAME]
        ]
        return dataset


class ResnetGenDataset(DatasetGen):
    """Dataset with random tensors"""

    inputs = [Input(name="pixel_values", dtype="FP32", dims=[3, 224, 224])]

    def __init__(self):
        super().__init__(ResnetGenDataset.inputs)
