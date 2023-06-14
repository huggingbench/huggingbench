import logging
from datasets import load_dataset
from transformers import AutoConfig
from transformers.models.resnet.configuration_resnet import ResNetOnnxConfig
from torchvision import transforms
from client.base import DatasetProvider, DatasetGen
from hugging_bench.hugging_bench_config import Input


MODEL_NAME = "microsoft/resnet-50"
MODEL_VERSION = "1"
DATASET_NAME = "huggingface/cats-image"
DATASET_COLUMN_NAME = "image"

log = logging.getLogger(__name__)


class ResnetDataset(DatasetProvider):
    def __init__(self, dataset_name: str = DATASET_NAME) -> None:
        self.dataset_name = dataset_name
        dataset = load_dataset(self.dataset_name, split="test")
        model_config = AutoConfig.from_pretrained(MODEL_NAME)
        onnx_config = ResNetOnnxConfig(model_config)
        input_names = tuple(onnx_config.inputs.keys())  # pixel_values
        self.input_names = input_names
        self.dataset = dataset.with_transform(self.transform)

    def transform(self, dataset):
        img_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

        # We need to convert the image to tensor
        # and then tensor to numpy (triton user operates on numpy arrays)
        dataset[self.input_names[0]] = [img_transform(image).numpy() for image in dataset[DATASET_COLUMN_NAME]]
        return dataset


class ResnetGenDataset(DatasetGen):
    """Dataset with random tensors"""

    inputs = [Input(name="pixel_values", dtype="FP32", dims=[3, 224, 224])]

    def __init__(self):
        super().__init__(ResnetGenDataset.inputs)
