from datasets import load_dataset
from transformers import AutoConfig
from transformers.models.resnet.configuration_resnet import ResNetOnnxConfig
from torchvision import transforms
from base import UserContext, InfDataset
from client.triton_user import TritonUser

MODEL_NAME = "microsoft/resnet-50"
MODEL_VERSION = "1"

dataset = load_dataset("huggingface/cats-image", split="test")

model_config = AutoConfig.from_pretrained(MODEL_NAME)
onnx_config = ResNetOnnxConfig(model_config)
input_names = tuple(onnx_config.inputs.keys())  # pixel_values


def transform(dataset):
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # We need to convert the image to tensor
    # and then tensor to numpy (triton user operates on numpy arrays)
    dataset[input_names[0]] = [img_transform(
        image).numpy() for image in dataset["image"]]
    return dataset


# we run map to process the whole dataset instead of lazy processing 'with_transform' call
DATASET = InfDataset(dataset.with_transform(transform))


class ResnetUser(TritonUser):

    def __init__(self, environment):
        super().__init__(environment, UserContext(DATASET, MODEL_NAME, MODEL_VERSION))
