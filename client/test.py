from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer
from transformers.models.bert.configuration_bert import BertOnnxConfig
from base import InfDataset

MODEL_NAME = "bert-base-uncased"
MODEL_VERSION = "1"
DATASET_NAME = "squad"
COLUMN_NAME1 = "question"
COLUMN_NAME2 = "context"

model_config = AutoConfig.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
max_sequence_length = getattr(
    tokenizer, "model_max_length", None)
onnx_config = BertOnnxConfig(model_config)
input_names = tuple(onnx_config.inputs.keys())
dataset = load_dataset(DATASET_NAME, split="train")


def preprocess_function(dataset):
    return tokenizer(dataset[COLUMN_NAME1], dataset[COLUMN_NAME2], truncation=True, padding='longest', max_length=max_sequence_length, return_tensors="np")


item = tokenizer(dataset[COLUMN_NAME1][0], dataset[COLUMN_NAME2][0], truncation=True,
                 padding='longest', max_length=max_sequence_length, return_tensors="np")
# print(item)

tokenized_dataset = dataset.with_transform(preprocess_function)
inf_dataset = InfDataset(tokenized_dataset)

cnt = 0
for sample in inf_dataset:
    if cnt % 100 == 0:
        print(cnt)
    cnt += 1


# print(tokenized_dataset[0])
