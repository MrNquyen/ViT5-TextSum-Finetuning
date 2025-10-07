import torch
import json
import os

from torch.utils.data import Dataset, DataLoader

from utils.utils import load_json, load_npy
from utils.registry import registry
from icecream import ic
from tqdm import tqdm

#----------DATASET----------
class ViInforgraphicSummarizeDataset:
    def __init__(self, tokenizer, split):
        super().__init__()
        self.config_dataset = registry.get_config("dataset_attributes")
        self.config_model = registry.get_config("model_attributes")
        annotation_path = self.config_dataset["annotations"][split]
        annotation_dict = load_json(annotation_path)
        
        self.tokenizer = tokenizer
        self.data = {
            "ids": [],
            "inputs": [],
            "labels": [],
        }

        for im_id, item in tqdm(list(annotation_dict.items())):
            self.data["ids"].append(im_id)
            self.data["inputs"].append(item["ocr_description"])
            self.data["labels"].append(item["caption"])


    def preprocess_function(self):
        max_input_length = self.config_model["max_input_length"]
        max_target_length = self.config_model["max_target_length"]
        model_inputs = self.tokenizer(
            self.data["inputs"], max_length=max_input_length, truncation=True, padding=True
        )
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                self.data["labels"], max_length=max_target_length, truncation=True, padding=True
            )
        model_inputs['labels'] = labels['input_ids']
        model_inputs['input_ids'] = model_inputs['input_ids']
        return model_inputs


    def get_tokenized_dataset(self):
        tokenized_dataset = Dataset.from_dict(self.data)
        tokenized_dataset = dataset.map(
            self.preprocess_function, 
            batched=True, 
            remove_columns=['inputs'], 
            num_proc=8
        )
        return tokenized_dataset


def get_test_loader(self, tokenized_dataset, data_collator):
    dataloader = DataLoader(
        tokenized_dataset, 
        collate_fn=data_collator, 
        batch_size=32,
        shuffle=False
    )
    return dataloader
