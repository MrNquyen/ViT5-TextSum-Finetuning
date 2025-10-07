import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments
from torch.nn import functional as F

from utils.registry import registry


class Trainer:
    def __init__(self):
        self.model_config = registry.get_config("model_attributes")
        self.device = registry.get_args("device")
        self.writer = registry.get_writer("common")
        self.load_model()


    def load_model(self):
        self.model_name = self.model_config["pretrained"]
        config = AutoConfig(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)  
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            config=config
        ).to(self.device)

    
    def trainer_setup(self):
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, 
            model=self.model, 
            return_tensors="pt"
        )

