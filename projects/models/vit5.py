import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments
from torch.nn import functional as F

from utils.registry import registry


class ViT5:
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
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, 
            model=self.model, 
            return_tensors="pt"
        )

        self.training_args = Seq2SeqTrainingArguments(
            "tmp/",
            do_train=True,
            do_eval=False,
            num_train_epochs=30,
            learning_rate=1e-5,
            warmup_ratio=0.05,
            weight_decay=0.01,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            logging_dir='./log',
            group_by_length=True,
            save_strategy="epoch",
            save_total_limit=3,
            fp16=True,
        )
