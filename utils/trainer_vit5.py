import torch
import warnings
import os
import numpy as np
import torch.nn.functional as F

from transformers.optimization import Adafactor, AdafactorSchedule
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments

from icecream import ic
from torch import nn
from tqdm import tqdm

from projects.datasets.dataset import ViInforgraphicSummarizeDataset, get_test_loader
from utils.configs import Config
from utils.logger import Logger
from utils.metrics import metric_calculate
from utils.utils import save_json, count_nan
from utils.registry import registry

# ~Trainer~
class Trainer():
    def __init__(self, config, args):
        self.args = args
        self.device = args.device
        self.config = Config(config)

        print("Build Logger")
        self.writer = Logger(name="all")
        self.writer_evaluation = Logger(name="evaluation")
        self.writer_inference = Logger(name="inference")
        
        self.build()


    #---- BUILD
    def build(self):
        self.writer.LOG_INFO("")
        self.build_registry()
        self.load_model()
        self.load_task()
        self.build_optimizer()
        self.trainer_setup()


    #---- LOAD TASK
    def load_task(self):
        #~ Load config
        self.config_training = self.config.config_training
        self.config_model = self.config.config_model
        self.config_optimizer = self.config.config_optimizer
        self.config_dataset = self.config.config_dataset

        #~ Load Dataset
        train_dataset_object = ViInforgraphicSummarizeDataset(self.dataset_config, self.tokenizer, "train")
        val_dataset_object = ViInforgraphicSummarizeDataset(self.dataset_config, self.tokenizer, "val")
        test_dataset_object = ViInforgraphicSummarizeDataset(self.dataset_config, self.tokenizer, "test")

        self.tokenized_train_dataset = train_dataset_object.get_tokenized_dataset
        self.tokenized_val_dataset = val_dataset_object.get_tokenized_dataset
        self.tokenized_test_dataset = test_dataset_object.get_tokenized_dataset

        test_dataloader = get_test_loader(
            self.tokenized_test_dataset
            data_collator=self.data_collator
        )

    #---- REGISTER
    def build_registry(self):
        # Build writer
        registry.set_module("writer", name="common", instance=self.writer)
        registry.set_module("writer", name="evaluation", instance=self.writer_evaluation)
        registry.set_module("writer", name="inference", instance=self.writer_inference)
        # Build args
        registry.set_module("args", name=None, instance=self.args)
        # Build config
        self.config.build_registry()


    #--- lOAD MODEL
    def load_model():
        #~ Load model
        self.model_name = self.config_model["pretrained"]
        config = AutoConfig(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)  
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            config=config
        ).to(self.device)

        #~ Load collator
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, 
            model=self.model, 
            return_tensors="pt"
        )


    #---- Load Training Modules
    def build_optimizer(self):
        self.optimizer = Adafactor(
            self.model.parameters(),
            **self.config_optimizer
            
        )
        self.lr_scheduler = AdafactorSchedule(optimizer)


    def trainer_setup(self):
        self.training_args = Seq2SeqTrainingArguments(
            "output/",
            **self.config_training
        )

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_train_dataset,
            eval_dataset=self.tokenized_val_dataset,
            data_collator=self.data_collator,
        )


    #---- TRAINING
    def train(self):
        self.trainer.train()

    
    #---- INFERENCE
    def inference(self):
        metrics = load_metric('rouge')
        max_input_length = self.config_model["max_input_length"]
        max_target_length = self.config_model["max_target_length"]

        predictions = []
        references = []

        for i, batch in tqdm(enumerate(tqdm(dataloader)), desc="Evaluate Test Set"):
            outputs = self.model.generate(
                input_ids=batch['input_ids'].to(self.device),
                max_length=max_target_length,
                attention_mask=batch['attention_mask'].to(self.device),
            )
            with tokenizer.as_target_tokenizer():
                outputs = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in outputs]

                labels = np.where(batch['labels'] != -100,  batch['labels'], tokenizer.pad_token_id)
                actuals = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in labels]
            predictions.extend(outputs)
            references.extend(actuals)
            metrics.add_batch(predictions=outputs, references=actuals)
    
    metric_score = metrics.compute()
    self.writer_evaluation.LOG_INFO(f"Test Evaluation Score is: {metric_score}")