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
import evaluate

from projects.datasets.dataset import ViInforgraphicSummarizeDataset, get_test_loader
from projects.models.vit5 import ViT5
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

        # os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device.index)  # chỉ GPU này
        # os.environ["NCCL_DEBUG"] = "INFO"       # debug NCCL
        # os.environ["NCCL_BLOCKING_WAIT"] = "1"  # tránh deadlock
        # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        self.writer = Logger(name="all")
        self.writer_evaluation = Logger(name="evaluation")
        self.writer_inference = Logger(name="inference")
        
        self.build()


    #---- BUILD
    def build(self):
        self.writer.LOG_INFO("=== BUILD REGISTRY ===")
        self.build_registry()

        self.writer.LOG_INFO("=== BUILD CONFIG ===")
        self.load_config()

        self.writer.LOG_INFO("=== BUILD MODEL ===")
        self.load_model()

        self.writer.LOG_INFO("=== BUILD TASK ===")
        self.load_task()

        self.writer.LOG_INFO("=== BUILD OPTIMIZER ===")
        self.build_optimizer()

        self.writer.LOG_INFO("=== BUILD TRAIN ===")
        self.trainer_setup()


    #---- LOAD TASK
    def load_config(self):
        #~ Load config
        self.config_training = self.config.config_training
        self.config_model = self.config.config_model
        self.config_optimizer = self.config.config_optimizer
        self.config_dataset = self.config.config_dataset


    def load_task(self):
        #~ Load Dataset
        train_dataset_object = ViInforgraphicSummarizeDataset(self.tokenizer, "train")
        val_dataset_object = ViInforgraphicSummarizeDataset(self.tokenizer, "val")
        test_dataset_object = ViInforgraphicSummarizeDataset(self.tokenizer, "test")

        self.tokenized_train_dataset = train_dataset_object.get_tokenized_dataset()
        self.tokenized_val_dataset = val_dataset_object.get_tokenized_dataset()
        self.tokenized_test_dataset = test_dataset_object.get_tokenized_dataset()

        self.test_dataloader = get_test_loader(
            self.tokenized_test_dataset,
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
    def load_model(self):
        #~ Load model
        self.model_name = self.config_model["pretrained"]
        config = AutoConfig.from_pretrained(self.model_name)
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
        self.lr_scheduler = AdafactorSchedule(self.optimizer)


    def trainer_setup(self):
        self.training_args = Seq2SeqTrainingArguments(
            "output/",
            **self.config_training,
            local_rank=-1,              # vô hiệu hóa DDP
            ddp_find_unused_parameters=False,
        )
        self.training_args._n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        ic(self.model.device)
        
        # Đảm bảo Trainer chỉ thấy GPU đúng
        # if self.device.type == "cuda":
        
        #     self.writer.LOG_INFO(f"Using GPU device: cuda:{self.device.index}")
        # else:
        #     self.writer.LOG_INFO("Using CPU")

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_train_dataset,
            eval_dataset=self.tokenized_val_dataset,
            data_collator=self.data_collator,
            optimizers=(self.optimizer, self.lr_scheduler)  # << thêm dòng này
        )


    #---- TRAINING
    def train(self):
        self.writer.LOG_INFO("=== Start Training ===")
        self.trainer.train()

    
    #---- INFERENCE
    def inference(self):
        metrics = evaluate('rouge')
        max_input_length = self.config_model["max_input_length"]
        max_target_length = self.config_model["max_target_length"]

        predictions = []
        references = []

        for i, batch in tqdm(enumerate(tqdm(self.test_dataloader)), desc="Evaluate Test Set"):
            outputs = self.model.generate(
                input_ids=batch['input_ids'].to(self.device),
                max_length=max_target_length,
                attention_mask=batch['attention_mask'].to(self.device),
            )
            with self.tokenizer.as_target_tokenizer():
                outputs = [self.tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in outputs]

                labels = np.where(batch['labels'] != -100,  batch['labels'], self.tokenizer.pad_token_id)
                actuals = [self.tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in labels]
            predictions.extend(outputs)
            references.extend(actuals)
            metrics.add_batch(predictions=outputs, references=actuals)
            metrics.compute()
        metric_scores = [{k: v.mid.fmeasure} for k,v in metrics.compute(predictions=predictions, references=references).items()]
        self.writer_evaluation.LOG_INFO(f"Test Evaluation Score is: \n{metric_scores}")