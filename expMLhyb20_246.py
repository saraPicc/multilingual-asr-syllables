import pandas as pd
from datasets import ClassLabel
from datasets import Dataset
import random
import os
import re
import torch
import json
from IPython.display import display, HTML
from transformers import Wav2Vec2ForCTC 
import pickle as pkl
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import AutoModelForCTC, Wav2Vec2Processor
from datasets.utils.version import Version
from datasets import load_dataset, load_metric, Audio
import numpy as np
import argparse
from torch import Tensor
from segments import Profile, Tokenizer
from HybridML_ITESFRPhoSylCTCTokenizer import HybridML_ITESFRPhoSylCTCTokenizer
import pickle as pkl


lang = 'ML_ITESFR'    # storing the arg setting in variables
train_h = '20h' 
vocab_len = 246
units = 'hybPhoSyl'

## --------------- LOAD DATASET 


print('*-------- loading clean, filtered (7s), transcribed, downsampled dataset  --------*')

with open('MLitesfrCVdataset7_transcribed20.pkl', 'rb') as file:
  common_voice_train, common_voice_test, common_voice_validation = pkl.load(file)

len_train = len(common_voice_train)
len_test = len(common_voice_test)
len_validation=len(common_voice_validation)

print(f" FILE AUDIO PER SET   train: {len_train},     test: {len_test},     validation: {len_validation}")



## ---------------- TOKENIZER
print('*------- loading custom tokenizer -------*')

tokenizer = HybridML_ITESFRPhoSylCTCTokenizer.from_pretrained(f"./tokenizerML_ITESFR_{units}{vocab_len}", do_phonemize = False, local_files_only=True)


## ---------------- FEATURE EXTRACTOR + PROCESSOR 

print('*------- setting feature extractor and processor -------*')

from transformers import Wav2Vec2FeatureExtractor
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

from transformers import Wav2Vec2Processor
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)



## ---------------- DATASET PREPARATION 

print('*------- prepare dataset -------*')

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch["phonl_tr"]).input_ids
    return batch

common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names)
common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names)
common_voice_validation = common_voice_validation.map(prepare_dataset, remove_columns=common_voice_validation.column_names)


## ---------------- MODEL PREPARATION

print('*------- preparing the model -------*')

import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

"""Data Collator"""
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

"""Metrics WER - PER""" 
wer_metric = load_metric("wer")
ter_metric = load_metric("cer") # Token Error Rate

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id 

    pred_str = processor.batch_decode(pred_ids)       # decoding > labels to string
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    i=random.randrange(0,len(pred_str)-1)
    print(f"\n\nH: {pred_str[i]}, R: {label_str[i]}\n\n") # print prediction to analyze learning trend

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    ter = ter_metric.compute(predictions=pred_str, references=label_str)
    print("wer:", wer )
    print("ter:", ter) 

    return {"wer": wer,
            "ter": ter,}


# -------------------------------------------------------------------------------------------------------------------------------------------

## ---------------- PRETRAINED MODEL LOADING

from transformers import WavLMForCTC
import torch

print('*------- loading pretrained model -------*')

model = WavLMForCTC.from_pretrained(
    "microsoft/wavlm-large", 
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.0,
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=vocab_len,
) 

print('*------- model loaded -------*')



## ---------------- TRAINING PARAMETERS DEFINITION

from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir= f"microsoft/wavlm-large-{lang}-{units}-{train_h}_{vocab_len}",
 
  group_by_length=True, # helps speed up the training by grouping together similar batches and aid the padding
  per_device_train_batch_size=4,  ##16 #8
  per_device_eval_batch_size=1,   ##
  gradient_accumulation_steps=2,
  evaluation_strategy="epoch",   
  num_train_epochs=30,
  gradient_checkpointing=True,
  fp16=True,  # comment when GPU is not avilable --------------------------------------------
 # optim="adamw_torch",  # inserted this to avoid future warning

  #save_steps=400,
  #eval_steps=400,
  #logging_steps=400,
  learning_rate=3e-4,
  warmup_steps=500,
  save_total_limit=2,
  save_strategy= "epoch",             ##
  metric_for_best_model="eval_loss",  ##
  load_best_model_at_end = True,      ##
)

print('*------- parameters set up -------*')



## ---------------- BUILDING THE TRAINER

from transformers import Trainer, EarlyStoppingCallback 

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=common_voice_train,
    eval_dataset=common_voice_validation, ##
    tokenizer=processor.feature_extractor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)



## ---------------- TRAINING

"""if "out-of-memory" error: reduce per_device_train_batch_size to 8 or even less and increase gradient_accumulation."""

print("*------- TRAINING STARTED -------*")
trainer.train()
#trainer.train(resume_from_checkpoint = True)

print("*------- TRAINING DONE -------*")

print("model and tokenizer have been saved in the output_dir directory")



