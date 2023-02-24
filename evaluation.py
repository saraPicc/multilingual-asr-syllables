## imports 
import random
import pandas as pd
import re
import torch
import json
from IPython.display import display, HTML
from transformers import Wav2Vec2ForCTC
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2PhonemeCTCTokenizer, AutoTokenizer, AutoModelForCTC, Wav2Vec2Processor
from datasets.utils.version import Version
from datasets import load_dataset, load_metric, Audio
import os
import numpy as np
import sys
import warnings
import argparse
from torch import Tensor
import pickle as pkl
from back2words import *
from random import sample
from CustomML_ITESFRPhoSylCTCTokenizer  import HybridML_ITESFRPhoSylCTCTokenizer  


lang = 'ML_ITESFR'             
train_h = '20h'                
vocab_len = 246                 
check_p = 24350          #  **** ----------------- update with last checkpoint of the model  ----------------- ****
units = 'hybPhoSyl'      
my_path = '/data/disk1/data/spicciau/experiments/' # ---------- update here with your path


model_name = f'wavlm-large-{lang}-{units}-{train_h}_{vocab_len}'
checkpoint = f'checkpoint-{check_p}'
tokenizer_name = f'tokenizerML_ITESFR_{units}{vocab_len}'      

wer_metric = load_metric("wer")
ter_metric = load_metric("cer") # Token Error Rate

"""import the model, processor, tokenizer"""
print(" *--------- loading saved model ---------* ")
saved_model = AutoModelForCTC.from_pretrained(f"{my_path}{train_h}/microsoft/{model_name}/{checkpoint}/", local_files_only = True) 
saved_model.to("cuda")
print(" *--------- loading tokenizer ---------* ")
tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(f"{my_path}{train_h}/{tokenizer_name}/", local_files_only = True)
print(" *--------- loading processor ---------* ")
processor = Wav2Vec2Processor.from_pretrained(f"{my_path}{train_h}/microsoft/{model_name}/{checkpoint}/", local_files_only=True)


print(f" *--------- evaluating {units} test set ---------*")

print(f" *--------- loading {lang} test set from training ---------*")

with open('MLitesfrCVdataset7_transcribed20.pkl', 'rb') as file:     #  **** ----------------- UPDATE HERE ----------------- ****
  common_voice_train, common_voice_test, common_voice_validation = pkl.load(file)

def get_langID(dataset):

    it_indexes = []
    es_indexes = []
    fr_indexes = []

    for i, sample in enumerate(dataset['path']):
        filen = os.path.split(sample)[-1] # getting rid of path
        filen_ok = os.path.splitext(filen)[0] # getting rid of ext
        if '_it_' in filen_ok:
            it_indexes.append(i)
        elif '_es_' in filen_ok:
            es_indexes.append(i)
        elif '_fr_' in filen_ok:
            fr_indexes.append(i)
        else:
            print('ops! file is nor it, es or fr.')
        
    return it_indexes, es_indexes, fr_indexes

it_ids, es_ids, fr_ids = get_langID(common_voice_test) 


"""Prepare Dataset"""

print(" *--------- preparing dataset ---------* ")

def prepare_dataset(batch):
    audio = batch["audio"]    
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch["phonl_tr"]).input_ids
    return batch

common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names , keep_in_memory=True)

print('len test set: ', len(common_voice_test))
## ---------------- EVALUATION

max_input_length_in_sec = 7.0

chars_to_remove_regex = '[\,\#\?\.\!\-\;\:\"\“\%\‘\”\�\°\(\)\–\…\\\[\]\«\»\\\/\^\<\>\~\_\-\¿\¡\—]'

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
    return batch

def replace_hatted_characters(batch):
    batch["sentence"] = re.sub('[’]', "'", batch["sentence"])
    return batch

print(f" *--------- evaluating {units} test set ---------*")

print(f" *--------- loading {lang} original test set ---------*")

with open('MLitesfrCVdataset7_transcribed20.pkl', 'rb') as file:                #  **** ----------------- UPDATE HERE ----------------- ****          
  common_voice_train, common_voice_testEV, common_voice_validation = pkl.load(file)


original_transcriptions = common_voice_testEV['phonl_tr'] # SAMPA transcribed
original_sentence = common_voice_testEV['sentence']       # grapheme transcription
ref2phonseqsList = ref2phonseq(common_voice_testEV['phonl_tr'])
print(ref2phonseqsList[:20])
common_voice_testEV = common_voice_testEV.add_column("pho_seq", ref2phonseqsList)
original_phoseq = common_voice_testEV["pho_seq"]

print('len original test set: ', len(original_transcriptions))

ph_predictions = [ ]
predictions_pho_word_sent = [ ]
ph_seq_predictions = [ ]
gr_predictions = [ ]


for i, el in enumerate(common_voice_test["input_values"]): 
    input_dict = processor(el, return_tensors="pt", padding=True)
    logits= saved_model(input_dict.input_values.to("cuda")).logits 
    pred_ids = torch.argmax(logits, dim=-1)[0]

    predicted_sentences = processor.decode(pred_ids)  # for TER, token con word separator
    ph_predictions.append(predicted_sentences)  # tokenizer tokens - prediction list as it comes out of the model

    # (1) phonemes in words - only for CSV

    predicted_word_sent = back2wordsMLT(predicted_sentences, MLTlex= None, graph = False)  #  prediction pho words + spaces    
    predictions_pho_word_sent.append(predicted_word_sent) # - list of predictions in phoneme words

    # (2) phoneme sequency - per PER

    pred_phoneme_seq_sent = ref2phonseq(predicted_word_sent, dataset = False)  #  prediction pho words + spaces 
    ph_seq_predictions.append(pred_phoneme_seq_sent)       # for PER -  predictions list as phoneme sequence
 
    # (3) graphemes with known sentence language

    if i in it_ids:
        predicted_grapehme = back2wordsIT(predicted_sentences,'ITpho2gr.pkl', graph = True) 
        gr_predictions.append(predicted_grapehme)  # for WER (graphemes)
    elif i in es_ids:
        predicted_grapehme = back2wordsES(predicted_sentences,'ESpho2gr.pkl', graph = True) 
        gr_predictions.append(predicted_grapehme)  # for WER  (graphemes)
    elif i in fr_ids:
        predicted_grapehme = back2wordsFR(predicted_sentences,'FRpho2gr.pkl', graph = True) 
        gr_predictions.append(predicted_grapehme)  # for WER  (graphemes)


tokenized_transcriptions = [] 
for sent in original_transcriptions:
   toke_s =  HybridML_ITESFRPhoSylCTCTokenizer(f'./{tokenizer_name}/vocab.json').wb_format(sent)
   tokenized_transcriptions.append(toke_s)

decoded_toke = tokenized_transcriptions


list_sentTER=[]
list_refTER=[]
list_sentPER=[]
list_refPER=[]
list_sentWER=[]
list_refWER=[]

for i, sentence_ in enumerate(ph_predictions):
    print(i, "Sentence: ",  sentence_)
    print(i, "Reference: ",  decoded_toke[i])
    list_sentTER.append(sentence_)
    list_refTER.append(decoded_toke[i])

result_ter= ter_metric.compute(predictions=[" ".join(list_sentTER)], references=[" ".join(list_refTER)] )

# let's get the samples for the csv

sent_random_sample = sample(range(len(list_sentTER)), 300)
pred_samplePHO = [list_sentTER[i] for i in sent_random_sample]
ref_samplePHO = [list_refTER[i] for i in sent_random_sample]
pred_word_samplePHO = [predictions_pho_word_sent[i] for i in sent_random_sample] 


d={ "TER_predictions":pred_samplePHO, "TER_reference":ref_samplePHO, 'PHO_word_predictions': pred_word_samplePHO } # storing 300 samples of predictions in a CSV
df = pd.DataFrame(d)


print(f" *--------- TER evaluation done ---------*")

for i, sentence_ in enumerate(ph_seq_predictions):
    print(i, "Sentence: ",  sentence_)
    print(i, "Reference: ",  original_phoseq[i])
    list_sentPER.append(sentence_)
    list_refPER.append(original_phoseq[i])

result_per= ter_metric.compute(predictions=[" ".join(list_sentPER)], references=[" ".join(list_refPER)] )

pred_samplePHOSEQ = [list_sentPER[i] for i in sent_random_sample]
ref_samplePHOSEQ = [list_refPER[i] for i in sent_random_sample]

df["PER_predictions"] = pred_samplePHOSEQ
df["PER_reference"] =  ref_samplePHOSEQ

print(f" *--------- PER evaluation done ---------*")

### GRAPHEME PREDICTIONS SAMPLE

print(f" *--------- evaluating GRAPHEMES sample test set ---------*")

for i, sentence_ in enumerate(gr_predictions):
    print(i, "Sentence: ",  sentence_)
    print(i,"Reference: ",  original_sentence[i])
    list_sentWER.append(sentence_)
    list_refWER.append(original_sentence[i])

result_wer= wer_metric.compute(predictions=[list_sentWER], references=[list_refWER])


pred_sampleGRAPH = [list_sentWER[i] for i in sent_random_sample]
ref_sampleGRAPH = [list_refWER[i] for i in sent_random_sample]

df["WER_predictions"] = pred_sampleGRAPH
df["WER_reference"] =  ref_sampleGRAPH


df.to_csv(f"{my_path}{train_h}/CSV_{lang}-TERPERWER-{train_h}_{vocab_len}.csv")

print(f" *--------- grapheme evaluation done ---------*")

print("phoneme_TER", result_ter)
print("phoneme_PER", result_per)
print("grapheme_WER: ", result_wer)
