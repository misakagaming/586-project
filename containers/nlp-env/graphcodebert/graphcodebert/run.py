# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import os
import sys
import pickle
import torch
import json
import random
import logging
import argparse
import csv
import numpy as np
from io import open
from pathlib import Path
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq
from tqdm import tqdm, trange
from bleu import _bleu
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

## (TA) [MOD] Modified to include CoolGen DFG function

from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript,DFG_csharp,\
    DFG_coolgen
## [TA] [MOD] END
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser
## (TA) [MOD] Modified to include CoolGen DFG function
dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'javascript':DFG_javascript,
    'c_sharp':DFG_csharp,
    'coolgen':DFG_coolgen,
}
## [TA] [MOD] END

## (TA) [ADDED] Added to block of constants to bypass command line arguments, added new ones

FLAG_MODE = 1  ## 1 i for training 2 for test

params_training={
    '--do_train':True,
    '--do_eval':True,
    '--do_test':None,
    '--model_type':'roberta',
    '--source_lang':'coolgen',
    '--target_lang':'java',
    '--source_ext':'gensrc',
    '--target_ext':'java',
    ## '--source_lang':'java',
    ## '--target_lang':'c_sharp',
    ## '--source_ext':'java',
    ## '--target_ext':'cs',
    '--model_name_or_path':'microsoft/graphcodebert-base',
    '--tokenizer_name':'tokenizer_coolgen',
    '--config_name':'microsoft/graphcodebert-base',
    '--load_model_path':None,
    '--train_filename':'C:/Work/GraphCodeBert/translation/data/javacs/train.java-cs.txt.java,C:/Work/GraphCodeBert/translation/data/javacs/train.java-cs.txt.cs',
    '--train_directories':None,
    '--dev_filename':'C:/Work/GraphCodeBert/translation/data/javacs/valid.java-cs.txt.java,C:/Work/GraphCodeBert/translation/data/javacs/valid.java-cs.txt.cs',
    '--dev_directories':None,
    '--test_filename':'C:/Work/GraphCodeBert/translation/data/javacs/test.java-cs.txt.java,C:/Work/GraphCodeBert/translation/data/javacs/test.java-cs.txt.cs',
    '--test_directories':None,
    ## '--output_dir':'C:/Work/GraphCodeBert/translation/saved_models/java-cs/',
    '--output_dir':'saved_models/coolgen-java/',
    '--max_source_length':514,
    '--max_target_length':514,
    '--beam_size':10,
    '--train_batch_size':32,
    '--eval_batch_size':32,
    '--learning_rate':1e-4,
    '--num_train_epochs':100, ##  2>&1| tee saved_models/java-cs/train.log',
    '--no_cuda':False,
}

params_test={
    '--do_train':None,
    '--do_eval':None,
    '--do_test':True,
    '--model_type':'roberta',
    '--source_lang':'coolgen',
    '--target_lang':'java',
    '--source_ext':'gensrc',
    '--target_ext':'java',
##    '--source_lang':'java',
##    '--target_lang':'c_sharp',
    '--model_name_or_path':'microsoft/graphcodebert-base',
    '--tokenizer_name':'tokenizer_coolgen',
    '--config_name':'microsoft/graphcodebert-base',
    '--load_model_path':'saved_models/coolgen-java/checkpoint-best-bleu/pytorch_model.bin',
    '--train_filename':'C:/Work/GraphCodeBert/translation/data/javacs/train.java-cs.txt.java,C:/Work/GraphCodeBert/translation/data/javacs/train.java-cs.txt.cs',
    '--train_directories':'C:/Users/eraye/Desktop/graphcodebert-env/containers/graphcodebert-env/gen2java/Training/train.coolgen,C:/Users/eraye/Desktop/graphcodebert-env/containers/graphcodebert-env/gen2java/Training/train.java',
    '--dev_filename':'C:/Work/GraphCodeBert/translation/data/javacs/valid.java-cs.txt.java,C:/Work/GraphCodeBert/translation/data/javacs/valid.java-cs.txt.cs',
    '--dev_directories':'C:/Users/eraye/Desktop/graphcodebert-env/containers/graphcodebert-env/gen2java/Training/valid.coolgen,C:/Users/eraye/Desktop/graphcodebert-env/containers/graphcodebert-env/gen2java/Training/valid.java',
    '--test_filename':'C:/Work/GraphCodeBert/translation/data/javacs/test.java-cs.txt.java,C:/Work/GraphCodeBert/translation/data/javacs/test.java-cs.txt.cs',
    '--test_directories':'C:/Users/eraye/Desktop/graphcodebert-env/containers/graphcodebert-env/gen2java/Training/test.coolgen,C:/Users/eraye/Desktop/graphcodebert-env/containers/graphcodebert-env/gen2java/Training/test.java',
    ## '--output_dir':'C:/Work/GraphCodeBert/translation/saved_models/java-cs/',
    '--output_dir':'C:/Users/eraye/Desktop/graphcodebert-env/containers/graphcodebert-env/translation/saved_models/coolgen-java/',
    '--max_source_length':514,
    '--max_target_length':514,
    '--beam_size':10,
    '--train_batch_size':54,
    '--eval_batch_size':64,     ## 2>&1| tee saved_models/java-cs/test.log',
    '--learning_rate':1e-4,
    '--num_train_epochs':100,  ## 2>&1| tee saved_models/java-cs/train.log',
    '--no_cuda':False,
}

## [TA] [ADD] END


logger = logging.getLogger(__name__)
#load parsers
parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser
    
#remove comments, tokenize code and extract dataflow     
def extract_dataflow(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    return code_tokens,dfg


class Example(object):
    """A single training/test example."""
    def __init__(self,
                 source,
                 target,
                 lang
                 ):
        self.source = source
        self.target = target
        self.lang=lang

## (TA) [MOD] Modified to detect language, file via arguments 

def read_examples(srclang, filename):
    """Read examples from filename."""
    is_csv = False
    print(filename)
    if ".csv" in filename:
        is_csv = True
    examples=[]
    if is_csv:
        with open(filename,encoding="utf-8") as f:
            heading = next(f)
            reader_obj = csv.reader(f)
            for idx, row in enumerate(reader_obj):
                src=row[2]
                trg=row[3]           
                examples.append(
                    Example(
                            source=src,
                            target=trg,
                            lang=srclang
                            ) 
                )
        return examples
            
    source,target=filename.split(',')
        
    with open(source,encoding="utf-8") as f1,open(target,encoding="utf-8") as f2:
        for line1,line2 in zip(f1,f2):
            line1=line1.strip()
            line2=line2.strip()
            examples.append(
                Example(
                    source=line1,
                    target=line2,
                    lang=srclang
                        ) 
            )

    return examples

## [TA] [MOD] END

## (TA) [ADD] Added to retrieve set of source files from a folder 

def read_folder_examples(srclang, srcext, trgext, filename):
    """Read examples from filename."""
    examples=[]
    source,target=filename.split(',')
    
    srcFold = Path(source)    
    ## print('Source: ', source)
    trgFold = Path(target)
    ## print('Target: ', target)
    if srcFold.is_dir() and trgFold.is_dir():     
        for spath in srcFold.glob('**/*.'+srcext):
            dname = spath.name[0:spath.name.index('.'+srcext)]+'.'+trgext
            ## print('Java File: ', str(spath))
            tpath = Path(target+'/'+dname)
            ## print('Coolgen File: ', str(tpath))
            if os.path.exists(tpath):
                examples.append(
                    Example(
                        source=read_source(str(spath)),
                        target=read_source(str(tpath)),
                        lang=srclang
                            ) 
                    )
                
    return examples

## [TA] [ADD] END

def read_source(filename):
    text = ''
    with open(filename,encoding="utf-8") as sf:
        for line in sf:
            line=line.strip()
            text += line + '\n'
    
    ## print(text)
            
    return text
    
class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 example_id,
                 source_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,                 
                 target_ids,
                 source_mask,
                 target_mask,

    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.position_idx = position_idx
        self.dfg_to_code = dfg_to_code
        self.dfg_to_dfg = dfg_to_dfg
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask       
        

parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser
    
def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []
    for example_index, example in enumerate(tqdm(examples,total=len(examples))):
        ##extract data flow
        code_tokens,dfg=extract_dataflow(example.source,
                                         parsers[args.source_lang],
                                         args.source_lang)
        code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
        ori2cur_pos={}
        ori2cur_pos[-1]=(0,0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
        code_tokens=[y for x in code_tokens for y in x]  
        
        #truncating
        code_tokens=code_tokens[:args.max_source_length-3][:512-3]
        source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
        dfg=dfg[:args.max_source_length-len(source_tokens)]
        source_tokens+=[x[0] for x in dfg]
        position_idx+=[0 for x in dfg]
        source_ids+=[tokenizer.unk_token_id for x in dfg]
        padding_length=args.max_source_length-len(source_ids)
        position_idx+=[tokenizer.pad_token_id]*padding_length
        source_ids+=[tokenizer.pad_token_id]*padding_length      
        source_mask = [1] * (len(source_tokens))
        source_mask+=[0]*padding_length        
        
        #reindex
        reverse_index={}
        for idx,x in enumerate(dfg):
            reverse_index[x[1]]=idx
        for idx,x in enumerate(dfg):
            dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
        dfg_to_dfg=[x[-1] for x in dfg]
        dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
        length=len([tokenizer.cls_token])
        dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]        
      

        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length-2]
        target_tokens = [tokenizer.cls_token]+target_tokens+[tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length   
   
        if example_index < 5:
            if stage=='train':
                logger.info("*** Example ***")
                logger.info("source_tokens: {}".format([x.replace('\u0120','_') for x in source_tokens]))
                logger.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(' '.join(map(str, source_mask))))
                logger.info("position_idx: {}".format(position_idx))
                logger.info("dfg_to_code: {}".format(' '.join(map(str, dfg_to_code))))
                logger.info("dfg_to_dfg: {}".format(' '.join(map(str, dfg_to_dfg))))
                
                logger.info("target_tokens: {}".format([x.replace('\u0120','_') for x in target_tokens]))
                logger.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(' '.join(map(str, target_mask))))
       
        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,
                 target_ids,
                 source_mask,
                 target_mask,
            )
        )
    return features

class TextDataset(Dataset):
    def __init__(self, examples, args):
        self.examples = examples
        self.args=args  
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):
        #calculate graph-guided masked function
        attn_mask=np.zeros((self.args.max_source_length,self.args.max_source_length),dtype=bool)
        #calculate begin index of node and max length of input
        node_index=sum([i>1 for i in self.examples[item].position_idx])
        max_length=sum([i!=1 for i in self.examples[item].position_idx])
        #sequence can attend to sequence
        attn_mask[:node_index,:node_index]=True
        #special tokens attend to all tokens
        for idx,i in enumerate(self.examples[item].source_ids):
            if i in [0,2]:
                attn_mask[idx,:max_length]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(self.examples[item].dfg_to_code):
            if a<node_index and b<node_index:
                attn_mask[idx+node_index,a:b]=True
                attn_mask[a:b,idx+node_index]=True
        #nodes attend to adjacent nodes         
        for idx,nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a+node_index<len(self.examples[item].position_idx):
                    attn_mask[idx+node_index,a+node_index]=True  
                    
        return (torch.tensor(self.examples[item].source_ids),
                torch.tensor(self.examples[item].source_mask),
                torch.tensor(self.examples[item].position_idx),
                torch.tensor(attn_mask), 
                torch.tensor(self.examples[item].target_ids),
                torch.tensor(self.examples[item].target_mask),)
    
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser()
    
    argset = params_training if FLAG_MODE == 1 else params_test
    
    ## Required parameters  
    parser.add_argument("--model_type", default=argset['--model_type'], type=str, required=False,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=argset['--model_name_or_path'], type=str, required=False,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--output_dir", default=argset['--output_dir'], type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=argset['--load_model_path'], type=str, 
                        help="Path to trained model: Should contain the .bin files" )    
    ## Other parameters
    parser.add_argument("--train_filename", default=argset['--train_filename'], type=str, 
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--train_directories", default=argset['--train_directories'], type=str, 
                        help="The train directories. Should contain the source and target language files.")
    parser.add_argument("--dev_filename", default=argset['--dev_filename'], type=str, 
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_directories", default=argset['--dev_directories'], type=str, 
                        help="The dev directories. Should contain files for this training evaluation task.")
    parser.add_argument("--test_filename", default=argset['--test_filename'], type=str, 
                        help="The test filename. Should contain the .jsonl files for this task.")  
    parser.add_argument("--test_directories", default=argset['--test_directories'], type=str, 
                        help="The test directories. Should contain files for testing.")  

    parser.add_argument("--source_lang", default=argset['--source_lang'], type=str, 
                        help="The language of input")  
    parser.add_argument("--target_lang", default=argset['--target_lang'], type=str, 
                        help="The language of output")  
    parser.add_argument("--source_ext", default=argset['--source_ext'], type=str, 
                        help="The source language file extension")  
    parser.add_argument("--target_ext", default=argset['--target_ext'], type=str, 
                        help="The target language file extension")  
    parser.add_argument("--config_name", default=argset['--config_name'], type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default=argset['--tokenizer_name'], type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name") 
    parser.add_argument("--max_source_length", default=argset['--max_source_length'], type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=argset['--max_target_length'], type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    parser.add_argument("--do_train", default=argset['--do_train'], action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=argset['--do_eval'], action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", default=argset['--do_test'], action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--train_batch_size", default=argset['--train_batch_size'], type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=argset['--eval_batch_size'], type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=argset['--learning_rate'], type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=argset['--beam_size'], type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=argset['--num_train_epochs'], type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")   
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    # print arguments
    args = parser.parse_args()
    
    print(args.source_lang)
    print(args.target_lang)
    print(args.source_ext)
    print(args.target_ext)
    print(args.model_type)
    print(args.model_name_or_path)
    print(args.output_dir)
    print(args.load_model_path)
    print(args.train_filename)
    print(args.train_directories)
    print(args.dev_filename)
    print(args.dev_directories)
    print(args.test_filename)
    print(args.test_directories)
    print(args.config_name)
    print(args.tokenizer_name)
    print(args.max_source_length)
    print(args.max_target_length)
    print(args.do_train)
    print(args.do_eval)
    print(args.do_test)
    print(args.do_lower_case)
    print(args.no_cuda)
    print(args.train_batch_size)
    print(args.eval_batch_size)
    print(args.gradient_accumulation_steps)
    print(args.learning_rate)
    print(args.beam_size)
    print(args.weight_decay)
    print(args.adam_epsilon)
    print(args.max_grad_norm)
    print(args.num_train_epochs)
    print(args.max_steps)
    print(args.eval_steps)
    print(args.train_steps)
    print(args.warmup_steps)
    print(args.local_rank)
    print(args.seed)
    
    read_folder_examples(args.source_lang, args.source_ext, args.target_ext, args.train_directories)
    
    files = []
    if args.dev_filename is not None:
        files.append(args.dev_filename)
    if args.test_filename is not None:
        files.append(args.test_filename)
    for idx, file in enumerate(files):   
        print(file)
        print("Test file: {}".format(file))

    
def main_m():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    parser = argparse.ArgumentParser()
    
    argset = params_training if FLAG_MODE == 1 else params_test
    
    ## (TA) [MOD] Modified all arguments to set the defaults from preset variable
    
    ## Required parameters  
    parser.add_argument("--model_type", default=argset['--model_type'], type=str, required=False,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=argset['--model_name_or_path'], type=str, required=False,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--output_dir", default=argset['--output_dir'], type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=argset['--load_model_path'], type=str, 
                        help="Path to trained model: Should contain the .bin files" )    
    ## Other parameters
    parser.add_argument("--train_filename", default=argset['--train_filename'], type=str, 
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--train_directories", default=argset['--train_directories'], type=str, 
                        help="The train directories. Should contain the source and target language files.")
    parser.add_argument("--dev_filename", default=argset['--dev_filename'], type=str, 
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_directories", default=argset['--dev_directories'], type=str, 
                        help="The dev directories. Should contain files for this training evaluation task.")
    parser.add_argument("--test_filename", default=argset['--test_filename'], type=str, 
                        help="The test filename. Should contain the .jsonl files for this task.")  
    parser.add_argument("--test_directories", default=argset['--test_directories'], type=str, 
                        help="The test directories. Should contain files for testing.")  

    parser.add_argument("--source_lang", default=argset['--source_lang'], type=str, 
                        help="The language of input")  
    parser.add_argument("--target_lang", default=argset['--target_lang'], type=str, 
                        help="The language of output")  
    parser.add_argument("--source_ext", default=argset['--source_ext'], type=str, 
                        help="The source language file extension")  
    parser.add_argument("--target_ext", default=argset['--target_ext'], type=str, 
                        help="The target language file extension")  
    parser.add_argument("--config_name", default=argset['--config_name'], type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default=argset['--tokenizer_name'], type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name") 
    parser.add_argument("--max_source_length", default=argset['--max_source_length'], type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=argset['--max_target_length'], type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", default=argset['--no_cuda'], action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--train_batch_size", default=argset['--train_batch_size'], type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=argset['--eval_batch_size'], type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=argset['--learning_rate'], type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=argset['--beam_size'], type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=argset['--num_train_epochs'], type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")   
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    ## [TA] [MOD] END
    """
    ## Required parameters  
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str, 
                        help="Path to trained model: Should contain the .bin files" )    
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str, 
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--train_directories", default=None, type=str, 
                        help="The train directories. Should contain the source and target language files.")
    parser.add_argument("--dev_filename", default=None, type=str, 
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_directories", default=None, type=str, 
                        help="The dev directories. Should contain files for this training evaluation task.")
    parser.add_argument("--test_filename", default=None, type=str, 
                        help="The test filename. Should contain the .jsonl files for this task.")  
    parser.add_argument("--test_directories", default=None, type=str, 
                        help="The test directories. Should contain files for testing.")  

    parser.add_argument("--source_lang", default=None, type=str, 
                        help="The language of input")  
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--source_ext", default=argset['--source_ext'], type=str, 
                        help="The source language file extension")  
    parser.add_argument("--target_ext", default=argset['--target_ext'], type=str, 
                        help="The target language file extension")  
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name") 
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")   
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    """
    
    # print arguments
    args = parser.parse_args()
    logger.info(args)
   

    # Setup CUDA, GPU
    if torch.cuda.is_available():
        print("we have cuda")
    else:
        print("we don't have cuda")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    
    # Set seed
    set_seed(args.seed)
    
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
        
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    
    #budild model
    if args.model_type == 'roberta':
        encoder = model_class.from_pretrained(args.model_name_or_path,config=config)    
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                      beam_size=args.beam_size,max_length=args.max_target_length,
                      sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    else:
        model = model_class.from_pretrained(args.model_name_or_path)
    
    ## (TA) [ADD] Added to optimize cuda memory
    #torch.cuda.empty_cache()
    ## [TA] [ADD] END
    
    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
        
    model.to(device)
    if args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        # Prepare training data loader
        
        ## (TA) [MOD] Modified to form a list of source contents from each file in a directory
        trainRep = args.train_directories
        train_examples = None
        if trainRep is not None:
            train_examples = read_folder_examples(args.source_lang, args.source_ext, args.target_ext, trainRep)
        else:
            train_examples = read_examples(args.source_lang, args.train_filename)
        ## [TA] [MOD] END
        
        train_features = convert_examples_to_features(train_examples, tokenizer, args, stage='train')
        train_data = TextDataset(train_features,args)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size//args.gradient_accumulation_steps,num_workers=4)

        num_train_optimization_steps =  args.train_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader)*args.num_train_epochs*0.1,num_training_steps=len(train_dataloader)*args.num_train_epochs)
    
        #Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", args.num_train_epochs)
        
        model.train()
        dev_dataset={}
        nb_tr_examples, nb_tr_steps,tr_loss,global_step,best_bleu,best_loss = 0, 0,0,0,0,1e6 
        for epoch in range(args.num_train_epochs):
            bar = tqdm(train_dataloader,total=len(train_dataloader))
            for batch in bar:
                batch = tuple(t.to(device) for t in batch)
                source_ids,source_mask,position_idx,att_mask,target_ids,target_mask = batch
                if args.model_type == 'roberta':
                    loss,_,_ = model(source_ids,source_mask,position_idx,att_mask,target_ids,target_mask)
                else:
                    outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                    labels=target_ids, decoder_attention_mask=target_mask)
                    loss = outputs.loss

                if args.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    
                tr_loss += loss.item()
                train_loss=round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
                bar.set_description("epoch {} loss {}".format(epoch,train_loss))
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                    #Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

            if args.do_eval and epoch in [ int(args.num_train_epochs*(i+1)//20) for i in range(20)]:
                #Eval model with dev dataset
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0                     
                eval_flag=False    
                if 'dev_loss' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_loss']
                else:
                    ## (TA) [MOD] Modified to form a list of source contents from each file in a directory
                    evalRep = args.dev_directories
                    eval_examples = None
                    if evalRep is not None:
                        eval_examples = read_folder_examples(args.source_lang, args.source_ext, args.target_ext, evalRep)
                    else:
                        eval_examples = read_examples(args.source_lang, args.dev_filename)
                    ## [TA] [MOD] END
                    
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='dev')
                    eval_data = TextDataset(eval_features,args)
                    dev_dataset['dev_loss']=eval_examples,eval_data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4)

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                #Start Evaling model
                model.eval()
                eval_loss,tokens_num = 0,0
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)               
                    source_ids,source_mask,position_idx,att_mask,target_ids,target_mask = batch
                    with torch.no_grad():
                        if args.model_type == 'roberta':
                            _,loss,num = model(source_ids,source_mask,position_idx,att_mask,target_ids,target_mask)
                            eval_loss += loss.sum().item()
                            tokens_num += num.sum().item()
                        else:
                            outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                    labels=target_ids, decoder_attention_mask=target_mask)
                            loss = outputs.loss
                            eval_loss += loss.sum().item()
                            tokens_num += 1
                #Pring loss of dev dataset    
                model.train()
                eval_loss = eval_loss / tokens_num
                result = {'eval_ppl': round(np.exp(eval_loss),5),
                          'global_step': global_step+1,
                          'train_loss': round(train_loss,5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  "+"*"*20)   

                #save last checkpoint
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)                    
                if eval_loss<best_loss:
                    logger.info("  Best ppl:%s",round(np.exp(eval_loss),5))
                    logger.info("  "+"*"*20)
                    best_loss=eval_loss
                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)  


                #Calculate bleu  
                if 'dev_bleu' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_bleu']
                else:
                    ## (TA) [MOD] Modified to form a list of source contents from each file in a directory
                    evalRep = args.dev_directories
                    eval_examples = None
                    if evalRep is not None:
                        eval_examples = read_folder_examples(args.source_lang, args.source_ext, args.target_ext, evalRep)
                    else:
                        eval_examples = read_examples(args.source_lang, args.dev_filename)
                    ## [TA] [MOD] END
                    eval_examples = random.sample(eval_examples,min(1000,len(eval_examples)))
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
                    eval_data = TextDataset(eval_features,args)
                    dev_dataset['dev_bleu']=eval_examples,eval_data

                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4)
                model.eval() 
                pred_ids=[]
                p=[]
                if args.model_type == 'roberta':
                    for batch in eval_dataloader:
                        batch = tuple(t.to(device) for t in batch)
                        source_ids,source_mask,position_idx,att_mask,target_ids,target_mask = batch                 
                        with torch.no_grad():
                            preds = model(source_ids,source_mask,position_idx,att_mask)  
                            for pred in preds:
                                t=pred[0].cpu().numpy()
                                t=list(t)
                                if 0 in t:
                                    t=t[:t.index(0)]
                                text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                                p.append(text)
                else:
                
                    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format('dev')):
                                        
                        with torch.no_grad():
                            source_ids = batch[0].to(args.device)
                            source_mask = source_ids.ne(tokenizer.pad_token_id)
                            preds = model.generate(source_ids,
                                                   attention_mask=source_mask,
                                                   use_cache=True,
                                                   num_beams=args.beam_size,
                                                   max_length=args.max_target_length)
                            top_preds = list(preds.cpu().numpy())
                            pred_ids.extend(top_preds)
                            for id in pred_ids:
                                id = list(id)
                                for i in id:
                                    if i > 1099:
                                        id[id.index(i)] = 1099
                                p.append(tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False))
                model.train()
                predictions=[]
                accs=[]
                
                ## (TA) [MOD] Modified to output evaluation sources into separate files
                testIdx = 1
                for ref,gold in zip(p,eval_examples):
                    #with open(os.path.join(args.output_dir,"devout_{}.".format(str(testIdx))+args.target_ext),'w') as f, open(os.path.join(args.output_dir,"devgold_{}.".format(str(testIdx))+args.target_ext),'w') as f1:
                    with open(os.path.join(args.output_dir,"dev.output"),'w') as f, open(os.path.join(args.output_dir,"dev.gold"),'w') as f1:
                        predictions.append(ref)
                        f.write(ref)
                        f.close
                        f1.write(gold.target+'\n')
                        f1.close     
                        accs.append(ref==gold.target)
                    testIdx += 1
                ## [TA] [MOD] END
                    
                dev_bleu=round(_bleu(os.path.join(args.output_dir, "dev.gold"), os.path.join(args.output_dir, "dev.output")),2)
                xmatch=round(np.mean(accs)*100,4)
                logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
                logger.info("  %s = %s "%("xMatch",str(round(np.mean(accs)*100,4))))
                logger.info("  "+"*"*20)    
                if dev_bleu+xmatch>best_bleu:
                    logger.info("  Best BLEU+xMatch:%s",dev_bleu+xmatch)
                    logger.info("  "+"*"*20)
                    best_bleu=dev_bleu+xmatch
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
               
    if args.do_test:
        ## (TA) [MOD] Modified to form a list of source contents from each file in a directory
        testRep = args.test_directories
        files=[]
        if testRep is not None:
            if args.dev_directories is not None:
                files.append(args.dev_directories)
            files.append(args.test_directories)
        else:
            if args.dev_filename is not None:
                files.append(args.dev_filename)
            if args.test_filename is not None:
                files.append(args.test_filename)
        ## [TA] [MOD] END
        
        for idx,file in enumerate(files):   
            logger.info("Test file: {}".format(file))
            
            ## (TA) [MOD] Modified to form a list of source contents from each file in a directory
            eval_examples = None
            if testRep is not None:
                eval_examples = read_folder_examples(args.source_lang, args.source_ext, args.target_ext, file)
            else:
                eval_examples = read_examples(args.source_lang, file)
            ## [TA] [MOD] END
            print("number of examples: " + str(eval_examples))
            for x in eval_examples:
                print(str(x.source) + " " + str(x.target))
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
            eval_data = TextDataset(eval_features, args) 

            # Calculate bleu
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4)

            model.eval() 
            pred_ids=[]
            p=[]
            if args.model_type == 'roberta':
                for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
                    batch = tuple(t.to(device) for t in batch)
                    source_ids,source_mask,position_idx,att_mask,target_ids,target_mask = batch                    
                    with torch.no_grad():
                        preds = model(source_ids,source_mask,position_idx,att_mask)  
                        for pred in preds:
                            t=pred[0].cpu().numpy()
                            t=list(t)
                            if 0 in t:
                                t=t[:t.index(0)]
                            text = tokenizer.decode(t,clean_up_tokenization_spaces=False)
                            p.append(text)
            else:
                for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format('dev')):
                                    
                    with torch.no_grad():
                        source_ids = batch[0].to(args.device)
                        source_mask = source_ids.ne(tokenizer.pad_token_id)
                        preds = model.generate(source_ids,
                                               attention_mask=source_mask,
                                               use_cache=True,
                                               num_beams=args.beam_size,
                                               max_length=args.max_target_length)
                        top_preds = list(preds.cpu().numpy())
                        pred_ids.extend(top_preds)
                        for id in pred_ids:
                            id = list(id)
                            for i in id:
                                if i > 1099:
                                    id[id.index(i)] = 1099
                            p.append(tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False))
            model.train()
            predictions=[]
            accs=[]
            
            ## (TA) [MOD] Modified to output individiual predicted source into a separate file
            testIdx = 1
            print("number of examples: " + str(len(eval_examples)))
            for x in eval_examples:
                print(str(x.source) + " " + str(x.target))
            for ref,gold in zip(p,eval_examples):
                #with open(os.path.join(args.output_dir,"test_{}.output.".format(str(testIdx))+args.target_ext),'w') as f, open(os.path.join(args.output_dir,"test_{}.gold.".format(str(testIdx))+args.target_ext),'w') as f1:
                with open(os.path.join(args.output_dir,"test_{}.output".format(str(idx))),'w') as f, open(os.path.join(args.output_dir,"test_{}.gold".format(str(idx))),'w') as f1:
                    predictions.append(ref)
                    f.write(ref)
                    f.close()
                    f1.write(gold.target)
                    f1.close()
                    accs.append(ref==gold.target)
                testIdx += 1
            ## [TA] [MOD] END
                
            dev_bleu=round(_bleu(os.path.join(args.output_dir, "test_{}.gold".format(str(idx))).format(file), 
                                 os.path.join(args.output_dir, "test_{}.output".format(str(idx))).format(file)),2)
            logger.info("  %s = %s "%("bleu-4",str(dev_bleu)))
            logger.info("  %s = %s "%("xMatch",str(round(np.mean(accs)*100,4))))
            logger.info("  "+"*"*20) 
            for x in predictions:
                print(x)  
            
if __name__ == "__main__":
    main_m()

