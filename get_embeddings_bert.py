# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
from pytorch_pretrained_bert.modeling import BertForPreTraining
from pytorch_pretrained_bert.tokenization import BertTokenizer
import sys
import gc
from progressbar import ProgressBar
from joblib import dump, load
import os
import re

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, text_id, url):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.text_id = text_id
        self.url = url


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, overlap, text_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.overlap = overlap
        self.text_id = text_id


class SamplesProcessor():
    """Processor for samples."""

    def get_dev_examples(self, data_dir):
        return self._create_examples(data_dir, 'train')

    def _create_examples(self, data_dir, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        df = pd.read_csv(data_dir)
        for i, line in df.iterrows():
            guid = "%s-%s" % (set_type, i)
            text = str(line.full_text)
            examples.append(
                InputExample(guid=guid, text=text, text_id=line.id, url=line.original_url))
        return examples


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        overlap_list = []
        example_para = re.split('\n', example.text)
        for para in example_para:
            tokens = tokenizer.tokenize(para)
            # if segment of text is more than max_seg_length, split into two each of size max_seg_length with overlap
            if len(tokens) > max_seq_length - 2:
                if len(tokens) > 2 * (max_seq_length - 2):
                    tokens = tokens[:2 * (max_seq_length - 2)]
                tokens_1 = tokens[:(max_seq_length - 2)]
                tokens_2 = tokens[-(max_seq_length - 2):]
                tokens_1 = ["[CLS]"] + tokens_1 + ["[SEP]"]
                tokens_2 = ["[CLS]"] + tokens_2 + ["[SEP]"]
                segment_ids_1 = [0] * len(tokens_1)
                segment_ids_2 = [0] * len(tokens_2)
                input_ids_1 = tokenizer.convert_tokens_to_ids(tokens_1)
                input_ids_2 = tokenizer.convert_tokens_to_ids(tokens_2)
                input_mask_1 = [1] * len(input_ids_1)
                input_mask_2 = [1] * len(input_ids_2)
                padding = [0] * (max_seq_length - len(input_ids_1))
                input_ids_1 += padding
                input_mask_1 += padding
                segment_ids_1 += padding
                padding = [0] * (max_seq_length - len(input_ids_2))
                input_ids_2 += padding
                input_mask_2 += padding
                segment_ids_2 += padding
                assert len(input_ids_1) == max_seq_length
                assert len(input_mask_1) == max_seq_length
                assert len(segment_ids_1) == max_seq_length
                assert len(input_ids_2) == max_seq_length
                assert len(input_mask_2) == max_seq_length
                assert len(segment_ids_2) == max_seq_length
                overlap = (max_seq_length - 2) - (len(tokens) - (max_seq_length - 2))
                input_ids = [input_ids_1, input_ids_2]
                input_mask = [input_mask_1, input_mask_2]
                segment_ids = [segment_ids_1, segment_ids_2]

            else:
                tokens = ["[CLS]"] + tokens + ["[SEP]"]
                segment_ids = [0] * len(tokens)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding = [0] * (max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                overlap = 0

            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
            overlap_list.append(overlap)

        features.append(
            InputFeatures(input_ids=input_ids_list, input_mask=input_mask_list, segment_ids=segment_ids_list,
                          overlap=overlap_list, text_id=example.text_id))
    return features


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--model_file",
                        default=None,
                        type=str,
                        required=True,
                        help="Where the pre-trained/fine-tuned model is stored for loading.")
    parser.add_argument("--override_features",
                        default=False,
                        type=bool,
                        required=True,
                        help="Override pickled feature files.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the output files will be written.")
    ## Other parameters

    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--untuned',
                        action='store_true',
                        help="Whether to use fine-tuned BERT on finance articles")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    model = BertForPreTraining.from_pretrained(args.bert_model)
    if not args.untuned:
        model.load_state_dict(torch.load(args.model_file))
        print('Loaded model')

    if args.fp16:
        model.half()
    model.to(device)

    # Prepare optimizer
    processor = SamplesProcessor()
    if not os.path.isfile('eval_features.gz'):
        # save processed articles into features
        eval_examples = processor.get_dev_examples(args.data_file)
        eval_features = convert_examples_to_features(eval_examples, args.max_seq_length, tokenizer)
        dump(eval_features, 'eval_features.gz')
    else:
        if not args.override_features:
            eval_features = load('eval_features.gz')
        else:
            # override processed articles into features
            eval_examples = processor.get_dev_examples(args.data_file)
            eval_features = convert_examples_to_features(eval_examples, args.max_seq_length, tokenizer)
            dump(eval_features, 'eval_features.gz')

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_features))

    model.eval()
    for eval_count, eval_feature in enumerate(eval_features):
        if os.path.isfile(args.output_dir + '/embedding_' + str(eval_feature.text_id) + '.gz'):
            continue
        para_embed_list = []
        for para in range(len(eval_feature.input_ids)):
            #  if segment has no overlap
            if eval_feature.overlap[para] == 0:
                encoded_layer_array = np.zeros((0, args.max_seq_length, 1024))
                input_ids = torch.tensor(eval_feature.input_ids[para]).view(1, -1)
                input_mask = torch.tensor(eval_feature.input_mask[para]).view(1, -1)
                segment_ids = torch.tensor(eval_feature.segment_ids[para]).view(1, -1)
                para_len = np.sum(np.array(eval_feature.input_mask[para]) != 0)
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                encoded_layers, _ = model.bert.forward(input_ids, segment_ids, input_mask)
                for encoded_layer in encoded_layers:
                    encoded_layer_array = np.concatenate((encoded_layer.detach().cpu().numpy(), encoded_layer_array))
                encoded_layers = encoded_layer_array[-2, 1:para_len - 1, :]
            else:
                #  if segment has overlap
                encoded_layer_array_1 = np.zeros((0, args.max_seq_length, 1024))
                encoded_layer_array_2 = np.zeros((0, args.max_seq_length, 1024))
                para_len_1 = np.sum(np.array(eval_feature.input_mask[para][0]) != 0)
                para_len_2 = np.sum(np.array(eval_feature.input_mask[para][1]) != 0)
                input_ids_1 = torch.tensor(eval_feature.input_ids[para][0]).view(1, -1)
                input_mask_1 = torch.tensor(eval_feature.input_mask[para][0]).view(1, -1)
                segment_ids_1 = torch.tensor(eval_feature.segment_ids[para][0]).view(1, -1)
                input_ids_1 = input_ids_1.to(device)
                input_mask_1 = input_mask_1.to(device)
                segment_ids_1 = segment_ids_1.to(device)
                encoded_layers_1, _ = model.bert.forward(input_ids_1, segment_ids_1, input_mask_1)
                for encoded_layer in encoded_layers_1:
                    encoded_layer_array_1 = np.concatenate(
                        (encoded_layer.detach().cpu().numpy(), encoded_layer_array_1))
                encoded_layers_1 = encoded_layer_array_1[-2, 1:para_len_1 - 1, :]

                input_ids_2 = torch.tensor(eval_feature.input_ids[para][1]).view(1, -1)
                input_mask_2 = torch.tensor(eval_feature.input_mask[para][1]).view(1, -1)
                segment_ids_2 = torch.tensor(eval_feature.segment_ids[para][1]).view(1, -1)
                input_ids_2 = input_ids_2.to(device)
                input_mask_2 = input_mask_2.to(device)
                segment_ids_2 = segment_ids_2.to(device)
                encoded_layers_2, _ = model.bert.forward(input_ids_2, segment_ids_2, input_mask_2)
                for encoded_layer in encoded_layers_2:
                    encoded_layer_array_2 = np.concatenate(
                        (encoded_layer.detach().cpu().numpy(), encoded_layer_array_2))
                encoded_layers_2 = encoded_layer_array_2[-2, 1:para_len_2 - 1, :]
                # average the overlapped portion
                overlap = eval_feature.overlap[para]
                encoded_overlap = (encoded_layers_1[-overlap:, :] + encoded_layers_2[:overlap, :]) / 2
                encoded_layers = np.concatenate(
                    (encoded_layers_1[:-overlap, :], encoded_overlap, encoded_layers_2[overlap:, :]))
            para_embed_list.append(encoded_layers)
        dump(para_embed_list, args.output_dir + '/embedding_' + str(eval_feature.text_id) + '.gz')


if __name__ == "__main__":
    main()
