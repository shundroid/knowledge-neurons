"""
BERT MLM runner
"""

import logging
import argparse
import math
import os
import torch
import random
import numpy as np
import json, jsonlines
import pickle
import time

import transformers
from transformers import BertTokenizer
from custom_bert import BertForMaskedLM
import torch.nn.functional as F

# set logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def example2feature(example, max_seq_length, tokenizer):
    """Convert an example into input features"""
    features = []
    tokenslist = []

    ori_tokens = tokenizer.tokenize(example[0])
    # All templates are simple, almost no one will exceed the length limit.
    if len(ori_tokens) > max_seq_length - 2:
        ori_tokens = ori_tokens[:max_seq_length - 2]

    # add special tokens
    tokens = ["[CLS]"] + ori_tokens + ["[SEP]"]
    base_tokens = ["[UNK]"] + ["[UNK]"] * len(ori_tokens) + ["[UNK]"]
    segment_ids = [0] * len(tokens)

    # Generate id and attention mask
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    baseline_ids = tokenizer.convert_tokens_to_ids(base_tokens)
    input_mask = [1] * len(input_ids)

    # Pad [PAD] tokens (id in BERT-base-cased: 0) up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    baseline_ids += padding
    segment_ids += padding
    input_mask += padding

    assert len(baseline_ids) == max_seq_length
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    features = {
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
        'baseline_ids': baseline_ids,
    }
    tokens_info = {
        "tokens":tokens,
        "relation":example[2],
        "gold_obj":example[1],
        "pred_obj": None
    }
    return features, tokens_info


def scaled_input(emb, batch_size, num_batch):
    # emb: (1, ffn_size)
    baseline = torch.zeros_like(emb)  # (1, ffn_size)

    num_points = batch_size * num_batch
    step = (emb - baseline) / num_points  # (1, ffn_size)

    res = torch.cat([torch.add(baseline, step * i) for i in range(num_points)], dim=0)  # (num_points, ffn_size)
    return res, step[0]


def convert_to_triplet_ig(ig_list):
    ig_triplet = []
    ig = np.array(ig_list)  # 12, 3072
    max_ig = ig.max()
    for i in range(ig.shape[0]):
        for j in range(ig.shape[1]):
            if ig[i][j] >= max_ig * 0.1:
                ig_triplet.append([i, j, ig[i][j]])
    return ig_triplet

if __name__ == "__main__":
    kn_dir = "../../results/kn/"
    target_rel = "P108"
    target_bag = 149
    sentence_idx = 1

    tmp_data_path = "../../data/PARAREL/data_all_allbags.json"
    bert_model = "bert-base-cased"
    max_seq_length = 128
    seed = 42
    # Integrated Gradients Params
    batch_size = 20
    num_batch = 1

    # prepare eval set
    with open(tmp_data_path, "r") as f:
        # Array<[SentenceWithMask, Answer, Relation]>
        sentences = json.load(f)[target_rel][target_bag]

    # retrieve kn_bag_list
    with open(os.path.join(kn_dir, f"base_kn_bag-{target_rel}.json"), "r") as fr:
        kn_bag_list = json.load(fr)
    kn_bag = kn_bag_list[target_bag]
    print(kn_bag, sentences)

    # model setup
    device = torch.device("cuda:0")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=False)

    torch.cuda.empty_cache()
    model = BertForMaskedLM.from_pretrained(bert_model)
    model.to(device)
    model = torch.nn.DataParallel(model)
    model.eval()

    # test
    eval_features, tokens_info = example2feature(sentences[sentence_idx], max_seq_length, tokenizer)
    baseline_ids, input_ids, input_mask, segment_ids = eval_features['baseline_ids'], eval_features['input_ids'], eval_features['input_mask'], eval_features['segment_ids']
    baseline_ids = torch.tensor(baseline_ids, dtype=torch.long).unsqueeze(0)
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0)
    baseline_ids = baseline_ids.to(device)
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)

    tgt_pos = tokens_info["tokens"].index("[MASK]")

    # test run
    # _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0)
    # pred_label_id = logits[0].argmax()
    # pred_label = tokenizer.convert_ids_to_tokens(pred_label_id.item())
    # print(sentences[sentence_idx], pred_label)

    # for tgt_layer in range(model.bert.config.num_hidden_layers):
    print("prediction from each value slot of knowledge neurons")
    for kn in kn_bag:
        layer = kn[0]
        pos = kn[1]
        value_slot = model.module.bert.encoder.layer[layer].output.dense.weight[:, pos]
        logits = model.module.cls(value_slot)
        label_id = logits.argmax()
        label = tokenizer.convert_ids_to_tokens(label_id.item())
        print(label)


    print("layers: ", model.module.bert.config.num_hidden_layers)
    # for tgt_layer in range(model.bert.config.num_hidden_layers):
    #     ffn_weights, _ = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer)

    # main()