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
from collections import Counter

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
    target_rel = "P463"
    target_bag = 0

    tmp_data_path = "../../data/PARAREL/data_all_allbags.json"
    bert_model = "bert-base-cased"
    max_seq_length = 128
    seed = 42
    # Integrated Gradients Params
    batch_size = 20
    num_batch = 1

    threshold_ratio = 0.2
    mode_ratio_bag = 0.7

    allpos_dir = "../../results/allpos/"

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

    kn_candidates_bag = []
    max_igs = []

    cnt = Counter()
    for eval_example in sentences:
        eval_features, tokens_info = example2feature(eval_example, max_seq_length, tokenizer)
        baseline_ids, input_ids, input_mask, segment_ids = eval_features['baseline_ids'], eval_features['input_ids'], eval_features['input_mask'], eval_features['segment_ids']
        baseline_ids = torch.tensor(baseline_ids, dtype=torch.long).unsqueeze(0)
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        input_mask = torch.tensor(input_mask, dtype=torch.long).unsqueeze(0)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long).unsqueeze(0)
        baseline_ids = baseline_ids.to(device)
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        gold_label = tokenizer.convert_tokens_to_ids(tokens_info["gold_obj"])

        mask_pos = tokens_info["tokens"].index("[MASK]")
        print("mask_pos", mask_pos)

        kn_candidates = []

        max_ig = -999

        for tgt_layer in range(model.module.bert.config.num_hidden_layers):
            for tgt_pos in range(len(tokens_info["tokens"])):
            # for tgt_pos in [mask_pos]:
                ffn_weights, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, mask_pos=mask_pos, tgt_layer=tgt_layer)
                scaled_weights, weights_step = scaled_input(ffn_weights, batch_size, num_batch)
                scaled_weights.requires_grad_(True)
                ig = torch.zeros(ffn_weights.shape[1]).to(device)
                for batch_idx in range(num_batch):
                    batch_weights = scaled_weights[batch_idx*batch_size:(batch_idx+1)*batch_size]
                    _, grad = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, mask_pos=mask_pos, tgt_layer=tgt_layer, tmp_score=batch_weights, tgt_label=gold_label)
                    grad = grad.sum(dim=0)
                    ig = torch.add(ig, grad)
                ig = ig * weights_step
                max_ig = max(max_ig, torch.max(ig).item())
                # ig = ig[ig >= max_ig * threshold_ratio]
                for ig_idx, v in enumerate(ig.tolist()):
                    kn_candidates.append((tgt_layer, tgt_pos, ig_idx, v))

        max_igs.append(max_ig)
        # re-check
        kn_candidates = list(filter(lambda x: x[3] >= max_ig * threshold_ratio, kn_candidates))
        # sorted_candidates = sorted(kn_candidates, key=lambda x: -x[3])
        # print(sorted_candidates[0:10])
        for (l, _, idx, _) in kn_candidates:
            cnt.update([(l, idx)])

        kn_candidates_bag.append(kn_candidates)
        # for l, idx in cnt.most_common():
    dump = { "max_igs": max_igs, "kn": [] }
    for ((l, idx), count) in cnt.most_common(20):
        dump["kn"].append([[l, idx], count])
    
    with open(os.path.join(allpos_dir, f"{target_rel}-bag{target_bag}.json"), "w") as f:
        json.dump(dump, f, indent=2)

    # squeeze kn
    # for 
    # threshold_ratio = 0.2
    # mode_ratio_bag = 0.7
    # for max_it in range(6):
    #     ave

    # test run
    # _, logits = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=0)
    # pred_label_id = logits[0].argmax()
    # pred_label = tokenizer.convert_ids_to_tokens(pred_label_id.item())
    # print(sentences[sentence_idx], pred_label)

    # for tgt_layer in range(model.bert.config.num_hidden_layers):
    # print("prediction from each value slot of knowledge neurons")
    # for kn in kn_bag:
    #     layer = kn[0]
    #     pos = kn[1]
    #     value_slot = model.module.bert.encoder.layer[layer].output.dense.weight[:, pos]
    #     logits = model.module.cls(value_slot)
    #     label_id = logits.argmax()
    #     label = tokenizer.convert_ids_to_tokens(label_id.item())
    #     print(label)


    # print("layers: ", model.module.bert.config.num_hidden_layers)
    # for tgt_layer in range(model.bert.config.num_hidden_layers):
    #     ffn_weights, _ = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer)

    # main()
