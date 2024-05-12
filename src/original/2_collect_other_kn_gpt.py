"""
BERT MLM runner
"""

import logging
import os
import torch
import random
import numpy as np
import json
from collections import Counter

from transformers import GPT2Tokenizer
from custom_gpt import GPT2LMHeadModel

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
    target_rel = "P463"
    target_bag = 0

    tmp_data_path = "../../data/original/data.json"
    gpt2_model = "gpt2"
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
        sentences = json.load(f)[target_rel][f"B{target_bag}"]

    # model setup
    device = torch.device("cuda:0")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model, do_lower_case=False)

    torch.cuda.empty_cache()
    model = GPT2LMHeadModel.from_pretrained(gpt2_model)
    model.to(device)
    model = torch.nn.DataParallel(model)
    model.eval()

    kn_candidates_bag = []
    max_igs = []

    cnt = Counter()
    for eval_example in sentences:
        input_text = eval_example[0]
        gold_label_text = eval_example[1]
        inputs = tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        gold_label = tokenizer.convert_tokens_to_ids(gold_label_text)
        input_len = input_ids.shape[1]

        kn_candidates = []

        max_ig = -999

        for tgt_layer in range(model.module.transformer.config.n_layer):
            for tgt_pos in range(input_len):
            # for tgt_pos in [mask_pos]:
                ffn_weights, logits, _ = model(input_ids=input_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer)
                scaled_weights, weights_step = scaled_input(ffn_weights, batch_size, num_batch)
                scaled_weights.requires_grad_(True)
                ig = torch.zeros(ffn_weights.shape[1]).to(device)
                for batch_idx in range(num_batch):
                    batch_weights = scaled_weights[batch_idx*batch_size:(batch_idx+1)*batch_size]
                    grad, _, _ = model(input_ids=input_ids, tgt_pos=tgt_pos, tgt_layer=tgt_layer, tmp_score=batch_weights, tgt_label=gold_label)
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
    
    with open(os.path.join(allpos_dir, f"gpt2-{target_rel}-bag{target_bag}.json"), "w") as f:
        json.dump(dump, f, indent=2)
