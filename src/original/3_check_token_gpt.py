"""
BERT MLM runner
"""

import logging
import os
import torch
import random
import numpy as np
import json
import matplotlib.pyplot as plt

from transformers import GPT2Tokenizer
from custom_gpt import GPT2LMHeadModel

# set logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


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
    target_bag = 3

    tmp_data_path = "../../data/original/data.json"
    gpt2_model = "gpt2"
    max_seq_length = 128
    seed = 42
    # Integrated Gradients Params
    batch_size = 20
    num_batch = 1

    allpos_dir = "../../results/allpos"
    allpos_kn = []
    max_igs = 0
    with open(os.path.join(allpos_dir, f"gpt2-{target_rel}-bag{target_bag}.json"), "r") as f:
        raw = json.load(f)
        max_igs = raw["max_igs"]
        for kn in raw["kn"]:
            allpos_kn.append((kn[0][0], kn[0][1]))
    print(allpos_kn)

    threshold_ratio = 0.2
    mode_ratio_bag = 0.7

    # prepare eval set
    with open(tmp_data_path, "r") as f:
        # Array<[SentenceWithMask, Answer, Relation]>
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

    n_layer = model.module.transformer.config.n_layer
    print("n_layer", n_layer)

    for i, (eval_example, max_ig) in enumerate(zip(sentences, max_igs)):
        input_text = eval_example[0]
        gold_label_text = eval_example[1]
        inputs = tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        input_readable_tokens = tokenizer.batch_decode(input_ids[0])
        gold_label = tokenizer.encode(" " + gold_label_text)
        input_len = input_ids.shape[1]

        # Check probability
        _, lm_logits, _ = model(input_ids=input_ids)
        logits = lm_logits[0, -1]
        pred_label_id = logits.argmax()
        values, indices = torch.topk(logits, 10)
        for rank, index in enumerate(indices):
            print(f"top {rank+1}: {tokenizer.decode(index)} ({index})")
        pred_label = tokenizer.convert_ids_to_tokens(pred_label_id.item())

        gold_p = logits[gold_label]
        print(logits.shape)
        rank = (logits > gold_p).sum().item()
        print("rank", rank, gold_label)

        heatmap = np.zeros((input_len, n_layer))
        for tgt_layer in range(n_layer):
            for tgt_pos in range(input_len):
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
                # ig = ig[ig >= max_ig * threshold_ratio]
                token = input_ids[0, tgt_pos]
                for ig_idx, v in enumerate(ig.tolist()):
                    # if tgt_layer == 2 and ig_idx == 2842:
                    #     print("hoge")
                    if v >= max_ig * threshold_ratio:
                        if (tgt_layer, ig_idx) in allpos_kn:
                            relative_v = v / max_ig
                            print(f"({tgt_layer}, {ig_idx}) fired at pos {tgt_pos} ({token}), v={v} ({relative_v})")
                            heatmap[tgt_pos, tgt_layer] += 1
        print(input_readable_tokens)
        plt.yticks(range(input_len), [s.strip() for s in input_readable_tokens])
        im = plt.imshow(heatmap, cmap="Purples")
        plt.colorbar(im)
        plt.text(3, 8, f"pred: {pred_label}, gold rank: {rank}")
        plt.savefig(os.path.join(allpos_dir, f"gpt2-{target_rel}-P{target_bag}-{i}.pdf"))
        plt.close()