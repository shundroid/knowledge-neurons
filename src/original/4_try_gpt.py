import torch
import random
import numpy as np
from transformers import GPT2Tokenizer
from custom_gpt import GPT2LMHeadModel

seed = 42
device = torch.device("cuda:0")
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

gpt2_model = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)

torch.cuda.empty_cache()
model = GPT2LMHeadModel.from_pretrained(gpt2_model)
model.to(device)
model = torch.nn.DataParallel(model)
model.eval()


inputs = tokenizer("Alan Turing works in the field of", return_tensors="pt")
print(list(inputs))
print(inputs["input_ids"])
lm_logits, presents = model(
  input_ids=inputs["input_ids"],
)

print(lm_logits.shape)
pred_label_id = lm_logits[0, -1].argmax()
pred_label = tokenizer.convert_ids_to_tokens(pred_label_id.item())
print(pred_label)
