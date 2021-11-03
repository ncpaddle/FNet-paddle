import json
from reprod_log import ReprodLogger, ReprodDiffHelper

import paddle
import torch
from modeling_fnet_paddle import FNetModel as paddle_FNetModel
from transformers import FNetModel as torch_FNetModel, FNetTokenizer
import numpy as np

tokenizer = FNetTokenizer.from_pretrained('../fnet_large')
inputs = tokenizer.tokenize(['This is a very good day!', 'I was very sad!', 'I want to go to the school, but I do not like learning.'])

print(inputs)
assert 1 == 2

torch_inputs = {k: torch.tensor([v]) for k, v in inputs.items()}
paddle_inputs = {k: paddle.to_tensor([v]) for k, v in inputs.items()}

print("loading torch checkpoint...")
torch_model = torch_FNetModel.from_pretrained('../../../fnet_large/')
print("loading paddle checkpoint...")
paddle_model = paddle_FNetModel.from_pretrained('../../../fnet_large/')

torch_model.eval()
paddle_model.eval()


paddle_out = paddle_model(**paddle_inputs)[0].numpy()
torch_out = torch_model(**torch_inputs)[0].detach().numpy()


lr_paddle = ReprodLogger()
lr_paddle.add('sequence_output', paddle_out)
lr_paddle.save("forward_paddle.npy")

lr_torch = ReprodLogger()
lr_torch.add('sequence_output', torch_out)
lr_torch.save("forward_torch.npy")

diff = ReprodDiffHelper()
paddle_diff = diff.load_info("forward_paddle.npy")
torch_diff = diff.load_info("forward_torch.npy")
diff.compare_info(paddle_diff, torch_diff)
diff.report(diff_method='mean', diff_threshold=1e-5, path='../log_diff/model_diff.txt')