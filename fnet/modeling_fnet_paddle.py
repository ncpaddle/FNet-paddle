# coding=utf-8
# Copyright 2021 Google Research and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch FNet model. """
import copy

from functools import partial

import paddle.nn as nn
import paddle
from paddle.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import importlib
def is_scipy_available():
    return importlib.util.find_spec("scipy") is not None

if is_scipy_available():
    from scipy import linalg

import math
def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + paddle.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * paddle.pow(x, 3.0))))

ACT2FN = {
    "relu": paddle.nn.ReLU(),
    "gelu": paddle.nn.GELU(),
    "tanh": paddle.nn.Tanh(),
    "sigmoid": paddle.nn.Sigmoid(),
    'gelu_new': gelu_new,
}

from paddlenlp.transformers import PretrainedModel
from utilsx import apply_chunking_to_forward
import loggings as logging

import paddlenlp
import numpy as np


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "google/fnet-base"
_CONFIG_FOR_DOC = "FNetConfig"
_TOKENIZER_FOR_DOC = "FNetTokenizer"

FNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "fnet-base",
    "fnet-large"
    # See all FNet models at https://huggingface.co/models?filter=fnet
]


# Adapted from https://github.com/google-research/google-research/blob/master/f_net/fourier.py
def _two_dim_matmul(x, matrix_dim_one, matrix_dim_two):
    """Applies 2D matrix multiplication to 3D input arrays."""
    seq_length = x.shape[1]
    matrix_dim_one = matrix_dim_one[:seq_length, :seq_length]
    x = x.astype(paddle.complex64)
    return paddlenlp.ops.einsum("bij,jk,ni->bnk", x, matrix_dim_two, matrix_dim_one)


# # Adapted from https://github.com/google-research/google-research/blob/master/f_net/fourier.py
def two_dim_matmul(x, matrix_dim_one, matrix_dim_two):
    return _two_dim_matmul(x, matrix_dim_one, matrix_dim_two)


# Adapted from https://github.com/google-research/google-research/blob/master/f_net/fourier.py
def fftn(x):
    """
    Applies n-dimensional Fast Fourier Transform (FFT) to input array.

    Args:
        x: Input n-dimensional array.

    Returns:
        n-dimensional Fourier transform of input n-dimensional array.
    """
    out = x
    for axis in reversed(range(x.ndim)[1:]):  # We don't need to apply FFT to last axis
        out = paddle.fft.fft(out, axis=axis)
    return out



class FNetEmbeddings(nn.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        # NOTE: This is the project layer and will be needed. The original code allows for different embedding and different model dimensions.
        self.projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, input_ids=None, token_type_ids=None, position_ids=None):
        if input_ids is not None:
            input_shape = input_ids.shape

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + token_type_embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.projection(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class FNetBasicFourierTransform(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self._init_fourier_transform(config)

    def _init_fourier_transform(self, config):
        if not config.use_tpu_fourier_optimizations:
            self.fourier_transform = partial(paddle.fft.fftn, axes=(1, 2))
        elif config.max_position_embeddings <= 4096:
            if is_scipy_available():
                self.register_buffer(
                    "dft_mat_hidden", paddle.to_tensor(linalg.dft(config.hidden_size), dtype=paddle.complex64)
                )
                self.register_buffer(
                    "dft_mat_seq", paddle.to_tensor(linalg.dft(config.tpu_short_seq_length), dtype=paddle.complex64)
                )
                self.fourier_transform = partial(
                    two_dim_matmul, matrix_dim_one=self.dft_mat_seq, matrix_dim_two=self.dft_mat_hidden
                )
            else:
                logging.warning(
                    "SciPy is needed for DFT matrix calculation and is not found. Using TPU optimized fast fourier transform instead."
                )
                self.fourier_transform = fftn
        else:
            self.fourier_transform = fftn

    def forward(self, hidden_states):
        outputs = self.fourier_transform(hidden_states).real
        return (outputs,)


class FNetBasicOutput(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.LayerNorm(input_tensor + hidden_states)
        return hidden_states


class FNetFourierTransform(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.self = FNetBasicFourierTransform(config)
        self.output = FNetBasicOutput(config)

    def forward(self, hidden_states):
        self_outputs = self.self(hidden_states)
        fourier_output = self.output(self_outputs[0], hidden_states)
        outputs = (fourier_output,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->FNet
class FNetIntermediate(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->FNet
class FNetOutput(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FNetLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1  # The dimension which has the sequence length
        self.fourier = FNetFourierTransform(config)
        self.intermediate = FNetIntermediate(config)
        self.output = FNetOutput(config)

    def forward(self, hidden_states):
        self_fourier_outputs = self.fourier(hidden_states)
        fourier_output = self_fourier_outputs[0]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, fourier_output
        )

        outputs = (layer_output,)

        return outputs

    def feed_forward_chunk(self, fourier_output):
        intermediate_output = self.intermediate(fourier_output)
        layer_output = self.output(intermediate_output, fourier_output)
        return layer_output


class FNetEncoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.LayerList([copy.deepcopy(FNetLayer(config)) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states)

            hidden_states = layer_outputs[0]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)



# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->FNet
class FNetPooler(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->FNet
class FNetPredictionHeadTransform(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class FNetLMPredictionHead(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.transform = FNetPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = paddle.zeros(config.vocab_size)
        self.bias = paddle.create_parameter(
            shape=[config.vocab_size],
            dtype=paddle.float32,
            default_initializer=paddle.nn.initializer.Constant(0.0))
        # self.bias = nn.Parameter(paddle.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias



from paddlenlp.transformers import PretrainedModel, BertModel
class FNetPreTrainedModel(PretrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # config_class = FNetConfig
    base_model_prefix = "fnet"
    pretrained_init_configuration = {
        "roberta-wwm-ext": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 2,
            "vocab_size": 21128,
            "pad_token_id": 0
        }
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "roberta-wwm-ext":
                "https://paddlenlp.bj.bcebos.com/models/transformers/roberta_base/roberta_chn_base.pdparams",
        }
    }

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # only support dygraph, use truncated_normal and make it inplace
            # and configurable later
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else
                    self.config.initializer_range,
                    shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12



class FNetModel(FNetPreTrainedModel):
    """

    The model can behave as an encoder, following the architecture described in `FNet: Mixing Tokens with Fourier
    Transforms <https://arxiv.org/abs/2105.03824>`__ by James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, Santiago
    Ontanon.

    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embeddings = FNetEmbeddings(config)
        self.encoder = FNetEncoder(config)
        self.pooler = FNetPooler(config) if config.add_pooling_layer else None
        self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        output_hidden_states=None
    ):
        output_hidden_states = output_hidden_states if output_hidden_states else self.config.output_hidden_states
        return_dict = self.config.use_return_dict

        input_shape = input_ids.shape
        batch_size, seq_length = input_shape

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        pooler_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooler_output) + encoder_outputs[1:]


class FNetForSequenceClassification(FNetPreTrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.fnet = FNetModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.apply(self.init_weights)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        labels=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        return_dit: True or False
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.fnet(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == paddle.int64 or labels.dtype == paddle.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.reshape([-1, self.num_labels]), labels.reshape([-1]))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        dic = {
            'loss': loss,
            'logits': logits,
            'hidden_states': output_hidden_states
        }
        return dic




class Config(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith("__") and k.endswith("__")) and not k in (
                "update",
                "pop",
            ):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(Config, self).__setattr__(name, value)
        super(Config, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(Config, self).pop(k, d)


config_params = {
    # FNetForSequenceClassification
    "num_labels": 2,
    "hidden_dropout_prob": 0.1,
    "hidden_size": 1024,
    "output_hidden_states": 1024,
    "problem_type": "single_label_classification",

    # FNetModel
    "add_pooling_layer": True,

    # FNetEmbeddings
    "vocab_size": 32000,
    "max_position_embeddings": 512,
    "type_vocab_size": 4,
    "pad_token_id": 3,

    # FNetEncoder
    "num_hidden_layers": 24,

    # FNetLayer
    'chunk_size_feed_forward': 256,

    #
    "max_seq_length": 512,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "gelu_new",

    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "layer_norm_eps": 1e-12,
    "tpu_short_seq_length": 512,
    "use_fft": True,
    "use_tpu_fourier_optimizations": False,

}
config = Config(config_params)
model = FNetForSequenceClassification(config)