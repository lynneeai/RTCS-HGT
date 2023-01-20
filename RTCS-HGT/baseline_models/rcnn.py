import logging
import os
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from transformers import AutoModel

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""

from configs import TRAIN_CONFIGS, MODEL_CONFIGS
from utils import init_logger

"""Init logger"""
if not os.path.exists(TRAIN_CONFIGS.LOGS_ROOT):
    os.makedirs(TRAIN_CONFIGS.LOGS_ROOT)
log_filename = os.path.basename(__file__).split(".")[0]
init_logger(TRAIN_CONFIGS.LOGS_ROOT, log_filename)
LOGGER = logging.getLogger(log_filename)
"""------------------"""


class RCNNModel(nn.Module):
    def __init__(self, rnn_model="LSTM"):
        super(RCNNModel, self).__init__()

        """
        The model is inspired from: Lai, Siwei, et al. "Recurrent convolutional neural networks for text classification." Twenty-ninth AAAI conference on artificial intelligence. 2015.
        """

        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.5
        self.bidirectional = True
        self.rnn_model = rnn_model

        self.bert = AutoModel.from_pretrained("vinai/bertweet-covid19-base-cased")
        for name, param in self.bert.named_parameters():
            param.requires_grad = False

        if rnn_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=self.bert.config.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
            )
        elif rnn_model == "GRU":
            self.rnn = nn.GRU(
                input_size=self.bert.config.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
            )
        else:
            raise LookupError("only support LSTM and GRU")

        if self.bidirectional:
            layer_size = 2
        else:
            layer_size = 1

        self.linear = nn.Linear(layer_size * self.hidden_size + self.bert.config.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, MODEL_CONFIGS.N_CLASS)

    def forward(self, tweets_input_ids, tweets_attention_mask):

        batch_size = tweets_input_ids.shape[0]

        input = self.bert(tweets_input_ids, attention_mask=tweets_attention_mask)[0]
        input = input.permute(1, 0, 2)

        if self.bidirectional:
            layer_size = 2 * self.num_layers
        else:
            layer_size = 1 * self.num_layers

        h_0 = Variable(torch.zeros(layer_size, batch_size, self.hidden_size).to(TRAIN_CONFIGS.DEVICE))
        c_0 = Variable(torch.zeros(layer_size, batch_size, self.hidden_size).to(TRAIN_CONFIGS.DEVICE))

        if self.rnn_model == "LSTM":
            output, (h_n, c_n) = self.rnn(input, (h_0, c_0))

        elif self.rnn_model == "GRU":
            output, _ = self.rnn(input, h_0)

        concat_embeddings = torch.cat((output, input), 2).permute(1, 0, 2)
        embeddings = self.linear(concat_embeddings)

        embeddings = embeddings.permute(0, 2, 1)
        embeddings = F.max_pool1d(embeddings, embeddings.size()[2])
        embeddings = embeddings.squeeze(2)

        output = self.out(embeddings)
        output = F.log_softmax(output, dim=1)

        return output, None
