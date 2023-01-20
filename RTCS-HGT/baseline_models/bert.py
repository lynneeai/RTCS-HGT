import logging
import os
import sys

import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = current_file_dir
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


class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()

        self.text_model = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-covid19-base-cased", output_hidden_states=True, attention_probs_dropout_prob=0.2, hidden_dropout_prob=0.2)
        self.text_linear = nn.Linear(self.text_model.config.hidden_size, 256)
        self.text_dropout = nn.Dropout(p=0.2)
        self.text_out = nn.Linear(256, 32)
        self.out = nn.Linear(32, MODEL_CONFIGS.N_CLASS)

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.text_linear.weight)
        nn.init.xavier_normal_(self.text_out.weight)
        nn.init.xavier_normal_(self.out.weight)

    def forward(self, tweets_input_ids, tweets_attention_mask):
        hidden_states = self.text_model(tweets_input_ids, attention_mask=tweets_attention_mask).hidden_states
        text_output = self.text_dropout(hidden_states[-1][:, 0, :])
        text_output = F.relu(self.text_linear(text_output))
        text_output = self.text_dropout(text_output)
        text_out = F.relu(self.text_out(text_output))

        output = self.out(text_out)
        output = F.log_softmax(output, dim=1)

        return output, None
