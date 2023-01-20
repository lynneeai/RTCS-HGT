import logging
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

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

from model.hgt import HGT
from model.dgl_graph import pos_neg_user_sampler


class RTCS_HGT(nn.Module):
    def __init__(self, G, hgt_n_inp_dict):
        super(RTCS_HGT, self).__init__()

        self.hgt = HGT(
            G=G, n_inp_dict=hgt_n_inp_dict, n_hid=MODEL_CONFIGS.HGT_N_HID, n_out=MODEL_CONFIGS.HGT_N_OUT, n_layers=MODEL_CONFIGS.HGT_N_LAYERS, n_heads=MODEL_CONFIGS.HGT_N_HEADS, dropout=MODEL_CONFIGS.HGT_DROPOUT, use_norm=MODEL_CONFIGS.HGT_USE_NORM
        )

        self.hgt_linear = nn.Linear(MODEL_CONFIGS.HGT_N_OUT, MODEL_CONFIGS.OUTPUT_EMBED_SIZE)
        self.out = nn.Linear(MODEL_CONFIGS.OUTPUT_EMBED_SIZE, MODEL_CONFIGS.N_CLASS)

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.hgt_linear.weight)
        nn.init.xavier_normal_(self.out.weight)

    def forward(self, sg, sg_nodes, output_nodes):
        hgt_sampled_out, user_embed = self.hgt(sg, "tweet")
        node2emb = {}
        for i, node in enumerate(sg_nodes):
            node2emb[node] = hgt_sampled_out[i]
        hgt_out = [node2emb[node] for node in output_nodes]
        hgt_out = torch.stack(hgt_out, dim=0)
        hgt_out = F.relu(self.hgt_linear(hgt_out))

        output = self.out(hgt_out)
        output = F.log_softmax(output, dim=1)

        return output, user_embed

    def loss(self, outputs, labels, prox_loss=False, sg=None, user_embed=None, sample_nodes_portion=None, sample_nodes_per_node=None):

        loss = F.nll_loss(outputs, labels)

        if prox_loss:
            all_user_nodes = [i for i in range(sg.number_of_nodes("user"))]
            node2emb = {all_user_nodes[i]: user_embed[i] for i in range(len(all_user_nodes))}
            nodes_to_sample = int(len(all_user_nodes) * sample_nodes_portion)
            sampled_nodes = random.sample(all_user_nodes, nodes_to_sample)

            prox_loss = []
            for node in sampled_nodes:
                cur_node_emb = node2emb[node]
                pos_nodes, neg_nodes = pos_neg_user_sampler(sg, node, sample_nodes_per_node)

                if pos_nodes and neg_nodes:
                    # positive score
                    pos_emb_list = [node2emb[node] for node in pos_nodes]
                    pos_embs = torch.stack(pos_emb_list, dim=0)
                    cur_node_emb_mask = cur_node_emb.expand(pos_embs.shape)
                    pos_sim_score = F.cosine_similarity(cur_node_emb_mask, pos_embs)
                    pos_score = torch.mean(torch.log(torch.sigmoid(pos_sim_score)), 0).to(TRAIN_CONFIGS.DEVICE)

                    # negative score
                    neg_emb_list = [node2emb[node] for node in neg_nodes]
                    neg_embs = torch.stack(neg_emb_list, dim=0)
                    cur_node_emb_mask = cur_node_emb.expand(neg_embs.shape)
                    neg_sim_score = F.cosine_similarity(cur_node_emb_mask, neg_embs)
                    neg_score = MODEL_CONFIGS.PROX_LOSS_Q * torch.mean(torch.log(torch.sigmoid(-neg_sim_score)), 0).to(TRAIN_CONFIGS.DEVICE)

                    prox_loss.append(torch.mean(-pos_score - neg_score).view(1, -1))

            if len(prox_loss) > 0:
                loss += MODEL_CONFIGS.PROX_LOSS_PERCENT * torch.mean(torch.cat(prox_loss, 0))

        return loss
