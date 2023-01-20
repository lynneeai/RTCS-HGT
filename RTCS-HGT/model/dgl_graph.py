import logging
import os
import pickle
import random
import sys

import dgl
import torch
import torch.nn as nn

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""

from configs import TRAIN_CONFIGS
from utils import init_logger

"""Init logger"""
if not os.path.exists(TRAIN_CONFIGS.LOGS_ROOT):
    os.makedirs(TRAIN_CONFIGS.LOGS_ROOT)
log_filename = os.path.basename(__file__).split(".")[0]
init_logger(TRAIN_CONFIGS.LOGS_ROOT, log_filename)
LOGGER = logging.getLogger(log_filename)
"""------------------"""


class DGL_Heterograph(object):
    def __init__(self, tweet_user_graph, data_file, reload_dgl_graph=False):
        super(DGL_Heterograph, self).__init__()

        self.root_data_dir = TRAIN_CONFIGS.DATA_ROOT
        self.tweet_user_graph = tweet_user_graph
        self.data_file = f"{self.root_data_dir}/{data_file}"

        self.G_edges = [("tweet", "tweeted", "user"), ("tweet", "retweeted_tu", "user"), ("user", "tweeting", "tweet"), ("user", "retweeting_ut", "tweet"), ("user", "retweeted_uu", "user"), ("user", "retweeting_uu", "user")]

        if reload_dgl_graph:
            edge_idx_dict, edge_weight_dict, x_tweets, x_users, y_tweets = self.reload_data(save_to_file=True)
        else:
            LOGGER.info(f"Loading G from {data_file}...")
            with open(self.data_file, "rb") as infile:
                edge_idx_dict, edge_weight_dict, x_tweets, x_users, y_tweets = pickle.load(infile)

        self.G = self.create_dgl_heterograph(edge_idx_dict, edge_weight_dict, x_tweets, x_users, y_tweets)

    def reload_data(self, save_to_file=False):
        LOGGER.info("Loading data from tweet_user graph...")
        edge_idx_dict, edge_weight_dict = {}, {}
        edge_type_pairs = list(zip(self.G_edges, ["t_tb_u", "t_rb_u", "u_t_t", "u_r_t", "u_rb_u", "u_rb_u"]))
        for dgl_edge, edge_type in edge_type_pairs:
            if dgl_edge == ("user", "retweeting_uu", "user"):
                mat_coo = self.tweet_user_graph.edgetype2mat[edge_type].transpose().tocoo()
            else:
                mat_coo = self.tweet_user_graph.edgetype2mat[edge_type].tocoo()
            mat_row = mat_coo.row.tolist()
            mat_col = mat_coo.col.tolist()
            mat_val = mat_coo.data.tolist()
            edges = list(zip(mat_row, mat_col))
            edge_idx_dict[dgl_edge] = edges
            edge_weight_dict[dgl_edge] = mat_val

        x_tweets = self.tweet_user_graph.x_tweets
        x_users = self.tweet_user_graph.x_users
        y_tweets = self.tweet_user_graph.y_tweets

        if save_to_file:
            with open(self.data_file, "wb") as outfile:
                pickle.dump([edge_idx_dict, edge_weight_dict, x_tweets, x_users, y_tweets], outfile)

        return edge_idx_dict, edge_weight_dict, x_tweets, x_users, y_tweets

    def create_dgl_heterograph(self, edge_idx_dict, edge_weight_dict, x_tweets, x_users, y_tweets):
        LOGGER.info("Creating G...")
        G = dgl.heterograph(edge_idx_dict)

        # assign weights
        for dgl_edge, val in edge_weight_dict.items():
            G.edges[dgl_edge[1]].data["weight"] = torch.FloatTensor(val)

        # assign feats
        for ntype in G.ntypes:
            if ntype == "tweet":
                assert G.number_of_nodes(ntype) == x_tweets.shape[0]
                emb = nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), x_tweets.shape[1]), requires_grad=False)
                emb.data.copy_(x_tweets)
                G.nodes[ntype].data["label"] = y_tweets
            elif ntype == "user":
                assert G.number_of_nodes(ntype) == x_users.shape[0]
                emb = nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), x_users.shape[1]), requires_grad=False)
                emb.data.copy_(x_users)
            G.nodes[ntype].data["inp"] = emb

        # assign ids
        G.node_dict = {}
        G.edge_dict = {}
        for ntype in G.ntypes:
            G.node_dict[ntype] = len(G.node_dict)
        for etype in G.etypes:
            G.edge_dict[etype] = len(G.edge_dict)
            G.edges[etype].data["id"] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * G.edge_dict[etype]

        return G

    def update_G(self, tweet_user_graph):
        self.tweet_user_graph = tweet_user_graph
        edge_idx_dict, edge_weight_dict, x_tweets, x_users, y_tweets = self.reload_data()
        self.G = self.create_dgl_heterograph(edge_idx_dict, edge_weight_dict, x_tweets, x_users, y_tweets)


class MultiLayerSampler(dgl.dataloading.BlockSampler):
    def __init__(self, fanouts, num_layers):
        super().__init__(num_layers)
        self.fanouts = fanouts

    def sample_frontier(self, block_id, g, seed_nodes):
        fanout = self.fanouts[block_id]
        frontier = dgl.sampling.sample_neighbors(g=g, nodes=seed_nodes, fanout=fanout, prob="weight")
        new_edges_masks = {}
        for etype in frontier.canonical_etypes:
            edge_mask = torch.zeros(g.number_of_edges(etype))
            selected_edges = frontier.edges[etype].data[dgl.EID]
            edge_mask[selected_edges] = 1
            new_edges_masks[etype] = edge_mask.bool()
        return frontier, new_edges_masks

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        edges_masks = {etype: torch.zeros(g.number_of_edges(etype)).bool() for etype in g.canonical_etypes}
        for block_id in range(self.num_layers):
            frontier, new_edge_masks = self.sample_frontier(block_id, g, seed_nodes)

            # update edge_masks
            for etype in new_edge_masks:
                edges_masks[etype] += new_edge_masks[etype]
            # update seed nodes
            block = dgl.to_block(frontier, dst_nodes=seed_nodes, include_dst_in_src=False)
            seed_nodes = {ntype: block.srcnodes[ntype].data[dgl.NID] for ntype in block.srctypes}

        sg = dgl.edge_subgraph(g, edges_masks, preserve_nodes=False)

        sg.node_dict = {}
        sg.edge_dict = {}
        for ntype in sg.ntypes:
            sg.node_dict[ntype] = len(sg.node_dict)
        for etype in sg.etypes:
            sg.edge_dict[etype] = len(sg.edge_dict)
            sg.edges[etype].data["id"] = torch.ones(sg.number_of_edges(etype), dtype=torch.long) * sg.edge_dict[etype]

        return sg

    def __len__(self):
        return self.num_layers


def pos_neg_user_sampler(g, node, sample_num):
    etype = "retweeting_uu"
    etype_reverse = "retweeted_uu"

    pos_nodes = set()
    pos_nodes.update(g.in_edges(node, etype=etype)[0].tolist())
    pos_nodes.update(g.in_edges(node, etype=etype_reverse)[0].tolist())
    pos_nodes.update(g.out_edges(node, etype=etype)[1].tolist())
    pos_nodes.update(g.out_edges(node, etype=etype_reverse)[1].tolist())
    pos_nodes = pos_nodes - {node}

    neg_nodes = set()
    all_nodes = [i for i in range(g.number_of_nodes("user"))]
    for n in all_nodes:
        if n not in pos_nodes and n != node:
            neg_nodes.add(n)

    pos_nodes = list(pos_nodes)
    neg_nodes = list(neg_nodes)

    random.shuffle(pos_nodes)
    random.shuffle(neg_nodes)

    selected_pos_nodes = pos_nodes[:sample_num]
    selected_neg_nodes = neg_nodes[:sample_num]

    return selected_pos_nodes, selected_neg_nodes
