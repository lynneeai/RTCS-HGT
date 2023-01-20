import csv
import logging
import os
import pickle
import sys
from collections import defaultdict

import scipy.sparse as sp
import torch

"""Solve import issue"""
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_file_dir}/.."
sys.path.append(current_file_dir)
sys.path.append(project_root_dir)
"""------------------"""

from configs import TRAIN_CONFIGS
from utils import init_logger, get_url_domain

"""Init logger"""
if not os.path.exists(TRAIN_CONFIGS.LOGS_ROOT):
    os.makedirs(TRAIN_CONFIGS.LOGS_ROOT)
log_filename = os.path.basename(__file__).split(".")[0]
init_logger(TRAIN_CONFIGS.LOGS_ROOT, log_filename)
LOGGER = logging.getLogger(log_filename)
"""------------------"""


class Tweet_User_Graph(object):
    def __init__(self, data_file, reload_graph=False):
        super(Tweet_User_Graph, self).__init__()

        self.root_data_dir = TRAIN_CONFIGS.DATA_ROOT
        self.tid2rtuid_timedelay = self.load_pkl_obj(f"{self.root_data_dir}/tid2rtuid_timedelay.pkl")

        tweet_bert_feats = self.load_pkl_obj(f"{self.root_data_dir}/tweet_feats.pkl")
        labels = self.load_pkl_obj(f"{self.root_data_dir}/labels.pkl")
        self.tid2bert_feat = {}
        self.tid2label = {}
        self.url2tid_set = defaultdict(set)
        with open(f"{self.root_data_dir}/tweets.txt", "r") as infile:
            idx = 0
            for line in infile:
                tid, _, urls = line.strip().split("\t")
                self.tid2bert_feat[tid] = torch.FloatTensor(tweet_bert_feats[idx])
                self.tid2label[tid] = torch.LongTensor([labels[idx]])
                idx += 1

                urls = urls.split(",")
                for url in urls:
                    if "http" in url:
                        tid_domain = get_url_domain(url, domain_name_only=False)
                        if tid_domain not in ["None", ""]:
                            self.url2tid_set[tid_domain].add(tid)

        user_feats, user_feats_names = self.load_pkl_obj(f"{self.root_data_dir}/user_feats.pkl")
        self.user_feat_size = len(user_feats_names)
        self.uid2feat = {}
        with open(f"{self.root_data_dir}/users.txt", "r") as infile:
            idx = 0
            for line in infile:
                uid = line.strip()
                self.uid2feat[uid] = torch.FloatTensor(user_feats[idx])
                idx += 1

        self.train_csv_file, self.dev_csv_file, self.test_csv_file = "train_tweets.csv", "dev_tweets.csv", "test_tweets.csv"
        self.tweet_user_graph_data_file = f"{self.root_data_dir}/{data_file}"

        self.x_tweets = None
        self.x_users = None
        self.y_tweets = None
        self.tid2idx = {}
        self.uid2idx = {}
        self.edgetype2mat = defaultdict(sp.csr_matrix)

        if reload_graph:
            self.add_train_batch()
            self.add_dev_batch()
        else:
            if os.path.isfile(self.tweet_user_graph_data_file):
                LOGGER.info(f"Loading tweet_user graph from {self.tweet_user_graph_data_file}...")
                tweet_user_graph_data_dict = self.load_pkl_obj(self.tweet_user_graph_data_file)
                for key, val in tweet_user_graph_data_dict.items():
                    setattr(self, key, val)
            else:
                self.add_train_batch()
                self.add_dev_batch()

    def load_pkl_obj(self, pkl_file):
        with open(pkl_file, "rb") as infile:
            obj = pickle.load(infile)
        return obj

    def build_tu_graph(self, tid2uid_rtuid2time, tid2idx, uid2idx):

        edgetype2row_col_val = defaultdict(lambda: defaultdict(list))
        for tid, (tweet_uid, rtuidtime_list) in tid2uid_rtuid2time.items():

            tid_idx = tid2idx[tid]

            if tweet_uid != "":
                puid_idx = uid2idx[tweet_uid]

                # tweet2user edge
                edgetype2row_col_val["t_tb_u"]["row"].append(tid_idx)
                edgetype2row_col_val["t_tb_u"]["col"].append(puid_idx)
                edgetype2row_col_val["t_tb_u"]["val"].append(1)

                # user2tweet edge
                edgetype2row_col_val["u_t_t"]["row"].append(puid_idx)
                edgetype2row_col_val["u_t_t"]["col"].append(tid_idx)
                edgetype2row_col_val["u_t_t"]["val"].append(1)

            for rt_uid, timedelay in rtuidtime_list:
                cuid_idx = uid2idx[rt_uid]

                # tweet2user edge
                edgetype2row_col_val["t_rb_u"]["row"].append(tid_idx)
                edgetype2row_col_val["t_rb_u"]["col"].append(cuid_idx)
                edgetype2row_col_val["t_rb_u"]["val"].append(1 / (timedelay + 1))
                # user2tweet edge
                edgetype2row_col_val["u_r_t"]["row"].append(cuid_idx)
                edgetype2row_col_val["u_r_t"]["col"].append(tid_idx)
                edgetype2row_col_val["u_r_t"]["val"].append(1 / (timedelay + 1))

                if tweet_uid != "":
                    # user2user edge
                    edgetype2row_col_val["u_rb_u"]["row"].append(puid_idx)
                    edgetype2row_col_val["u_rb_u"]["col"].append(cuid_idx)
                    edgetype2row_col_val["u_rb_u"]["val"].append(1)

        # tweet2tweet edge
        for domain, tid_set in self.url2tid_set.items():
            connected_tids = tid_set & set(tid2uid_rtuid2time.keys())
            connected_tids = list(connected_tids)
            if len(connected_tids) > 1:
                for i in range(len(connected_tids)):
                    tidi = connected_tids[i]
                    for j in range(len(connected_tids)):
                        if i != j:
                            tidj = connected_tids[j]
                            tidi_idx, tidj_idx = tid2idx[tidi], tid2idx[tidj]

                            # tweeti2tweetj edge
                            edgetype2row_col_val["t_su_t"]["row"].append(tidi_idx)
                            edgetype2row_col_val["t_su_t"]["col"].append(tidj_idx)
                            edgetype2row_col_val["t_su_t"]["val"].append(1)

        return edgetype2row_col_val

    def add_new_databatch(self, data_csv_file, batch_name, save_to_file=False):
        LOGGER.info(f"Adding new batch {batch_name} to tweet_user graph...")
        new_tid_set = set()
        new_uid_set = set()
        tid2uid_rtuid2time = {}

        with open(f"{self.root_data_dir}/partitions/{data_csv_file}", "r") as infile:
            csv_reader = csv.DictReader(infile)
            for row in csv_reader:
                tid = row["tid"]
                tweet_uid = ""
                rtuidtime_list = []
                for uid, timedelay in self.tid2rtuid_timedelay[tid].items():
                    if timedelay == 0:
                        if uid not in self.uid2idx:
                            new_uid_set.add(uid)
                        tweet_uid = uid
                        continue
                    if uid in self.uid2feat:
                        if uid not in self.uid2idx:
                            new_uid_set.add(uid)
                        rtuidtime_list.append((uid, timedelay))
                tid2uid_rtuid2time[tid] = (tweet_uid, rtuidtime_list)
                if tid not in self.tid2idx:
                    new_tid_set.add(tid)
        new_tid2idx = {tid: i + len(self.tid2idx) for i, tid in enumerate(sorted(list(new_tid_set)))}
        new_uid2idx = {uid: i + len(self.uid2idx) for i, uid in enumerate(sorted(list(new_uid_set)))}

        self.tid2idx.update(new_tid2idx)
        self.uid2idx.update(new_uid2idx)

        new_edgetype2row_col_val = self.build_tu_graph(tid2uid_rtuid2time, self.tid2idx, self.uid2idx)

        x_tweets = [0] * len(self.tid2idx)
        x_users = [0] * len(self.uid2idx)
        y_tweets = [0] * len(self.tid2idx)
        for tid, idx in self.tid2idx.items():
            x_tweets[idx] = self.tid2bert_feat[tid]
            y_tweets[idx] = self.tid2label[tid]
        for uid, idx in self.uid2idx.items():
            x_users[idx] = self.uid2feat[uid]
        self.x_tweets = torch.stack(x_tweets, dim=0)
        self.x_users = torch.stack(x_users, dim=0)
        self.y_tweets = torch.stack(y_tweets, dim=0).squeeze()

        for et, et_dict in new_edgetype2row_col_val.items():
            if et in self.edgetype2mat:
                mat_coo = self.edgetype2mat[et].tocoo()
                mat_row = mat_coo.row.tolist()
                mat_col = mat_coo.col.tolist()
                mat_val = mat_coo.data.tolist()

                mat_row.extend(et_dict["row"])
                mat_col.extend(et_dict["col"])
                mat_val.extend(et_dict["val"])
            else:
                mat_row = et_dict["row"]
                mat_col = et_dict["col"]
                mat_val = et_dict["val"]

            mat = sp.csc_matrix((mat_val, (mat_row, mat_col)))
            self.edgetype2mat[et] = mat

        if save_to_file:
            tweet_user_data_dict = {
                "x_tweets": self.x_tweets,
                "x_users": self.x_users,
                "y_tweets": self.y_tweets,
                "tid2idx": self.tid2idx,
                "uid2idx": self.uid2idx,
                "edgetype2mat": self.edgetype2mat,
            }
            with open(self.tweet_user_graph_data_file, "wb") as outfile:
                pickle.dump(tweet_user_data_dict, outfile)
            LOGGER.info(f"tweet_user graph data {batch_name} saved!")

    def add_train_batch(self):
        self.add_new_databatch(self.train_csv_file, "train", save_to_file=True)

    def add_dev_batch(self):
        self.add_new_databatch(self.dev_csv_file, "dev", save_to_file=True)

    def add_test_batch(self):
        self.add_new_databatch(self.test_csv_file, "test")

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)
