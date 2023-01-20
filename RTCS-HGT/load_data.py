import csv
import logging
import math
import os
import pickle
import random

import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from configs import TRAIN_CONFIGS
from utils import init_logger
from utils import program_sleep

"""Init logger"""
if not os.path.exists(TRAIN_CONFIGS.LOGS_ROOT):
    os.makedirs(TRAIN_CONFIGS.LOGS_ROOT)
log_filename = os.path.basename(__file__).split(".")[0]
init_logger(TRAIN_CONFIGS.LOGS_ROOT, log_filename)
LOGGER = logging.getLogger(log_filename)
"""------------------"""


class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, tids, texts, labels):
        self.tids = tids
        self.texts = texts
        self.labels = labels

    def __getitem__(self, idx):
        item = {}
        item["tids"] = self.tids[idx]
        item["texts"] = self.texts[idx]
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def create_partitions(train_split=0.7, dev_split=0.1):
    LOGGER.info(f"Creating labels and partitions...")

    label_file = f"{TRAIN_CONFIGS.DATA_ROOT}/labels.pkl"
    with open(label_file, "rb") as infile:
        labels = pickle.load(infile)

    tid_list = []
    text_list = []
    with open(f"{TRAIN_CONFIGS.DATA_ROOT}/tweets.txt", "r") as infile:
        for line in infile:
            line = line.strip().split("\t")
            tid_list.append(line[0])
            text_list.append(line[1])
    if labels is not None:
        assert labels.shape[0] == len(tid_list)

    # randomly generate train test sets
    total_samples = len(tid_list)
    sample_idx = [i for i in range(total_samples)]
    train_len = math.floor(total_samples * train_split)
    dev_len = math.floor(total_samples * dev_split)
    test_len = total_samples - train_len - dev_len

    random.shuffle(sample_idx)
    train_idx = sample_idx[:train_len]
    dev_idx = sample_idx[train_len : train_len + dev_len]
    test_idx = sample_idx[train_len + dev_len :]
    assert len(train_idx) == train_len
    assert len(dev_idx) == dev_len
    assert len(test_idx) == test_len

    csv_columns = ["tid", "text", "label"]
    partitions = list(
        zip(
            ["train_tweets", "dev_tweets", "test_tweets"],
            [train_idx, dev_idx, test_idx],
        )
    )
    for partition_name, partition_idx in partitions:
        with open(f"{TRAIN_CONFIGS.DATA_ROOT}/partitions/{partition_name}.csv", "w") as outfile:
            csv_writer = csv.DictWriter(outfile, fieldnames=csv_columns)
            csv_writer.writeheader()
            for idx in partition_idx:
                csv_writer.writerow({"tid": tid_list[idx], "text": text_list[idx], "label": labels[idx]})


def load_partitions(batch_size=64):
    def read_batch_tweets(batch_csv):
        tids = []
        texts = []
        labels = []
        with open(batch_csv, "r") as infile:
            csv_reader = csv.DictReader(infile)
            for row in csv_reader:
                tids.append(row["tid"])
                texts.append(row["text"])
                labels.append(int(row["label"]))
        return tids, texts, labels

    train_tids, train_texts, train_labels = read_batch_tweets(f"{TRAIN_CONFIGS.DATA_ROOT}/partitions/train_tweets.csv")
    dev_tids, dev_texts, dev_labels = read_batch_tweets(f"{TRAIN_CONFIGS.DATA_ROOT}/partitions/dev_tweets.csv")
    test_tids, test_texts, test_labels = read_batch_tweets(f"{TRAIN_CONFIGS.DATA_ROOT}/partitions/test_tweets.csv")

    train_dataset = TweetDataset(train_tids, train_texts, train_labels)
    dev_dataset = TweetDataset(dev_tids, dev_texts, dev_labels)
    test_dataset = TweetDataset(test_tids, test_texts, test_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, dev_dataloader, test_dataloader


def k_fold(k=5, train_dev_split=0.86, batch_size=64):
    def write_to_csv(tids, texts, labels, batch_name):
        csv_columns = ["tid", "text", "label"]
        with open(f"{TRAIN_CONFIGS.DATA_ROOT}/partitions/{batch_name}_tweets.csv", "w") as outfile:
            csv_writer = csv.DictWriter(outfile, fieldnames=csv_columns)
            csv_writer.writeheader()
            for idx in range(len(tids)):
                csv_writer.writerow({"tid": tids[idx], "text": texts[idx], "label": labels[idx]})

    label_file = f"{TRAIN_CONFIGS.DATA_ROOT}/labels.pkl"
    with open(label_file, "rb") as infile:
        labels = pickle.load(infile)

    tid_list = []
    text_list = []
    with open(f"{TRAIN_CONFIGS.DATA_ROOT}/tweets.txt", "r") as infile:
        for line in infile:
            line = line.strip().split("\t")
            tid_list.append(line[0])
            text_list.append(line[1])
    if labels is not None:
        assert labels.shape[0] == len(tid_list)

    # k-fold
    kf = KFold(n_splits=k, shuffle=True)
    for train_dev_idx, test_idx in kf.split(tid_list):
        train_len = math.floor(len(train_dev_idx) * train_dev_split)
        random.shuffle(train_dev_idx)

        train_idx = train_dev_idx[:train_len]
        dev_idx = train_dev_idx[train_len:]

        train_tids, train_texts, train_labels = [tid_list[i] for i in train_idx], [text_list[i] for i in train_idx], labels[train_idx]
        dev_tids, dev_texts, dev_labels = [tid_list[i] for i in dev_idx], [text_list[i] for i in dev_idx], labels[dev_idx]
        test_tids, test_texts, test_labels = [tid_list[i] for i in test_idx], [text_list[i] for i in test_idx], labels[test_idx]

        write_to_csv(train_tids, train_texts, train_labels, "train")
        write_to_csv(dev_tids, dev_texts, dev_labels, "dev")
        write_to_csv(test_tids, test_texts, test_labels, "test")
        program_sleep(5)

        train_dataset = TweetDataset(train_tids, train_texts, train_labels)
        dev_dataset = TweetDataset(dev_tids, dev_texts, dev_labels)
        test_dataset = TweetDataset(test_tids, test_texts, test_labels)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        yield train_dataloader, dev_dataloader, test_dataloader
