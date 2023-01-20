import logging
import os
import time

import dgl
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from tqdm import trange
from transformers import AdamW

from configs import MODEL_CONFIGS
from configs import TRAIN_CONFIGS
from utils import init_logger
from utils import program_sleep

"""Make directories"""
if not os.path.exists(TRAIN_CONFIGS.LOGS_ROOT):
    os.makedirs(TRAIN_CONFIGS.LOGS_ROOT)
if not os.path.exists(TRAIN_CONFIGS.RESULTS_ROOT):
    os.makedirs(TRAIN_CONFIGS.RESULTS_ROOT)
if not os.path.exists(TRAIN_CONFIGS.MODEL_STATES_ROOT):
    os.makedirs(TRAIN_CONFIGS.MODEL_STATES_ROOT)
if not os.path.exists(f"{TRAIN_CONFIGS.DATA_ROOT}/partitions"):
    os.makedirs(f"{TRAIN_CONFIGS.DATA_ROOT}/partitions")
"""------------------"""

"""Init logger"""
log_filename = os.path.basename(__file__).split(".")[0]
init_logger(TRAIN_CONFIGS.LOGS_ROOT, TRAIN_CONFIGS.OUTPUT_FILES_NAME)
LOGGER = logging.getLogger(log_filename)
"""------------------"""

from load_data import create_partitions, load_partitions, k_fold
from model import Tweet_User_Graph, DGL_Heterograph, MultiLayerSampler, RTCS_HGT

"""Constants"""
# create dgl sampler
G_SAMPLER = MultiLayerSampler(fanouts=MODEL_CONFIGS.SAMPLER_FANOUTS, num_layers=MODEL_CONFIGS.SAMPLER_LAYERS)
# define output files
MODEL_STATES_FILE = f"{TRAIN_CONFIGS.MODEL_STATES_ROOT}/{TRAIN_CONFIGS.OUTPUT_FILES_NAME}.weights.best"
"""------------------"""


def train(model, optimizer, best_dev_acc, patience, epoch_num, DGL_G, tu_graph, use_validation=True):
    LOGGER.info(f"=======Start training epoch {epoch_num}/{TRAIN_CONFIGS.EPOCHS}!=======")
    start_time = time.time()
    model.train()

    loss_accumulate = 0
    acc_accumulate = 0

    pbar = trange(len(TRAIN_LOADER), desc="Loss: ; Acc: ", leave=True)
    for tweet_items in TRAIN_LOADER:

        batch_tids, batch_labels = tweet_items["tids"], tweet_items["labels"]
        batch_labels = batch_labels.to(TRAIN_CONFIGS.DEVICE)

        batch_nodes = [tu_graph.tid2idx[tid] for tid in batch_tids]
        sg = G_SAMPLER.sample_blocks(DGL_G.G, {"tweet": torch.LongTensor(batch_nodes)})
        sampled_nodes = sg.nodes["tweet"].data[dgl.NID].tolist()

        outputs_batch, user_embed = model(sg=sg.to(TRAIN_CONFIGS.DEVICE), sg_nodes=sampled_nodes, output_nodes=batch_nodes)
        if TRAIN_CONFIGS.USE_PROX_LOSS:
            loss_batch = model.loss(outputs_batch, batch_labels, prox_loss=TRAIN_CONFIGS.USE_PROX_LOSS, sg=sg, user_embed=user_embed, sample_nodes_portion=MODEL_CONFIGS.SAMPLE_NODES_PORTION, sample_nodes_per_node=MODEL_CONFIGS.SAMPLE_NODES_PER_NODE)
        else:
            loss_batch = model.loss(outputs_batch, batch_labels)

        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()

        loss_batch_np = loss_batch.detach().cpu().numpy()
        loss_accumulate += loss_batch_np

        corrects_batch = (torch.max(outputs_batch, 1)[1].view(batch_labels.shape[0]).data == batch_labels.data).sum()
        acc_batch = corrects_batch.detach().cpu().numpy() / float(batch_labels.shape[0])
        acc_accumulate += acc_batch

        pbar.set_description(f"Loss: {loss_batch_np} ; Acc: {acc_batch}")
        pbar.refresh()
        pbar.update(1)

    pbar.close()
    epoch_loss = loss_accumulate / len(TRAIN_LOADER)
    epoch_acc = acc_accumulate / len(TRAIN_LOADER)
    LOGGER.info(f"Finished epoch {epoch_num}/{TRAIN_CONFIGS.EPOCHS}! Epoch Loss: {np.round(epoch_loss, 5)} ; Epoch Acc: {np.round(epoch_acc, 5)}; Time elapsed: {np.round(time.time() - start_time, 5)}")

    if use_validation:
        best_dev_acc, patience = evaluate(model, best_dev_acc, patience, DGL_G, tu_graph)
        model.train()

    return best_dev_acc, patience


def pass_data_iteratively(model, dataloader, DGL_G, tu_graph):
    model.eval()
    outputs = []
    labels = []

    pbar = trange(len(dataloader), leave=True)
    for tweet_items in dataloader:

        batch_tids, batch_labels = tweet_items["tids"], tweet_items["labels"]
        batch_labels = batch_labels.to(TRAIN_CONFIGS.DEVICE)

        batch_nodes = [tu_graph.tid2idx[tid] for tid in batch_tids]
        sg = G_SAMPLER.sample_blocks(DGL_G.G, {"tweet": torch.LongTensor(batch_nodes)})
        sampled_nodes = sg.nodes["tweet"].data[dgl.NID].tolist()

        outputs_batch, _ = model(sg=sg.to(TRAIN_CONFIGS.DEVICE), sg_nodes=sampled_nodes, output_nodes=batch_nodes)

        outputs.append(outputs_batch.detach())
        labels.append(batch_labels)
        pbar.update(1)
    pbar.close()

    return torch.cat(outputs, dim=0), torch.cat(labels, dim=0)


def evaluate(model, best_dev_acc, patience, DGL_G, tu_graph):
    LOGGER.info("Evaluating model...")
    outputs, labels = pass_data_iteratively(model, DEV_LOADER, DGL_G, tu_graph)
    predicted = torch.max(outputs, dim=1)[1]
    dev_y_pred = predicted.data.cpu().numpy().tolist()
    dev_labels = labels.data.cpu().numpy().tolist()
    dev_acc = accuracy_score(dev_labels, dev_y_pred)

    LOGGER.info(f"Current dev set acc: {np.round(dev_acc, 5)}. Previous best dev set acc: {np.round(best_dev_acc, 5)}")
    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        torch.save(
            model.state_dict(),
            MODEL_STATES_FILE,
        )
        LOGGER.info(classification_report(dev_labels, dev_y_pred, target_names=TRAIN_CONFIGS.LABEL_NAMES, digits=5))
        LOGGER.info("Best model saved!")
    else:
        patience += 1

    return best_dev_acc, patience


def test(model, DGL_G, tu_graph):
    LOGGER.info("Testing model...")
    outputs, labels = pass_data_iteratively(model, TEST_LOADER, DGL_G, tu_graph)
    predicted = torch.max(outputs, dim=1)[1]
    test_y_pred = predicted.data.cpu().numpy().tolist()
    test_labels = labels.data.cpu().numpy().tolist()
    LOGGER.info("=====================================")
    class_report = classification_report(test_labels, test_y_pred, target_names=TRAIN_CONFIGS.LABEL_NAMES, digits=5)
    LOGGER.info(class_report)

    cr = classification_report(test_labels, test_y_pred, target_names=TRAIN_CONFIGS.LABEL_NAMES, digits=5, output_dict=True)

    test_results_dict = {
        "outputs": outputs,
        "pred_labels": test_y_pred,
        "true_labels": test_labels,
        "class_report": class_report,
        "accuracy": cr["accuracy"],
        "macro_f1": cr["macro avg"]["f1-score"],
        "trustworthy_f1": cr["trustworthy"]["f1-score"],
        "untrustworthy_f1": cr["untrustworthy"]["f1-score"],
    }
    return test_results_dict


def adjust_learning_rate(optimizer, decay_rate=0.5):
    now_lr = 0
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * decay_rate
        now_lr = param_group["lr"]
    LOGGER.info(f"Adjusted learning rate. Current lr: {now_lr}")


def main(model, DGL_G, tu_graph):

    if TRAIN_CONFIGS.TRAIN:
        LOGGER.info("Training model...")
        optimizer = AdamW(model.parameters(), lr=TRAIN_CONFIGS.INITIAL_LR, weight_decay=TRAIN_CONFIGS.WEIGHT_DECAY)
        best_dev_acc = 0.0
        patience = 0
        start_time = time.time()
        for epoch in range(1, TRAIN_CONFIGS.EPOCHS + 1):
            best_dev_acc, patience = train(model, optimizer, best_dev_acc, patience, epoch, DGL_G, tu_graph)
            if patience >= TRAIN_CONFIGS.PATIENCE:
                LOGGER.info(f"Epoch {epoch} and patience {patience}. Reloading the best model...")
                model.load_state_dict(torch.load(MODEL_STATES_FILE, map_location=TRAIN_CONFIGS.DEVICE))
                adjust_learning_rate(optimizer)
                patience = 0

                best_dev_acc, patience = evaluate(model, best_dev_acc, patience, DGL_G, tu_graph)
        LOGGER.info(f"=======Finished training {TRAIN_CONFIGS.EPOCHS} epochs! Time elapsed: {np.round(time.time() - start_time, 5)}=======")

    if TRAIN_CONFIGS.TEST:
        LOGGER.info("=======Start testing model!=======")
        model.load_state_dict(torch.load(MODEL_STATES_FILE, map_location=TRAIN_CONFIGS.DEVICE))
        tu_graph.add_test_batch()
        DGL_G.update_G(tu_graph)
        test_results_dict = test(model, DGL_G, tu_graph)
        RESULTS_OUTFILE.write("Test Results:\n")
        RESULTS_OUTFILE.write(f"{test_results_dict['class_report']}\n")
        RESULTS_OUTFILE.flush()
        LOGGER.info("=======Finished testing!=======")

    return test_results_dict


if __name__ == "__main__":
    # # grid search
    # n_heads_list = [4, 2]
    # shapes_configs_list = [(64, 32, 16), (32, 16, 8), (16, 8, 4)]
    # lr_list = [0.0001, 0.0005, 0.001, 0.005]
    # wd_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

    if not TRAIN_CONFIGS.KFOLD:
        RESULTS_OUTPUT_FILE = f"{TRAIN_CONFIGS.RESULTS_ROOT}/{TRAIN_CONFIGS.OUTPUT_FILES_NAME}_results.txt"
        RESULTS_OUTFILE = open(RESULTS_OUTPUT_FILE, "w")
        # create and load partition
        if TRAIN_CONFIGS.CREATE_PARTITIONS:
            create_partitions(train_split=TRAIN_CONFIGS.TRAIN_SPLIT, dev_split=TRAIN_CONFIGS.DEV_SPLIT)
            program_sleep(5)
        TRAIN_LOADER, DEV_LOADER, TEST_LOADER = load_partitions(batch_size=TRAIN_CONFIGS.BATCH_SIZE)
        # create tweet_user_graph
        tu_graph = Tweet_User_Graph(data_file=TRAIN_CONFIGS.TU_GRAPH_DATA_FILENAME, reload_graph=TRAIN_CONFIGS.RELOAD_TU_GRAPH)
        # create dgl heterograph
        DGL_G = DGL_Heterograph(tweet_user_graph=tu_graph, data_file=TRAIN_CONFIGS.G_DATA_FILENAME, reload_dgl_graph=TRAIN_CONFIGS.RELOAD_DGL_GRAPH)
        LOGGER.info(DGL_G.G)

        model = RTCS_HGT(G=DGL_G.G, hgt_n_inp_dict={"tweet": tu_graph.x_tweets.shape[1], "user": tu_graph.x_users.shape[1]}).to(TRAIN_CONFIGS.DEVICE)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        LOGGER.info(f"Total parameters: {total_params}")
        LOGGER.info(f"Trainable parameters: {trainable_params}")

        test_results_dict = main(model, DGL_G, tu_graph)
        RESULTS_OUTFILE.close()

    else:
        RESULTS_OUTPUT_FILE = f"{TRAIN_CONFIGS.RESULTS_ROOT}/{TRAIN_CONFIGS.OUTPUT_FILES_NAME}_kfold_results.txt"
        RESULTS_OUTFILE = open(RESULTS_OUTPUT_FILE, "w")

        LOGGER.info(f"KFold {TRAIN_CONFIGS.OUTPUT_FILES_NAME.upper()} Model...")
        RESULTS_OUTFILE.write(f"KFold {TRAIN_CONFIGS.OUTPUT_FILES_NAME.upper()} Model...\n")
        RESULTS_OUTFILE.flush()

        test_acc_list = []
        macro_f1_list = []
        trust_f1_list = []
        untrust_f1_list = []
        for TRAIN_LOADER, DEV_LOADER, TEST_LOADER in k_fold(batch_size=TRAIN_CONFIGS.BATCH_SIZE):
            # create tweet_user_graph
            tu_graph = Tweet_User_Graph(data_file=TRAIN_CONFIGS.TU_GRAPH_DATA_FILENAME, reload_graph=True)
            # create dgl heterograph
            DGL_G = DGL_Heterograph(tweet_user_graph=tu_graph, data_file=TRAIN_CONFIGS.G_DATA_FILENAME, reload_dgl_graph=True)
            LOGGER.info(DGL_G.G)

            # initialize model
            model = RTCS_HGT(G=DGL_G.G, hgt_n_inp_dict={"tweet": tu_graph.x_tweets.shape[1], "user": tu_graph.x_users.shape[1]}).to(TRAIN_CONFIGS.DEVICE)

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            LOGGER.info(f"Total parameters: {total_params}")
            LOGGER.info(f"Trainable parameters: {trainable_params}")

            test_results_dict = main(model, DGL_G, tu_graph)
            test_acc_list.append(test_results_dict["accuracy"])
            macro_f1_list.append(test_results_dict["macro_f1"])
            trust_f1_list.append(test_results_dict["trustworthy_f1"])
            untrust_f1_list.append(test_results_dict["untrustworthy_f1"])

        assert len(test_acc_list) == 5
        avg_test_acc = np.mean(test_acc_list)
        avg_macro_f1 = np.mean(macro_f1_list)
        avg_trust_f1 = np.mean(trust_f1_list)
        avg_untrust_f1 = np.mean(untrust_f1_list)

        LOGGER.info(f"==============KFold {TRAIN_CONFIGS.OUTPUT_FILES_NAME.upper()} Model Results==============")
        LOGGER.info(f"Average test acc: {avg_test_acc}")
        LOGGER.info(f"Average macro f1: {avg_macro_f1}")
        LOGGER.info(f"Average trust f1: {avg_trust_f1}")
        LOGGER.info(f"Average untrust f1: {avg_untrust_f1}")

        RESULTS_OUTFILE.write(f"==============KFold {TRAIN_CONFIGS.OUTPUT_FILES_NAME.upper()} Model Results==============\n")
        RESULTS_OUTFILE.write(f"Average test acc: {avg_test_acc}\n")
        RESULTS_OUTFILE.write(f"Average macro f1: {avg_macro_f1}\n")
        RESULTS_OUTFILE.write(f"Average trust f1: {avg_trust_f1}\n")
        RESULTS_OUTFILE.write(f"Average untrust f1: {avg_untrust_f1}\n")
        RESULTS_OUTFILE.close()
