import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from tqdm import trange
from transformers import AdamW
from transformers import AutoTokenizer

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

from load_data import k_fold, create_partitions, load_partitions
from baseline_models import Bert, RCNNModel

"""Constants"""
TOKENIZER = AutoTokenizer.from_pretrained("vinai/bertweet-covid19-base-cased", normalization=True)
# define output files
MODEL_STATES_FILE = f"{TRAIN_CONFIGS.MODEL_STATES_ROOT}/{TRAIN_CONFIGS.OUTPUT_FILES_NAME}.weights.best"
"""------------------"""


def train(model, optimizer, best_dev_acc, patience, epoch_num, use_validation=True):
    LOGGER.info(f"=======Start training epoch {epoch_num}/{TRAIN_CONFIGS.EPOCHS}!=======")
    start_time = time.time()
    model.train()

    loss_accumulate = 0
    acc_accumulate = 0

    pbar = trange(len(TRAIN_LOADER), desc="Loss: ; Acc: ", leave=True)
    for tweet_items in TRAIN_LOADER:

        batch_texts, batch_labels = tweet_items["texts"], tweet_items["labels"]
        batch_labels = batch_labels.to(TRAIN_CONFIGS.DEVICE)

        encoding = TOKENIZER(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=256)
        input_ids = encoding["input_ids"].to(TRAIN_CONFIGS.DEVICE)
        atten_mask = encoding["attention_mask"].to(TRAIN_CONFIGS.DEVICE)

        outputs_batch, _ = model(input_ids, atten_mask)
        loss_batch = F.nll_loss(outputs_batch, batch_labels)

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
        best_dev_acc, patience = evaluate(model, best_dev_acc, patience)
        model.train()

    return best_dev_acc, patience


def pass_data_iteratively(model, dataloader):
    model.eval()
    outputs = []
    labels = []

    pbar = trange(len(dataloader), leave=True)
    for tweet_items in dataloader:

        batch_texts, batch_labels = tweet_items["texts"], tweet_items["labels"]
        batch_labels = batch_labels.to(TRAIN_CONFIGS.DEVICE)

        encoding = TOKENIZER(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=256)
        input_ids = encoding["input_ids"].to(TRAIN_CONFIGS.DEVICE)
        atten_mask = encoding["attention_mask"].to(TRAIN_CONFIGS.DEVICE)

        outputs_batch, _ = model(input_ids, atten_mask)

        outputs.append(outputs_batch.detach())
        labels.append(batch_labels)
        pbar.update(1)
    pbar.close()

    return torch.cat(outputs, dim=0), torch.cat(labels, dim=0)


def evaluate(model, best_dev_acc, patience):
    LOGGER.info("Evaluating model...")
    outputs, labels = pass_data_iteratively(model, DEV_LOADER)
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


def test(model):
    LOGGER.info("Testing model...")
    outputs, labels = pass_data_iteratively(model, TEST_LOADER)
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


def main(model):

    if TRAIN_CONFIGS.TRAIN:
        LOGGER.info("Training model...")
        optimizer = AdamW(model.parameters(), lr=TRAIN_CONFIGS.INITIAL_LR, weight_decay=TRAIN_CONFIGS.WEIGHT_DECAY)
        best_dev_acc = 0.0
        patience = 0
        start_time = time.time()
        for epoch in range(1, TRAIN_CONFIGS.EPOCHS + 1):
            best_dev_acc, patience = train(model, optimizer, best_dev_acc, patience, epoch)
            if patience >= TRAIN_CONFIGS.PATIENCE:
                LOGGER.info(f"Epoch {epoch} and patience {patience}. Reloading the best model...")
                model.load_state_dict(torch.load(MODEL_STATES_FILE, map_location=TRAIN_CONFIGS.DEVICE))
                adjust_learning_rate(optimizer)
                patience = 0

                best_dev_acc, patience = evaluate(model, best_dev_acc, patience)
        LOGGER.info(f"=======Finished training {TRAIN_CONFIGS.EPOCHS} epochs! Time elapsed: {np.round(time.time() - start_time, 5)}=======")

    if TRAIN_CONFIGS.TEST:
        LOGGER.info("=======Start testing model!=======")
        model.load_state_dict(torch.load(MODEL_STATES_FILE, map_location=TRAIN_CONFIGS.DEVICE))
        test_results_dict = test(model)
        RESULTS_OUTFILE.write("Test Results:\n")
        RESULTS_OUTFILE.write(f"{test_results_dict['class_report']}\n")
        RESULTS_OUTFILE.flush()
        LOGGER.info("=======Finished testing!=======")

    return test_results_dict


if __name__ == "__main__":

    if not TRAIN_CONFIGS.KFOLD:
        RESULTS_OUTPUT_FILE = f"{TRAIN_CONFIGS.RESULTS_ROOT}/{TRAIN_CONFIGS.OUTPUT_FILES_NAME}_results.txt"
        RESULTS_OUTFILE = open(RESULTS_OUTPUT_FILE, "w")
        # create and load partition
        if TRAIN_CONFIGS.CREATE_PARTITIONS:
            create_partitions(train_split=TRAIN_CONFIGS.TRAIN_SPLIT, dev_split=TRAIN_CONFIGS.DEV_SPLIT)
            program_sleep(5)
        TRAIN_LOADER, DEV_LOADER, TEST_LOADER = load_partitions(batch_size=TRAIN_CONFIGS.BATCH_SIZE)

        if TRAIN_CONFIGS.MODEL_NAME == "bert":
            model = Bert().to(TRAIN_CONFIGS.DEVICE)
        elif TRAIN_CONFIGS.MODEL_NAME == "rcnn_lstm":
            model = RCNNModel(rnn_model="LSTM").to(TRAIN_CONFIGS.DEVICE)
        elif TRAIN_CONFIGS.MODEL_NAME == "rcnn_gru":
            model = RCNNModel(rnn_model="GRU").to(TRAIN_CONFIGS.DEVICE)
        else:
            raise Exception(f"Unsupported model type {TRAIN_CONFIGS.MODEL_NAME}!")
        test_results_dict = main(model)
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
            if TRAIN_CONFIGS.MODEL_NAME == "bert":
                model = Bert().to(TRAIN_CONFIGS.DEVICE)
            elif TRAIN_CONFIGS.MODEL_NAME == "rcnn_lstm":
                model = RCNNModel(rnn_model="LSTM").to(TRAIN_CONFIGS.DEVICE)
            elif TRAIN_CONFIGS.MODEL_NAME == "rcnn_gru":
                model = RCNNModel(rnn_model="GRU").to(TRAIN_CONFIGS.DEVICE)
            else:
                raise Exception(f"Unsupported model type {TRAIN_CONFIGS.MODEL_NAME}!")
            test_results_dict = main(model)
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
