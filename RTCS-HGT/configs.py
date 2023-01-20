import torch


class TRAIN_CONFIGS:

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    LABEL_NAMES = ["trustworthy", "untrustworthy"]
    MODEL_NAME = "rtcshgt"

    # training configs
    INITIAL_LR = 0.001
    WEIGHT_DECAY = 0.001
    EPOCHS = 30
    BATCH_SIZE = 32
    PATIENCE = 3

    # loss configs
    USE_PROX_LOSS = False

    # train dev test split
    KFOLD = False
    CREATE_PARTITIONS = True
    TRAIN_SPLIT = 0.7
    DEV_SPLIT = 0.1
    TRAIN = True
    TEST = True

    # output paths
    DATA_ROOT = "./processed_data"
    LOGS_ROOT = "./logs"
    RESULTS_ROOT = "./results"
    MODEL_STATES_ROOT = "./model_states"
    OUTPUT_FILES_NAME = f"{MODEL_NAME}+proxloss" if USE_PROX_LOSS else MODEL_NAME

    # data file configs
    RELOAD_TU_GRAPH = True
    TU_GRAPH_DATA_FILENAME = "tu_graph_data.pkl"

    RELOAD_DGL_GRAPH = True
    G_DATA_FILENAME = "G_data.pkl"


class MODEL_CONFIGS:

    # dgl sampler configs
    SAMPLER_LAYERS = 2
    SAMPLER_FANOUTS = [
        # hop 1
        {"tweeted": 20, "tweeting": 20, "retweeted_tu": 20, "retweeting_ut": 50, "retweeted_uu": 20, "retweeting_uu": 20},
        # hop 2
        {"tweeted": 10, "tweeting": 10, "retweeted_tu": 10, "retweeting_ut": 25, "retweeted_uu": 10, "retweeting_uu": 10},
    ]

    # trust model configs
    OUTPUT_EMBED_SIZE = 8
    N_CLASS = len(TRAIN_CONFIGS.LABEL_NAMES)

    # HGT Configs
    HGT_N_HID = 32
    HGT_N_OUT = 16
    HGT_N_LAYERS = 2
    HGT_N_HEADS = 2
    HGT_DROPOUT = 0.2
    HGT_USE_NORM = False

    if TRAIN_CONFIGS.USE_PROX_LOSS:
        PROX_LOSS_Q = 10
        PROX_LOSS_PERCENT = 0.2
        SAMPLE_NODES_PORTION = 0.2
        SAMPLE_NODES_PER_NODE = 10
