## RTCS-HGT

### Dependencies Installation
Requires `python 3.7`. To install all dependencies, run
> `pip install -r requirements.txt`

### Directory Structure
```
train.py: script to train RTCS-HGT model
train_baselines.py: script to train baseline models (RCNN, BERT)
configs.py: training configs and model hyperparameters
load_data.py: script to load dataloaders
utils.py: util functions
model
│   rtcs_hgt.py: RTCS-HGT model implementation
│   hgt.py: HGT model implementation
│   tweet_user_graph.py: object to maintain tweet-user heterogeneous graph data
│   dgl_graph.py: DGL heterogeneous graph and RT Cascade subgraph sampling
baseline_models
│   rcnn.py: RCNN model implementation
│   bert.py: BERTweet model implementation
processed_data
│   tweets.txt: all tweets used for training/validation/testing
│   users.txt: all users in tweet-user heterogeneous graph
│   tweets_feats.pkl: BERTweet embedding of all tweets, in the same order as tweets.txt
│   users_feats.pkl: user features embedding of all users, in the same order as users.txt
│   labels.pkl: labels of all tweets, in the same order as tweets.txt
│   tid2rtuid_timedelay.pkl: tweet retweet cascade intermediate file
│   tu_graph_data.pkl: output from tweet_user_graph.py
│   G_data.pkl: output from dgl_graph.py
└── partitions:
│   │   train_tweets.csv: train set
│   │   dev_tweets.csv: dev set
│   │   test_teets.csv: test set
model_states: folder to store model states of the best model, will be updated during training
results: folder to store model test classification reports, will be generated during training
logs: folder to store log files, will be generated during training
```

### Train and Test RTCS-HGT
To train RTCS-HGT, run
> `python train.py` 

Training configs, such as `BATCH_SIZE, EPOCHS, INITIAL_LR` and output files, and model hyperparameters can all be specified in `configs.py`. The current hyperparameter settings in `configs.py` are the configurations with best results after hyperparameter searching. `tu_graph_data.pkl` will be updated by setting `RELOAD_TU_GRAPH = True`, and `G_data.pkl` will be updated by setting `RELOAD_DGL_GRAPH = True`. `train_tweets.csv`, `dev_tweets.csv`, `test_tweets.csv` will be updated if `CREATE_PARTITIONS = True` or `KFOLD = True`. When `CREATE_PARTITIONS = True` or `KFOLD = True`, you also need to set `RELOAD_TU_GRAPH = True` and `RELOAD_DGL_GRAPH = True` to reload the whole graph with new train set and dev set.

To test the model, set `TRAIN = False` and `TEST = True` in `configs.py`, and run
> `python train.py` 

### Train and Test Baseline Models
To train a baseline model, such as an RCNN_LSTM model, set `MODEL_NAME = "rcnn_lstm"` in `configs.py`, and run
> `python train_baselines.py` 