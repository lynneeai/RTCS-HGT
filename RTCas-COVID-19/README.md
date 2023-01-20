## RTCas-COVID-19

This folder contains the novel RTCas-COVID-19 corpus we collected. 

- **cleaned\_corpus:** This folder contains the 10M source tweets and their corresponding retweet cascade (25M retweets). There are 12 jsonl files in this folder, named in the format `{year}_{month}.jsonl`. Each line of a jsonl file is a json object of a tweet, with the following fields:

```
{
    "tid": tweet id,
    "tweeter": tweeter's user id,
    "created_at": timestamp of when the tweet was posted,
    "rt_cascade": 
        [
            {
                "tid": retweet's tweet id,
                "uid": retweeter's user id,
                "timedelay": retweet time - tweet post time (in minutes)
            },
            ...
        ]
}
```
- **weak\_labeled\_tweets.csv:** This file contains the 2M tweets weak-labeled as trustworthy (0) or untrustworthy (1). The fields of the csv are `"tid", "label"`.
- **model\_tweets.csv:** This file contains the tweets used to train and test the RTCS-HGT model. The fields of the csv are `"tid", "label"`.
- **human\_annotated\_tweets.csv:** This file contains our manually labeled tweets. The fields of the csv are `"tid", "label"`.
