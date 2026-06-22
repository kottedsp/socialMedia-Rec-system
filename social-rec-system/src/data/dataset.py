"""
Loading + splitting utilities.

The one rule that matters most here: SPLIT BY TIME, NOT RANDOMLY.
A random split lets the model see interactions from "the future" relative
to other rows in training, which leaks information no real production
system would have and inflates offline metrics. We hold out the last
slice of days as validation/test, mimicking how you'd actually evaluate
a feed ranker (train on the past, predict the next period).
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def load_raw():
    users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    posts = pd.read_csv(os.path.join(DATA_DIR, "posts.csv"))
    edges = pd.read_csv(os.path.join(DATA_DIR, "social_edges.csv"))
    interactions = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))
    return users, posts, edges, interactions


def time_split(interactions: pd.DataFrame, val_frac=0.15, test_frac=0.15):
    """Chronological split. Last `test_frac` of the time window is test,
    the slice before that is val, everything earlier is train."""
    max_day = interactions["day"].max()
    min_day = interactions["day"].min()
    span = max_day - min_day
    test_cutoff = max_day - test_frac * span
    val_cutoff = test_cutoff - val_frac * span

    train = interactions[interactions["day"] <= val_cutoff]
    val = interactions[(interactions["day"] > val_cutoff) & (interactions["day"] <= test_cutoff)]
    test = interactions[interactions["day"] > test_cutoff]
    return train, val, test


def build_label_table(interactions: pd.DataFrame) -> pd.DataFrame:
    """Collapse raw events into one row per (user, post) with multi-task
    binary/continuous labels -- this is what the ranking model trains on."""
    g = interactions.groupby(["user_id", "post_id"])
    labels = g.agg(
        viewed=("event", lambda s: int((s == "view").any())),
        liked=("event", lambda s: int((s == "like").any())),
        commented=("event", lambda s: int((s == "comment").any())),
        shared=("event", lambda s: int((s == "share").any())),
        watch_time_s=("watch_time_s", "max"),
        day=("day", "min"),
    ).reset_index()
    return labels


class RetrievalDataset(Dataset):
    """For the two-tower model: (user, positive_post) pairs. Negatives are
    sampled in-batch at train time (see train_retrieval.py), not stored here
    -- in-batch negative sampling is the standard, scalable approach used by
    YouTube's / most industrial two-tower retrieval models."""

    def __init__(self, label_table: pd.DataFrame, n_users: int, n_posts: int):
        pos = label_table[label_table["viewed"] == 1]
        self.user_ids = pos["user_id"].to_numpy()
        self.post_ids = pos["post_id"].to_numpy()
        self.n_users = n_users
        self.n_posts = n_posts

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.user_ids[idx], dtype=torch.long), torch.tensor(self.post_ids[idx], dtype=torch.long)


class RankingDataset(Dataset):
    """For the multi-task ranking model: every (user, post) pair that was
    at least *viewed* (i.e. shown), with multi-task labels. Modeling only
    viewed pairs mirrors a real system, which only has labels for impressions
    it actually served."""

    def __init__(self, label_table: pd.DataFrame, social_features: pd.DataFrame):
        df = label_table.merge(social_features, on=["user_id", "post_id"], how="left").fillna(0.0)
        self.user_ids = df["user_id"].to_numpy()
        self.post_ids = df["post_id"].to_numpy()
        self.friend_engaged_ratio = df["friend_engaged_ratio"].to_numpy(dtype=np.float32)
        self.author_pagerank = df["author_pagerank"].to_numpy(dtype=np.float32)
        self.liked = df["liked"].to_numpy(dtype=np.float32)
        self.commented = df["commented"].to_numpy(dtype=np.float32)
        self.shared = df["shared"].to_numpy(dtype=np.float32)
        # log1p + scale watch time so the regression head isn't dominated by outliers
        self.watch_time = np.log1p(df["watch_time_s"].to_numpy(dtype=np.float32))

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return {
            "user_id": torch.tensor(self.user_ids[idx], dtype=torch.long),
            "post_id": torch.tensor(self.post_ids[idx], dtype=torch.long),
            "friend_engaged_ratio": torch.tensor(self.friend_engaged_ratio[idx]),
            "author_pagerank": torch.tensor(self.author_pagerank[idx]),
            "liked": torch.tensor(self.liked[idx]),
            "commented": torch.tensor(self.commented[idx]),
            "shared": torch.tensor(self.shared[idx]),
            "watch_time": torch.tensor(self.watch_time[idx]),
        }
