"""
Trains the multi-task ranking model on (user, post) pairs that were shown
(viewed), predicting like/comment/share/watch-time jointly.

Run: python src/train_ranking.py
(Run src/data/generate_synthetic_data.py and src/features/graph_features.py first.)
"""

import os
import sys

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src.data.dataset import DATA_DIR, RankingDataset, build_label_table, load_raw, time_split
from src.models.ranking_model import RankingModel, multitask_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5
BATCH_SIZE = 512
LR = 1e-3


def auc_safe(y_true, y_score):
    """ROC-AUC is undefined if a batch/epoch has only one class present;
    guard against that rather than crashing the training loop."""
    if len(set(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def main():
    print("Loading data + social features...")
    users, posts, edges, interactions = load_raw()
    n_users, n_posts = len(users), len(posts)

    social_features_path = os.path.join(DATA_DIR, "social_features.csv")
    if not os.path.exists(social_features_path):
        raise FileNotFoundError(
            "social_features.csv not found -- run `python src/features/graph_features.py` first."
        )
    social_features = pd.read_csv(social_features_path)

    train_int, val_int, test_int = time_split(interactions)
    train_labels = build_label_table(train_int)
    val_labels = build_label_table(val_int)

    train_ds = RankingDataset(train_labels, social_features)
    val_ds = RankingDataset(val_labels, social_features)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = RankingModel(n_users, n_posts).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Training on {len(train_ds)} examples for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(batch["user_id"], batch["post_id"], batch["friend_engaged_ratio"], batch["author_pagerank"])
            loss = multitask_loss(outputs, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation AUC per task
        model.eval()
        all_preds, all_true = {"like": [], "comment": [], "share": []}, {"like": [], "comment": [], "share": []}
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(batch["user_id"], batch["post_id"], batch["friend_engaged_ratio"], batch["author_pagerank"])
                all_preds["like"] += torch.sigmoid(outputs["like_logit"]).cpu().tolist()
                all_true["like"] += batch["liked"].cpu().tolist()
                all_preds["comment"] += torch.sigmoid(outputs["comment_logit"]).cpu().tolist()
                all_true["comment"] += batch["commented"].cpu().tolist()
                all_preds["share"] += torch.sigmoid(outputs["share_logit"]).cpu().tolist()
                all_true["share"] += batch["shared"].cpu().tolist()
        model.train()

        like_auc = auc_safe(all_true["like"], all_preds["like"])
        comment_auc = auc_safe(all_true["comment"], all_preds["comment"])
        share_auc = auc_safe(all_true["share"], all_preds["share"])

        avg_loss = total_loss / len(train_loader)
        print(
            f"Epoch {epoch}/{EPOCHS} | train_loss={avg_loss:.4f} | "
            f"val_AUC like={like_auc:.3f} comment={comment_auc:.3f} share={share_auc:.3f}"
        )

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/ranking_model.pt")
    print("Saved checkpoints/ranking_model.pt")


if __name__ == "__main__":
    main()
