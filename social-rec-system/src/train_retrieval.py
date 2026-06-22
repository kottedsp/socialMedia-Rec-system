"""
Trains the two-tower retrieval model and reports Recall@K / NDCG@K on the
held-out (time-split) validation set.

Run: python src/train_retrieval.py
"""

import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import RetrievalDataset, build_label_table, load_raw, time_split
from src.eval.metrics import ndcg_at_k, recall_at_k
from src.models.two_tower import TwoTowerModel, in_batch_softmax_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5
BATCH_SIZE = 256
LR = 1e-3
TOP_K = 50


def evaluate(model, n_posts, val_labels, k=TOP_K):
    model.eval()
    item_embeddings = model.all_item_embeddings(n_posts, DEVICE)

    relevant_by_user = val_labels[val_labels["viewed"] == 1].groupby("user_id")["post_id"].apply(set).to_dict()

    recalls, ndcgs = [], []
    for user_id, relevant in relevant_by_user.items():
        recommended, _ = model.recommend(user_id, item_embeddings, top_k=k, device=DEVICE)
        recalls.append(recall_at_k(recommended, relevant, k))
        ndcgs.append(ndcg_at_k(recommended, relevant, k))

    model.train()
    return sum(recalls) / len(recalls), sum(ndcgs) / len(ndcgs)


def main():
    print("Loading data...")
    users, posts, edges, interactions = load_raw()
    n_users, n_posts = len(users), len(posts)

    train_int, val_int, test_int = time_split(interactions)
    train_labels = build_label_table(train_int)
    val_labels = build_label_table(val_int)

    train_ds = RetrievalDataset(train_labels, n_users, n_posts)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = TwoTowerModel(n_users, n_posts).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Training on {len(train_ds)} (user, viewed-post) pairs for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for user_ids, post_ids in train_loader:
            user_ids, post_ids = user_ids.to(DEVICE), post_ids.to(DEVICE)
            user_emb, item_emb = model(user_ids, post_ids)
            loss = in_batch_softmax_loss(user_emb, item_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        recall, ndcg = evaluate(model, n_posts, val_labels)
        print(f"Epoch {epoch}/{EPOCHS} | train_loss={avg_loss:.4f} | val_recall@{TOP_K}={recall:.4f} | val_ndcg@{TOP_K}={ndcg:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/two_tower.pt")
    print("Saved checkpoints/two_tower.pt")


if __name__ == "__main__":
    main()
