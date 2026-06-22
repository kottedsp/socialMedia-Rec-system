"""
Two-tower retrieval model.

This is the "candidate generation" stage: cheap to run over millions of
items because the user and item embeddings are computed independently
(hence "two towers") and matched with a simple dot product. In production
the item tower's embeddings get pre-computed and stored in an ANN index
(FAISS / ScaNN); at request time only the user tower runs, and the index
returns the top-N nearest items in milliseconds.

Training uses in-batch negatives: for a batch of B (user, positive_item)
pairs, every other item in the batch acts as a negative for every user.
This is the standard scalable alternative to fully evaluating the softmax
over the whole catalog (the "sampled softmax" trick used in the YouTube DNN
paper and most large-scale two-tower systems since).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Tower(nn.Module):
    def __init__(self, n_ids: int, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(n_ids, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(ids)
        x = x + self.mlp(x)  # residual: keeps raw embedding signal, adds capacity
        return F.normalize(x, dim=-1)  # unit-normalize so dot product == cosine similarity


class TwoTowerModel(nn.Module):
    def __init__(self, n_users: int, n_posts: int, embed_dim: int = 64):
        super().__init__()
        self.user_tower = Tower(n_users, embed_dim)
        self.item_tower = Tower(n_posts, embed_dim)

    def forward(self, user_ids: torch.Tensor, post_ids: torch.Tensor):
        return self.user_tower(user_ids), self.item_tower(post_ids)

    @torch.no_grad()
    def all_item_embeddings(self, n_posts: int, device) -> torch.Tensor:
        all_ids = torch.arange(n_posts, device=device)
        return self.item_tower(all_ids)

    @torch.no_grad()
    def recommend(self, user_id: int, item_embeddings: torch.Tensor, top_k: int = 50, device="cpu"):
        user_emb = self.user_tower(torch.tensor([user_id], device=device))  # (1, d)
        scores = (user_emb @ item_embeddings.T).squeeze(0)  # (n_posts,)
        top_scores, top_idx = torch.topk(scores, k=min(top_k, scores.shape[0]))
        return top_idx.cpu().tolist(), top_scores.cpu().tolist()


def in_batch_softmax_loss(user_emb: torch.Tensor, item_emb: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Treat the diagonal of the (B, B) similarity matrix as the positive
    pair, everything off-diagonal as a negative -- in-batch negative
    sampling, the standard trick for training retrieval towers at scale."""
    logits = (user_emb @ item_emb.T) / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, targets)
