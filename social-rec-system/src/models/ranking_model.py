"""
Multi-task ranking model.

Takes the ~500 candidates returned by the retrieval stage and scores each
one on multiple engagement objectives simultaneously: P(like), P(comment),
P(share), and expected watch time. Real feed rankers blend these into a
single "value score" (e.g. a weighted sum, with weights tuned by the
product/growth team based on what the business wants to optimize for that
quarter) rather than training one objective -- a single-label model can't
distinguish "mindless rage-bait that gets likes" from "content that drives
genuine retention," which is exactly the kind of nuance interviewers like to
probe.
"""

import torch
import torch.nn as nn


class RankingModel(nn.Module):
    def __init__(self, n_users: int, n_posts: int, embed_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.post_embedding = nn.Embedding(n_posts, embed_dim)

        # +2 for the social-graph scalar features (friend_engaged_ratio, author_pagerank)
        input_dim = embed_dim * 2 + 2
        self.shared_trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Separate small heads per task -- lets each objective specialize
        # on top of a shared representation (classic multi-task pattern).
        self.like_head = nn.Linear(hidden_dim // 2, 1)
        self.comment_head = nn.Linear(hidden_dim // 2, 1)
        self.share_head = nn.Linear(hidden_dim // 2, 1)
        self.watch_time_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, user_ids, post_ids, friend_engaged_ratio, author_pagerank):
        u = self.user_embedding(user_ids)
        p = self.post_embedding(post_ids)
        x = torch.cat(
            [u, p, friend_engaged_ratio.unsqueeze(-1), author_pagerank.unsqueeze(-1)], dim=-1
        )
        h = self.shared_trunk(x)
        return {
            "like_logit": self.like_head(h).squeeze(-1),
            "comment_logit": self.comment_head(h).squeeze(-1),
            "share_logit": self.share_head(h).squeeze(-1),
            "watch_time_pred": self.watch_time_head(h).squeeze(-1),
        }

    def blended_score(self, outputs: dict, weights: dict | None = None) -> torch.Tensor:
        """Combine task outputs into one ranking score. The weights are a
        product decision, not an ML one -- exposing them as a parameter
        here is the point: it shows you understand that 'the model' isn't
        the whole system, the objective weighting is a separate lever."""
        weights = weights or {"like": 1.0, "comment": 2.0, "share": 3.0, "watch_time": 0.5}
        like_p = torch.sigmoid(outputs["like_logit"])
        comment_p = torch.sigmoid(outputs["comment_logit"])
        share_p = torch.sigmoid(outputs["share_logit"])
        watch_time = outputs["watch_time_pred"]
        return (
            weights["like"] * like_p
            + weights["comment"] * comment_p
            + weights["share"] * share_p
            + weights["watch_time"] * watch_time
        )


def multitask_loss(outputs: dict, batch: dict) -> torch.Tensor:
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    loss = (
        bce(outputs["like_logit"], batch["liked"])
        + bce(outputs["comment_logit"], batch["commented"])
        + bce(outputs["share_logit"], batch["shared"])
        + 0.1 * mse(outputs["watch_time_pred"], batch["watch_time"])  # downweight regression head
    )
    return loss
