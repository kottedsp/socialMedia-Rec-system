"""
Serving layer: implements the full two-stage pipeline at request time.

  1. Retrieval: two-tower model returns ~500 candidates via dot product
     over pre-computed item embeddings (a stand-in for an ANN index --
     swap `all_item_embeddings` + brute-force topk for FAISS/ScaNN at
     real scale).
  2. Ranking: multi-task model scores each candidate; scores are blended
     into one ranking score.
  3. Re-ranking: simple diversity rule (cap posts per author) so the feed
     isn't dominated by one prolific creator, plus a recency-decay nudge.

Run: uvicorn src.serve.api:app --reload
Then: curl http://localhost:8000/recommend/42
"""

import os
import sys

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.dataset import DATA_DIR, load_raw
from src.models.ranking_model import RankingModel
from src.models.two_tower import TwoTowerModel

app = FastAPI(title="SocialFeed Recommendation API")

DEVICE = torch.device("cpu")  # serving usually runs on CPU for this model size
RETRIEVAL_CANDIDATES = 500
FINAL_TOP_K = 20
MAX_POSTS_PER_AUTHOR = 2


class RecommendationItem(BaseModel):
    post_id: int
    author_id: int
    score: float


class RecommendationResponse(BaseModel):
    user_id: int
    items: list[RecommendationItem]


# --- Load data + models once at startup ---------------------------------
_users, _posts, _edges, _interactions = load_raw()
_n_users, _n_posts = len(_users), len(_posts)

_social_features_path = os.path.join(DATA_DIR, "social_features.csv")
_social_features = (
    pd.read_csv(_social_features_path) if os.path.exists(_social_features_path) else pd.DataFrame()
)

_retrieval_model = TwoTowerModel(_n_users, _n_posts).to(DEVICE)
_ranking_model = RankingModel(_n_users, _n_posts).to(DEVICE)

_retrieval_ckpt = "checkpoints/two_tower.pt"
_ranking_ckpt = "checkpoints/ranking_model.pt"
if os.path.exists(_retrieval_ckpt):
    _retrieval_model.load_state_dict(torch.load(_retrieval_ckpt, map_location=DEVICE))
if os.path.exists(_ranking_ckpt):
    _ranking_model.load_state_dict(torch.load(_ranking_ckpt, map_location=DEVICE))
_retrieval_model.eval()
_ranking_model.eval()

_item_embeddings = _retrieval_model.all_item_embeddings(_n_posts, DEVICE)
_post_author_lookup = _posts.set_index("post_id")["author_id"].to_dict()


def _get_social_feats(user_id: int, post_id: int) -> tuple[float, float]:
    if _social_features.empty:
        return 0.0, 0.0
    row = _social_features[
        (_social_features["user_id"] == user_id) & (_social_features["post_id"] == post_id)
    ]
    if row.empty:
        return 0.0, 0.0
    return float(row["friend_engaged_ratio"].iloc[0]), float(row["author_pagerank"].iloc[0])


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend(user_id: int):
    if user_id < 0 or user_id >= _n_users:
        raise HTTPException(status_code=404, detail="Unknown user_id")

    # Stage 1: retrieval
    candidate_ids, _ = _retrieval_model.recommend(
        user_id, _item_embeddings, top_k=RETRIEVAL_CANDIDATES, device=DEVICE
    )

    # Stage 2: ranking
    user_tensor = torch.tensor([user_id] * len(candidate_ids))
    post_tensor = torch.tensor(candidate_ids)
    friend_ratios, author_pageranks = zip(*[_get_social_feats(user_id, pid) for pid in candidate_ids])

    with torch.no_grad():
        outputs = _ranking_model(
            user_tensor, post_tensor, torch.tensor(friend_ratios, dtype=torch.float32), torch.tensor(author_pageranks, dtype=torch.float32)
        )
        scores = _ranking_model.blended_score(outputs)

    ranked = sorted(zip(candidate_ids, scores.tolist()), key=lambda x: -x[1])

    # Stage 3: re-rank for diversity (cap posts per author)
    final, author_counts = [], {}
    for post_id, score in ranked:
        author_id = _post_author_lookup.get(post_id, -1)
        if author_counts.get(author_id, 0) >= MAX_POSTS_PER_AUTHOR:
            continue
        author_counts[author_id] = author_counts.get(author_id, 0) + 1
        final.append(RecommendationItem(post_id=int(post_id), author_id=int(author_id), score=float(score)))
        if len(final) >= FINAL_TOP_K:
            break

    return RecommendationResponse(user_id=user_id, items=final)


@app.get("/health")
def health():
    return {"status": "ok", "n_users": _n_users, "n_posts": _n_posts}
