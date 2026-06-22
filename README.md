
# SocialFeed — A Two-Stage Social Media Recommendation System

A portfolio-grade recommendation system that mirrors how real social platforms
(Instagram, TikTok, Twitter/X, Pinterest) rank feed content.


## Architecture

```
                 ┌─────────────────────────┐
                 │   Raw interactions       │
                 │ (views/likes/comments/   │
                 │  shares + social graph)  │
                 └────────────┬─────────────┘
                              │
                 ┌────────────▼─────────────┐
                 │   Feature engineering     │
                 │ user/item embeddings seed │
                 │ + social-graph features   │
                 │ (friend overlap, PageRank)│
                 └────────────┬─────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                            │
┌───────▼────────┐                          ┌────────▼────────┐
│ CANDIDATE GEN    │   top ~500 candidates   │     RANKING      │
│ Two-Tower model   │ ───────────────────▶  │  Multi-task MLP   │
│ (user/item embed,│   (ANN / dot product)  │ like / comment /  │
│  in-batch negs)  │                        │ share / watch-time│
└───────┬────────┘                          └────────┬────────┘
        │                                            │
        └─────────────────────┬──────────────────────┘
                              │
                 ┌────────────▼─────────────┐
                 │  Re-ranking / business    │
                 │  rules: diversity, recency│
                 │  decay, dedup by author   │
                 └────────────┬─────────────┘
                              │
                 ┌────────────▼─────────────┐
                 │   FastAPI /recommend      │
                 │   endpoint → top-K feed   │
                 └───────────────────────────┘
```

## Repo layout

```
social-rec-system/
├── data/                        # generated/raw data lands here (gitignored)
├── src/
│   ├── data/
│   │   ├── generate_synthetic_data.py   # creates a realistic toy dataset
│   │   └── dataset.py                   # loading, time-based split, torch Datasets
│   ├── features/
│   │   └── graph_features.py            # social-graph feature engineering
│   ├── models/
│   │   ├── two_tower.py                 # retrieval model
│   │   └── ranking_model.py             # multi-task ranking model
│   ├── eval/
│   │   └── metrics.py                   # Recall@K, NDCG@K, MAP@K, diversity
│   ├── serve/
│   │   └── api.py                       # FastAPI serving layer
│   ├── train_retrieval.py
│   └── train_ranking.py
├── tests/
│   └── test_metrics.py
├── docs/
│   └── ARCHITECTURE.md          # deep-dive + interview talking points
├── requirements.txt
└── README.md
```

