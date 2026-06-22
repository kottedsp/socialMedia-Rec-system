"""
Social-graph feature engineering.

This is the module that makes the project a *social media* recommender
rather than a generic item recommender. Two features are computed:

1. `friend_engaged_ratio(user, post)` — what fraction of the user's direct
   friends already liked/commented/shared this post. This is the single
   strongest "social proof" signal on real platforms (it's a large part of
   why content goes viral within a friend cluster before spreading wider).

2. `author_pagerank` — a graph-centrality score for the post's author.
   High-PageRank authors (well-connected "hub" users) systematically get
   more reach; encoding this lets the ranking model learn an appropriate
   prior instead of being surprised by it.

In a real system these would be computed incrementally / online (e.g. via a
graph database or a streaming feature store) rather than recomputed from
scratch, but the logic is the same.
"""

import os

import networkx as nx
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def build_graph(edges: pd.DataFrame) -> nx.Graph:
    g = nx.Graph()
    g.add_edges_from(edges[["user_id_a", "user_id_b"]].itertuples(index=False, name=None))
    return g


def compute_author_pagerank(graph: nx.Graph, posts: pd.DataFrame) -> pd.DataFrame:
    pr = nx.pagerank(graph, alpha=0.85)
    posts = posts.copy()
    posts["author_pagerank"] = posts["author_id"].map(pr).fillna(0.0)
    return posts[["post_id", "author_pagerank"]]


def compute_friend_engaged_ratio(graph: nx.Graph, interactions: pd.DataFrame) -> pd.DataFrame:
    """For every (user, post) pair that has any interaction, compute the
    fraction of the user's friends who engaged (liked/commented/shared)
    with that same post."""
    engaged_events = interactions[interactions["event"].isin(["like", "comment", "share"])]
    post_to_engagers: dict[int, set[int]] = engaged_events.groupby("post_id")["user_id"].apply(set).to_dict()

    pairs = interactions[["user_id", "post_id"]].drop_duplicates()
    ratios = []
    for uid, pid in pairs.itertuples(index=False, name=None):
        friends = set(graph.neighbors(uid)) if uid in graph else set()
        if not friends:
            ratios.append(0.0)
            continue
        engagers = post_to_engagers.get(pid, set())
        ratios.append(len(friends & engagers) / len(friends))

    pairs = pairs.copy()
    pairs["friend_engaged_ratio"] = ratios
    return pairs


def build_social_features(edges: pd.DataFrame, posts: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    graph = build_graph(edges)
    author_pr = compute_author_pagerank(graph, posts)
    friend_ratio = compute_friend_engaged_ratio(graph, interactions)
    features = friend_ratio.merge(posts[["post_id", "author_id"]], on="post_id").merge(
        author_pr, on="post_id"
    )
    return features[["user_id", "post_id", "friend_engaged_ratio", "author_pagerank"]]


if __name__ == "__main__":
    edges = pd.read_csv(os.path.join(DATA_DIR, "social_edges.csv"))
    posts = pd.read_csv(os.path.join(DATA_DIR, "posts.csv"))
    interactions = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))

    features = build_social_features(edges, posts, interactions)
    out_path = os.path.join(DATA_DIR, "social_features.csv")
    features.to_csv(out_path, index=False)
    print(f"Wrote {len(features)} (user, post) social feature rows to {out_path}")
