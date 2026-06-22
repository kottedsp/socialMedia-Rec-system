"""
Generates a synthetic but structurally realistic social-media dataset.

Why synthetic, and why not just random?
- A pure `np.random` ratings matrix has no exploitable structure, so any
  model "works" and you learn nothing about whether your architecture is
  doing its job.
- This generator instead builds:
    1. A scale-free social graph (Barabási–Albert) — real social networks
       have a few high-degree "hub" users, not uniform degree.
    2. Users with topic affinities, posts with topics, so content-based
       signal exists.
    3. Social influence: a user is more likely to engage with a post if
       their friends already engaged with it (this is the signal your
       graph features and your model should pick up on).
    4. A funnel of engagement types (view -> like -> comment/share), each
       rarer than the last, mimicking real engagement-type distributions.
    5. Timestamps, so a chronological train/val/test split is meaningful.

Swap this file out for a real-data loader once you've validated the
pipeline; keep the same four output files (see bottom of this file) and
nothing downstream needs to change.
"""

import os
import random

import networkx as nx
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

N_USERS = 2_000
N_POSTS = 6_000
N_TOPICS = 12
N_DAYS = 30
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def build_social_graph(n_users: int) -> nx.Graph:
    """Scale-free graph: a few users have many connections (influencers),
    most have few -- matches real social network degree distributions far
    better than a uniform random graph."""
    g = nx.barabasi_albert_graph(n=n_users, m=3, seed=42)
    return g


def build_users(n_users: int, n_topics: int) -> pd.DataFrame:
    rows = []
    for uid in range(n_users):
        # Dirichlet gives each user a "soft" topic affinity distribution
        # instead of a single hard-coded interest -- more realistic.
        affinity = np.random.dirichlet(alpha=np.ones(n_topics) * 0.5)
        rows.append({"user_id": uid, **{f"topic_affinity_{t}": affinity[t] for t in range(n_topics)}})
    return pd.DataFrame(rows)


def build_posts(n_posts: int, n_users: int, n_topics: int, n_days: int) -> pd.DataFrame:
    rows = []
    for pid in range(n_posts):
        rows.append(
            {
                "post_id": pid,
                "author_id": random.randint(0, n_users - 1),
                "topic": random.randint(0, n_topics - 1),
                # posts arrive over the window, not all at once
                "created_at_day": round(random.uniform(0, n_days), 3),
            }
        )
    return pd.DataFrame(rows)


def simulate_interactions(users_df, posts_df, graph: nx.Graph) -> pd.DataFrame:
    affinity_cols = [c for c in users_df.columns if c.startswith("topic_affinity_")]
    user_affinity = users_df[affinity_cols].values  # (n_users, n_topics)

    # Track who engaged with what, so friends can influence each other.
    post_engagers: dict[int, set[int]] = {pid: set() for pid in posts_df["post_id"]}

    interactions = []
    posts_sorted = posts_df.sort_values("created_at_day")

    for _, post in posts_sorted.iterrows():
        # iterrows() upcasts mixed-dtype rows to a common dtype (often
        # float64), so cast back to int explicitly before using as indices.
        pid, topic, created_day = int(post["post_id"]), int(post["topic"]), float(post["created_at_day"])

        # Each post is "shown" (impression) to a random sample of users,
        # weighted toward the author's friends -- mimics a follow-graph feed.
        author = int(post["author_id"])
        neighbors = list(graph.neighbors(author)) if author in graph else []
        candidate_viewers = set(neighbors)
        # plus some random discovery traffic (explore/recommend-from-strangers)
        candidate_viewers.update(np.random.choice(len(users_df), size=30, replace=False))

        for uid in candidate_viewers:
            base_p_like = user_affinity[uid, topic]  # content-based propensity

            # Social proof: boost if friends already engaged with this post.
            user_friends = set(graph.neighbors(uid)) if uid in graph else set()
            friends_engaged = len(user_friends & post_engagers[pid])
            social_boost = min(0.35, 0.08 * friends_engaged)

            p_view = min(0.98, 0.4 + base_p_like + social_boost)
            if np.random.rand() > p_view:
                continue  # not even viewed

            view_time = created_day + np.random.exponential(0.3)
            watch_time_s = float(np.clip(np.random.normal(8 + 25 * base_p_like, 5), 1, 60))
            interactions.append(
                {"user_id": uid, "post_id": pid, "event": "view", "day": view_time, "watch_time_s": watch_time_s}
            )

            p_like = min(0.9, base_p_like * 0.8 + social_boost)
            if np.random.rand() < p_like:
                interactions.append(
                    {"user_id": uid, "post_id": pid, "event": "like", "day": view_time, "watch_time_s": watch_time_s}
                )
                post_engagers[pid].add(uid)

                if np.random.rand() < 0.25:
                    interactions.append(
                        {"user_id": uid, "post_id": pid, "event": "comment", "day": view_time, "watch_time_s": watch_time_s}
                    )
                if np.random.rand() < 0.12:
                    interactions.append(
                        {"user_id": uid, "post_id": pid, "event": "share", "day": view_time, "watch_time_s": watch_time_s}
                    )

    return pd.DataFrame(interactions)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Building social graph...")
    graph = build_social_graph(N_USERS)
    edges = pd.DataFrame(list(graph.edges()), columns=["user_id_a", "user_id_b"])

    print("Building users with topic affinities...")
    users_df = build_users(N_USERS, N_TOPICS)

    print("Building posts...")
    posts_df = build_posts(N_POSTS, N_USERS, N_TOPICS, N_DAYS)

    print("Simulating interactions (this is the slow step)...")
    interactions_df = simulate_interactions(users_df, posts_df, graph)

    users_df.to_csv(os.path.join(OUT_DIR, "users.csv"), index=False)
    posts_df.to_csv(os.path.join(OUT_DIR, "posts.csv"), index=False)
    edges.to_csv(os.path.join(OUT_DIR, "social_edges.csv"), index=False)
    interactions_df.to_csv(os.path.join(OUT_DIR, "interactions.csv"), index=False)

    print(f"Users: {len(users_df)}, Posts: {len(posts_df)}, Social edges: {len(edges)}")
    print(f"Interactions: {len(interactions_df)} ({interactions_df['event'].value_counts().to_dict()})")
    print(f"Wrote data to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
