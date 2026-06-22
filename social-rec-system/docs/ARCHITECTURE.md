# Architecture Deep-Dive & Interview Talking Points

## 1. Why two stages instead of one model?

A single model that scores every post for every user doesn't scale: with
millions of posts and users, you can't run a heavy ranking model over the
full catalog per request within a latency budget (typically <100-200ms for
a feed load). Splitting into **retrieval** (cheap, approximate, casts a wide
net) and **ranking** (expensive, precise, runs on a small candidate set)
is how every large-scale system — YouTube, Pinterest, TikTok — actually
works. It's a latency/accuracy trade-off made explicit as an architecture
decision rather than hidden inside one model.

**If asked "why not just rank everything?"** — answer with the latency
budget math: ranking 500 candidates with a multi-task MLP is fast; ranking
6 million posts with the same model is not, and retrieval narrows the
search space cheaply via pre-computed embeddings + ANN search.

## 2. Why in-batch negative sampling for the two-tower model?

Computing a full softmax over the entire item catalog per training step is
intractable at scale. In-batch negatives reuse the other positives already
in the mini-batch as negatives "for free," which is biased toward popular
items (they show up as negatives more often, proportional to their
frequency) but is a known, accepted trade-off, and is exactly what's
described in the YouTube DNN retrieval paper. You can mention the bias and
that production systems often correct for it with **logQ correction**
(downweighting negatives by their sampling probability).

## 3. Why social graph features specifically?

This is the feature that differentiates "social media recommendation" from
generic item recommendation. Two are implemented:

- **`friend_engaged_ratio`** — social proof. Content that's already
  resonating inside a user's own social cluster is disproportionately
  likely to resonate with them too (homophily). This is also the feature
  most directly responsible for virality dynamics and filter bubbles —
  good material for a "how would you detect/mitigate a filter bubble"
  follow-up question (see §6).
- **`author_pagerank`** — graph centrality as a prior for reach. Well-
  connected authors get a baseline reach advantage; encoding it explicitly
  lets the model learn a calibrated adjustment instead of just overfitting
  popular authors implicitly through embeddings.

## 4. Why multi-task ranking instead of one engagement label?

Optimizing for a single proxy metric (e.g., raw click/like rate) is a
classic recommender-system failure mode: it rewards outrage-bait and
clickbait because those generate likes/clicks cheaply, even when they hurt
long-term retention. Predicting like/comment/share/watch-time separately,
then **blending** them with explicit, inspectable weights, makes the
trade-off a visible lever (`RankingModel.blended_score`) instead of an
implicit one buried in a single loss function. This is also a good prompt
for talking about **Goodhart's law** in ML systems.

## 5. Why time-based splits, not random splits?

Random splits let the model "see the future" relative to other training
rows (e.g., a post's later virality leaking into earlier predictions about
it), which inflates offline metrics in a way that does not transfer to
production, where you only ever have the past to predict the future. A
chronological split is the only honest way to estimate how a model already
trained on history will perform on tomorrow's traffic.

## 6. Likely follow-up questions and how to think about them

**"How would you handle cold-start users/posts?"**
New users/posts have no interaction history, so the embeddings in the
two-tower model are poorly learned for them. Mitigations: fall back to
content-based features (topic, text/image embeddings) and graph features
(which exist immediately, since the social graph is known at signup) until
enough interaction history accumulates; consider a short exploration phase
that intentionally shows new content to a sample of users to bootstrap
signal.

**"How would you detect or reduce filter-bubble effects?"**
Track `catalog_coverage` (what fraction of the catalog ever gets shown) and
`intra_list_diversity` (how similar a user's recommended set is to itself)
over time, per user-segment. A shrinking coverage or diversity trend over
weeks is the leading indicator. Mitigation: explicit diversity terms in
re-ranking (already partially implemented via the per-author cap in
`serve/api.py`), or epsilon-greedy / bandit-style exploration injected into
the candidate set.

**"How would you validate this online, not just offline?"**
Offline metrics (Recall@K, NDCG@K, AUC) tell you whether the model improved
relative to a previous model on historical data — they do not tell you
whether engagement actually increases by serving real traffic to it,
because user behavior is causally affected by what's shown.
The standard answer: an A/B test, randomizing users into control (old
ranker) vs. treatment (new ranker), with a pre-registered primary metric
(e.g., session-level engagement, day-N retention) and guardrail metrics
(e.g., reports/complaints rate, diversity) to catch regressions the primary
metric wouldn't show.

**"What would you change for a 1000x bigger system?"**
Swap the brute-force `all_item_embeddings` + dot product in
`two_tower.py`/`api.py` for a real ANN index (FAISS/ScaNN/HNSW); move
feature computation (graph features, embeddings) to a feature store with
online and offline parity; precompute and cache user embeddings on a
schedule rather than recomputing per-request; consider a lightweight
"pre-ranking" stage between retrieval and full ranking if 500 candidates is
still too many to score at the required latency.

## 7. Honest limitations to bring up proactively

Naming these yourself, before being asked, is a stronger signal than being
caught off guard by them:

- This is trained on synthetic data with simplified dynamics; real user
  behavior has much more complex temporal and contextual structure
  (session-level sequences, time-of-day effects, device/context features).
- No exploration/exploitation mechanism is implemented yet (e.g., a bandit
  layer) — recommendations are purely exploitative given current model
  estimates.
- The retrieval stage uses brute-force similarity search; at real scale
  this would need an ANN index.
- Fairness across creators (e.g., does the author-PageRank prior
  systematically suppress new/small creators?) is not evaluated here and
  would be worth adding.
