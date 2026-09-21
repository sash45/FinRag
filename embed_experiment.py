from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    # finance
    "The Federal Reserve raised interest rates to curb inflation.",
    "Apple reported record quarterly revenue driven by iPhone sales.",
    "The fund's net asset value increased after the dividend payout.",
    "Bond yields rose as investors sold off government debt.",
    "The ETF's expense ratio is 0.03 percent.",
    "The company filed its annual 10-K with the SEC.",
    "Inflation data came in hotter than expected.",
    "The merger was approved by shareholders last quarter.",
    # non-finance
    "The quarterback threw a touchdown pass in the final minute.",
    "Chocolate cake requires flour, sugar, and cocoa powder.",
    "The new sci-fi movie broke box office records this weekend.",
    "She hiked through the mountains at sunrise.",
    "The chef added fresh basil to the tomato sauce.",
]

embeddings = model.encode(sentences, normalize_embeddings=True)

print("=== Most similar pairs ===")
pairs = []
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        score = float(util.cos_sim(embeddings[i], embeddings[j]))
        pairs.append((score, sentences[i], sentences[j]))
pairs.sort(reverse=True)
for score, a, b in pairs[:5]:
    print(f"{score:.3f} | {a}\n        | {b}\n")

print("=== Query test ===")
query = model.encode("What did the Fed do with interest rates?", normalize_embeddings=True)
scores = util.cos_sim(query, embeddings)[0]
for score, s in sorted(zip(scores.tolist(), sentences), reverse=True)[:3]:
    print(f"{score:.3f} | {s}")
