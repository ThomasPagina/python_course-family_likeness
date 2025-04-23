import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import string
from gensim.models import KeyedVectors
import pandas as pd

def clean_term(term):
    return term.strip(string.punctuation)

def load_models():
    print("⏳ Loading 19th-century KeyedVectors (word2vec-1900.kv)…")
    kv_hist = KeyedVectors.load("models/word2vec-1900.kv", mmap='r')
    print("✅ 19th-century model loaded.")
    print("⏳ Loading modern fastText model (cc.de.300.vec.gz)…")
    kv_mod = KeyedVectors.load_word2vec_format(
        "models/cc.de.300.vec.gz",
        binary=False,
        unicode_errors='ignore'
    )
    print("✅ Modern model loaded.\n")
    return kv_hist, kv_mod

def get_top_terms(kv, word, n=5):
    cleaned_target = clean_term(word).lower()
    raw = kv.most_similar(word, topn=n*2)
    results = []
    for w, score in raw:
        cleaned = clean_term(w)
        if cleaned.lower() == cleaned_target:
            continue
        results.append((cleaned, score))
        if len(results) >= n:
            break
    return dict(results)

def make_dataframe(word, kv_hist, kv_mod):
    hist = get_top_terms(kv_hist, word, n=5)
    modern = get_top_terms(kv_mod, word, n=5)
    all_terms = sorted(set(hist) | set(modern),
                       key=lambda t: max(hist.get(t,0), modern.get(t,0)),
                       reverse=True)
    rows = []
    for term in all_terms:
        rows.append({"term": term, "model": "19th-century", "score": hist.get(term, 0.0)})
        rows.append({"term": term, "model": "Modern", "score": modern.get(term, 0.0)})
    return pd.DataFrame(rows)

def plot_all(words, kv_hist, kv_mod):
    sns.set(style="whitegrid")
    n = len(words)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8, 3*n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, word in zip(axes, words):
        df = make_dataframe(word, kv_hist, kv_mod)
        sns.barplot(
            data=df,
            y="term",
            x="score",
            hue="model",
            orient="h",
            dodge=True,
            ax=ax
        )
        ax.set_title(f"Top similar words for '{word}'")
        ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

def main():
    kv_hist, kv_mod = load_models()
    target_words = ["Haus", "Pferd", "Schüler", "Gast", "Mutter"]
    plot_all(target_words, kv_hist, kv_mod)

if __name__ == "__main__":
    main()
