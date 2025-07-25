import os
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec , FastText  # or FastText if using fasttext models
from scipy.spatial.distance import cosine

# ===== CONFIGURATION =====
MODELS_DIR = "../../models/aligned"
MODEL_SUFFIX = "_w2v.model"  # "_fasttext.model" for fastText
ERAS = [
    "1950_1970",
    "1970_1990",
    "1990_2010",
    "2010_2025",
]
TARGET_WORD = "করেছে"  # Change this to any word you want to analyze (should exist in all eras)
# =========================

def load_models(suffix):
    models = []
    aligned_filenames = [
        "1950_1970_aligned_to_1970_1990" + suffix,
        "1970_1990_aligned_to_1990_2010" + suffix,
        "1990_2010_aligned_to_2010_2025" + suffix,
        "2010_2025" + suffix,
    ]
    for fname in aligned_filenames:
        model_path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(model_path):
            models.append(Word2Vec.load(model_path)) #need to change here 
        else:
            print(f"Model not found: {model_path}")
            models.append(None)
    return models


def get_vectors(word, models):
    vecs = []
    for model in models:
        if model is not None and word in model.wv:
            vecs.append(model.wv[word])
        else:
            vecs.append(None)
    return vecs

def compute_cosine_similarities(vecs):
    sims = []
    for i in range(len(vecs) - 1):
        if vecs[i] is not None and vecs[i+1] is not None:
            sim = 1 - cosine(vecs[i], vecs[i+1])
        else:
            sim = np.nan
        sims.append(sim)
    return sims

def plot_cosine_similarities(eras, sims, word):
    plt.rcParams['font.family'] = 'Noto Sans Bengali'
    # Eras between pairs for x-axis labels
    labels = [f"{eras[i]}→{eras[i+1]}" for i in range(len(eras)-1)]
    plt.figure(figsize=(8,4))
    plt.plot(labels, sims, marker="o", linestyle="-", color="tab:blue")
    plt.ylim(0, 1)
    plt.xlabel("Era Pair")
    plt.ylabel("Cosine Similarity")
    plt.title(f"Semantic Stability of '{word}' Across Time (Cosine Similarity)")
    for i, sim in enumerate(sims):
        plt.text(i, sim+0.02, f"{sim:.2f}", ha='center')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    models = load_models(MODEL_SUFFIX)
    vecs = get_vectors(TARGET_WORD, models)
    sims = compute_cosine_similarities(vecs)
    print(f"Cosine similarities for '{TARGET_WORD}': {sims}")
    plot_cosine_similarities(ERAS, sims, TARGET_WORD)

if __name__ == "__main__":
    main()
