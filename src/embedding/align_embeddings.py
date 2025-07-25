import os
import numpy as np
from gensim.models import  FastText
from scipy.linalg import orthogonal_procrustes

# ==== Config ====
MODELS_DIR = "../../models"
ALIGNED_DIR = "../../models/aligned"
MODEL_SUFFIX = "_fasttext.model"   # use "_fasttext.model" for fastText if needed

# List your eras in correct order:
ERAS = [
    "1950_1970",
    "1970_1990",
    "1990_2010",
    "2010_2025",
]

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def load_model(path):
    print(f"Loading {path}")
    return FastText.load(path)

def get_common_vocab(model1, model2, top_n=None):
    vocab1 = set(model1.wv.index_to_key)
    vocab2 = set(model2.wv.index_to_key)
    common = list(vocab1 & vocab2)
    if top_n:
        # Optionally, restrict to top-N frequent words (stable)
        freq_sorted = sorted(common, key=lambda w: model1.wv.get_vecattr(w, "count") + model2.wv.get_vecattr(w, "count"), reverse=True)
        common = freq_sorted[:top_n]
    return common

def align_models(src_model, tgt_model, common_vocab):
    # Get matrices
    src_vecs = np.vstack([src_model.wv[w] for w in common_vocab])
    tgt_vecs = np.vstack([tgt_model.wv[w] for w in common_vocab])
    # Find orthogonal matrix R that maps src_vecs to tgt_vecs
    R, _ = orthogonal_procrustes(src_vecs, tgt_vecs)
    # Apply alignment to all word vectors in src_model
    for w in src_model.wv.index_to_key:
        src_model.wv[w] = np.dot(src_model.wv[w], R)
    return src_model, R

def main():
    ensure_dir(ALIGNED_DIR)
    prev_model = None
    prev_era = None

    for i, era in enumerate(ERAS):
        model_path = os.path.join(MODELS_DIR, f"{era}{MODEL_SUFFIX}")
        model = load_model(model_path)

        if prev_model is not None:
            print(f"\nAligning {prev_era} → {era}")
            # Only align common vocab to avoid instability from rare words
            common_vocab = get_common_vocab(prev_model, model, top_n=5000)  # top 5000 is a good start
            aligned_prev_model, R = align_models(prev_model, model, common_vocab)
            # Save aligned previous model for downstream tasks
            aligned_model_path = os.path.join(ALIGNED_DIR, f"{prev_era}_aligned_to_{era}{MODEL_SUFFIX}")
            aligned_prev_model.save(aligned_model_path)
            print(f"Aligned model saved to {aligned_model_path}")
            # Save the alignment matrix for transparency
            np.save(os.path.join(ALIGNED_DIR, f"{prev_era}_to_{era}_R.npy"), R)
        prev_model = model
        prev_era = era

    # Optionally, save the final model (not aligned to any later period)
    last_model_path = os.path.join(ALIGNED_DIR, f"{ERAS[-1]}{MODEL_SUFFIX}")
    prev_model.save(last_model_path)
    print(f"Final period model saved to {last_model_path}")

if __name__ == "__main__":
    main()
