import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, FastText
from scipy.spatial.distance import cosine

# =========== CONFIG =============
MODELS_DIR = "../../models/aligned"
ERAS = [
    "1950_1970",
    "1970_1990",
    "1990_2010",
    "2010_2025",
]
ERA_LABELS = [f"{ERAS[i]}→{ERAS[i+1]}" for i in range(len(ERAS)-1)]
MODEL_OPTIONS = {
    "Word2Vec": "_w2v.model",
    "fastText": "_fasttext.model"
}
# ================================

@st.cache_resource(show_spinner="Loading models...")
def load_models(suffix):
    filenames = [
        "1950_1970_aligned_to_1970_1990" + suffix,
        "1970_1990_aligned_to_1990_2010" + suffix,
        "1990_2010_aligned_to_2010_2025" + suffix,
        "2010_2025" + suffix,
    ]
    models = []
    for fname in filenames:
        model_path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(model_path):
            if suffix == "_w2v.model":
                models.append(Word2Vec.load(model_path))
            else:
                models.append(FastText.load(model_path))
        else:
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

def explain_shifts(word, sims):
    if np.isnan(sims).all():
        return f"'{word}' was not found in any era. Try a more common word."
    explanation = []
    for i, sim in enumerate(sims):
        label = ERA_LABELS[i]
        if np.isnan(sim):
            explanation.append(f"Not enough data for {label}.")
        elif sim > 0.85:
            explanation.append(f"• Between {label}: Very stable meaning (cosine similarity {sim:.2f})")
        elif sim > 0.6:
            explanation.append(f"• Between {label}: Mostly stable meaning (cosine similarity {sim:.2f})")
        elif sim > 0.3:
            explanation.append(f"• Between {label}: Moderate change in context/meaning (cosine similarity {sim:.2f})")
        else:
            explanation.append(f"• Between {label}: Big shift in meaning/context! (cosine similarity {sim:.2f})")
    return "\n".join(explanation)

def plot_similarity(sims, word, model_name):
    plt.rcParams['font.family'] = 'Noto Sans Bengali'
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(ERA_LABELS, sims, marker='o', linestyle='-', color="tab:blue")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Era Pair")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(f"Semantic Stability of '{word}' ({model_name})")
    for i, sim in enumerate(sims):
        if not np.isnan(sim):
            ax.text(i, sim+0.02, f"{sim:.2f}", ha='center')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig

# ========== Streamlit UI ==============
st.title("Bangla Semantic Shift Visualizer")
st.write("Enter a Bangla word and select a model to visualize how its meaning changed across decades.")

word = st.text_input("Word (Bangla):", value="কুল", max_chars=20)
model_choice = st.multiselect(
    "Select embedding model(s):", 
    options=list(MODEL_OPTIONS.keys()), 
    default=["Word2Vec"],
    help="You can select both to compare!"
)

if st.button("Show Semantic Shift") or word:
    for model_name in model_choice:
        st.markdown(f"### Model: {model_name}")
        models = load_models(MODEL_OPTIONS[model_name])
        vecs = get_vectors(word, models)
        sims = compute_cosine_similarities(vecs)
        st.subheader("Cosine Similarity Plot")
        fig = plot_similarity(sims, word, model_name)
        st.pyplot(fig)
        st.subheader("Explanation / Analysis")
        st.markdown(explain_shifts(word, sims))
        st.write("Cosine similarities:", {label: (f"{sim:.2f}" if not np.isnan(sim) else "N/A") for label, sim in zip(ERA_LABELS, sims)})

