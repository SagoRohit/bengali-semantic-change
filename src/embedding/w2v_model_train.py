import os
import re
import warnings
from collections import Counter
import pandas as pd
import numpy as np
from gensim.models import FastText
from gensim.models import Word2Vec
from scipy.linalg import orthogonal_procrustes
import umap
import plotly.graph_objects as go
from gensim.models import KeyedVectors

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_FOLDER = "../../data_cleaned" 
MODEL_FOLDER = "models" 
N_VISUAL_NEIGHBORS = 15

ERAS = ["1950-1970", "1970-1990", "1990-2010", "2010-2025"]
ERA_FILES = {
    "1950-1970": "1950_1970.txt",
    "1970-1990": "1970_1990.txt",
    "1990-2010": "1990_2010.txt",
    "2010-2025": "2010_2025.txt"
}

VISUALIZATION_ERA_PAIR = ("1990-2010", "2010-2025")
VISUALIZATION_TARGET_WORD = "স্বাধীনতা" 

FT_PARAMS = {
    "vector_size": 300,
    "window": 5,
    "min_count": 2,
    "sg": 1,  # 1 for Skip-gram
    "negative": 10
}
N_ANCHOR_WORDS = 5000 
K_NEIGHBORS = 50

# def train_models_per_era():
#     """Trains a fastText model for each era and saves it."""
#     print("--- Training Models for Each Era ---")
#     os.makedirs(MODEL_FOLDER, exist_ok=True)
    
#     file_sizes = {era: os.path.getsize(os.path.join(DATA_FOLDER, fname)) 
#                   for era, fname in ERA_FILES.items() if os.path.exists(os.path.join(DATA_FOLDER, fname))}
#     min_size = min(file_sizes.values()) if file_sizes else 1

#     for era, filename in ERA_FILES.items():
#         file_path = os.path.join(DATA_FOLDER, filename)
#         if not os.path.exists(file_path): 
#             print(f"Warning: File for era '{era}' not found. Skipping training.")
#             continue
        
#         print(f"Training model for era: {era}...")
#         with open(file_path, 'r', encoding='utf-8') as f:
#             text = f.read()
#         sentences_str = text.split('<s>')
#         sentences = [re.split(r'\s+', s.strip()) for s in sentences_str if s.strip()]

#         epochs = max(5, int(25 * min_size / file_sizes.get(era, min_size)))
#         print(f"  Using {epochs} epochs.")

#         model = FastText(sentences, epochs=epochs, **FT_PARAMS)
#         model.save(os.path.join(MODEL_FOLDER, f"fasttext_{era}.model"))
#     print("All models trained and saved.")
def train_models_per_era():
    """Trains a Word2Vec model for each era and saves it."""
    print("--- Training Word2Vec Models for Each Era ---")
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    
    # Get file sizes to adjust epochs
    file_sizes = {era: os.path.getsize(os.path.join(DATA_FOLDER, fname)) 
                  for era, fname in ERA_FILES.items() if os.path.exists(os.path.join(DATA_FOLDER, fname))}
    min_size = min(file_sizes.values()) if file_sizes else 1

    for era, filename in ERA_FILES.items():
        file_path = os.path.join(DATA_FOLDER, filename)
        if not os.path.exists(file_path): 
            print(f"Warning: File for era '{era}' not found. Skipping training.")
            continue
        
        print(f"Training model for era: {era}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        sentences_str = text.split('<s>')
        sentences = [re.split(r'\s+', s.strip()) for s in sentences_str if s.strip()]

        epochs = max(5, int(25 * min_size / file_sizes.get(era, min_size)))
        print(f"  Using {epochs} epochs.")

        
        model = Word2Vec(sentences, epochs=epochs, **FT_PARAMS)
        
        
        model.save(os.path.join(MODEL_FOLDER, f"word2vec_{era}.model"))
    print("All Word2Vec models trained and saved.")
def align_models(model1, model2):
    """Aligns model2's vector space to model1's using Orthogonal Procrustes."""
    vocab1 = {word for word, vo in model1.wv.key_to_index.items()}
    vocab2 = {word for word, vo in model2.wv.key_to_index.items()}
    shared_vocab = list(vocab1 & vocab2)
    shared_vocab.sort(key=lambda w: model1.wv.get_vecattr(w, "count"), reverse=True)
    anchor_words = shared_vocab[:N_ANCHOR_WORDS]

    if len(anchor_words) < 10:
        print("Warning: Very few anchor words. Alignment may be unstable.")

    A = np.array([model1.wv[w] for w in anchor_words])
    B = np.array([model2.wv[w] for w in anchor_words])

    R, _ = orthogonal_procrustes(B, A)
    aligned_vectors = {word: model2.wv[word] @ R for word in model2.wv.index_to_key}
    
    model2_aligned_kv = KeyedVectors(vector_size=model2.wv.vector_size)
    model2_aligned_kv.add_vectors(list(aligned_vectors.keys()), list(aligned_vectors.values()))
    
    return model1.wv, model2_aligned_kv
def calculate_drift_metrics(kv1, kv2_aligned):
    """Calculates self-similarity and neighbor overlap for the shared vocabulary."""
    shared_vocab = list(set(kv1.index_to_key) & set(kv2_aligned.index_to_key))
    metrics = []
    total_words = len(shared_vocab)
    
    for i, word in enumerate(shared_vocab):
        if (i+1) % 1000 == 0: print(f"  Calculating metrics for word {i+1}/{total_words}...")
        vec1, vec2 = kv1[word], kv2_aligned[word]
        cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        delta_cos = 1 - cos_sim

        neighbors1 = set([w for w, s in kv1.most_similar(word, topn=K_NEIGHBORS)])
        neighbors2 = set([w for w, s in kv2_aligned.most_similar(word, topn=K_NEIGHBORS)])
        jaccard_sim = len(neighbors1 & neighbors2) / len(neighbors1 | neighbors2) if len(neighbors1 | neighbors2) > 0 else 0
        
        metrics.append({"word": word, "delta_cos": delta_cos, "neighbor_jaccard": jaccard_sim})
    return pd.DataFrame(metrics)

def plot_neighborhood_drift(target_word, kv1, kv2_aligned, era1_name, era2_name):
    """Generates a 2D UMAP projection that shows all top neighbors, even if they don't exist in both eras."""
    print(f"\n--- Generating Improved 2D Drift Visualization for '{target_word}' ---")
    
 
    neighbors1 = {w for w, s in kv1.most_similar(target_word, topn=N_VISUAL_NEIGHBORS)}
    neighbors2 = {w for w, s in kv2_aligned.most_similar(target_word, topn=N_VISUAL_NEIGHBORS)}
    
 
    words_to_map = {target_word} | neighbors1 | neighbors2
    all_vectors, vector_labels = [], []
    for word in words_to_map:
        if word in kv1:
            all_vectors.append(kv1[word])
            vector_labels.append({'word': word, 'era': era1_name})
        if word in kv2_aligned:
            all_vectors.append(kv2_aligned[word])
            vector_labels.append({'word': word, 'era': era2_name})

    if not all_vectors:
        print("No vectors found for the given words. Skipping plot.")
        return
        

    reducer = umap.UMAP(n_components=2, random_state=42, min_dist=0.5, n_neighbors=15)
    embedding_2d = reducer.fit_transform(np.array(all_vectors))
    
   
    plot_df = pd.DataFrame(embedding_2d, columns=['x', 'y'])
    plot_df = plot_df.join(pd.DataFrame(vector_labels))
    plot_df['size'] = [30 if w == target_word else 10 for w in plot_df['word']]
    plot_df['symbol'] = ['star' if w == target_word else 'circle' for w in plot_df['word']]

    fig = go.Figure()
    
   
    for era_name, color in [(era1_name, 'blue'), (era2_name, 'red')]:
        era_df = plot_df[plot_df['era'] == era_name]
        fig.add_trace(go.Scatter(
            x=era_df['x'], y=era_df['y'],
            mode='markers+text', text=era_df['word'],
            textposition="top center",
            marker=dict(color=color, size=era_df['size'], symbol=era_df['symbol']),
            name=era_name,
            textfont=dict(size=12)
        ))
   
    pos_t1 = plot_df[(plot_df['word'] == target_word) & (plot_df['era'] == era1_name)]
    pos_t2 = plot_df[(plot_df['word'] == target_word) & (plot_df['era'] == era2_name)]
    if not pos_t1.empty and not pos_t2.empty:
        fig.add_annotation(
            ax=pos_t1['x'].iloc[0], ay=pos_t1['y'].iloc[0],
            x=pos_t2['x'].iloc[0], y=pos_t2['y'].iloc[0],
            showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor='black'
        )
        

    fig.update_layout(
        title=f"2D Neighborhood Drift for '{target_word}' ({era1_name} vs. {era2_name} Aligned)",
        template="plotly_white",
        font_family="Kalpurush, Arial",
        font_size=14,
        title_font_size=18
    )

  
    fig.update_xaxes(
        title_text="UMAP Dimension 1",
        visible=True,
        title_font_size=16
    )
    fig.update_yaxes(
        title_text="UMAP Dimension 2",
        visible=True,
        title_font_size=16
    )
    
    
    
    output_filename = f"figure_4_drift_{target_word}_{era1_name}_{era2_name}_corrected.jpg"
    fig.write_image(output_filename, width=1200, height=800, scale=2)
    print(f"Corrected drift plot saved as '{output_filename}'")
    fig.show()

if __name__ == "__main__":
    
    # train_models_per_era()
    
    
    print(f"\n--- Now generating the specific plot for {VISUALIZATION_ERA_PAIR[0]} vs {VISUALIZATION_ERA_PAIR[1]} ---")
    
    viz_era1_name, viz_era2_name = VISUALIZATION_ERA_PAIR
    try:
     
        model1_viz = Word2Vec.load(os.path.join(MODEL_FOLDER, f"word2vec_{viz_era1_name}.model"))
        model2_viz = Word2Vec.load(os.path.join(MODEL_FOLDER, f"word2vec_{viz_era2_name}.model"))
     
        kv1_viz, kv2_aligned_viz = align_models(model1_viz, model2_viz)
        
     
        if VISUALIZATION_TARGET_WORD in kv1_viz and VISUALIZATION_TARGET_WORD in kv2_aligned_viz:
            plot_neighborhood_drift(VISUALIZATION_TARGET_WORD, kv1_viz, kv2_aligned_viz, viz_era1_name, viz_era2_name)
        else:
            print(f"\n--- Skipping Visualization ---")
            print(f"Warning: Target word '{VISUALIZATION_TARGET_WORD}' not frequent enough in both eras for the pair {viz_era1_name} -> {viz_era2_name}.")

    except FileNotFoundError:
        print(f"Error: Could not find model files for the visualization pair. Please ensure models for {viz_era1_name} and {viz_era2_name} are trained.")

    print("\n--- Full Process Finished ---")