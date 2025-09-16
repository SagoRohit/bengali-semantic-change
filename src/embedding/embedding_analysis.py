import os
import re
import warnings
from collections import Counter
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from scipy.linalg import orthogonal_procrustes
import umap
import plotly.graph_objects as go

warnings.filterwarnings("ignore", category=FutureWarning)

# --- 1. CONFIGURATION ---
DATA_FOLDER = "../../data_cleaned" 
MODEL_FOLDER = "models" 
ERAS = ["1950-1970", "1970-1990", "1990-2010", "2010-2025"]
ERA_FILES = {
    "1950-1970": "1950_1970.txt",
    "1970-1990": "1970_1990.txt",
    "1990-2010": "1990_2010.txt",
    "2010-2025": "2010_2025.txt"
}

VISUALIZATION_ERA_PAIR = ("1990-2010", "2010-2025")
VISUALIZATION_TARGET_WORD = "ডিজিটাল" 
N_VISUAL_NEIGHBORS = 15 

# Hyperparameters for Word2Vec models
W2V_PARAMS = {
    "vector_size": 300,
    "window": 5,
    "min_count": 2,
    "sg": 1,      # 1 for Skip-gram
    "negative": 10
}

# Parameters for alignment and metric calculation
N_ANCHOR_WORDS = 5000 
K_NEIGHBORS = 50      # K for calculating Jaccard similarity
def train_models_per_era():
    """Trains a Word2Vec model for each era and saves it."""
    print("--- Training Word2Vec Models for Each Era ---")
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    
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

        epochs = max(5, int(50 * min_size / file_sizes.get(era, min_size)))
        print(f"  Using {epochs} epochs.")

        model = Word2Vec(sentences, epochs=epochs, **W2V_PARAMS)
        model.save(os.path.join(MODEL_FOLDER, f"word2vec_{era}.model"))
    print("All Word2Vec models trained and saved.")

# --- 3. SPACE ALIGNMENT VIA ORTHOGONAL PROCRUSTES ---
def align_models(model1, model2):
    """Aligns model2's vector space to model1's using Orthogonal Procrustes."""
    vocab1 = {word for word in model1.wv.key_to_index}
    vocab2 = {word for word in model2.wv.key_to_index}
    shared_vocab = list(vocab1 & vocab2)
    shared_vocab.sort(key=lambda w: model1.wv.get_vecattr(w, "count"), reverse=True)
    anchor_words = shared_vocab[:N_ANCHOR_WORDS]

    if len(anchor_words) < 100:
        print(f"Warning: Only {len(anchor_words)} anchor words found. Alignment may be unstable.")

    A = np.array([model1.wv[w] for w in anchor_words])
    B = np.array([model2.wv[w] for w in anchor_words])

    R, _ = orthogonal_procrustes(B, A)
    aligned_vectors = {word: model2.wv[word] @ R for word in model2.wv.index_to_key}
    
    model2_aligned_kv = KeyedVectors(vector_size=model2.wv.vector_size)
    model2_aligned_kv.add_vectors(list(aligned_vectors.keys()), list(aligned_vectors.values()))
    return model1.wv, model2_aligned_kv

# --- 4. DRIFT AND NEIGHBORHOOD SHIFT CALCULATION ---
def calculate_drift_metrics(kv1, kv2_aligned):
    """Calculates self-similarity and neighbor overlap for the shared vocabulary."""
    shared_vocab = list(set(kv1.index_to_key) & set(kv2_aligned.index_to_key))
    metrics = []
    
    for word in shared_vocab:
        vec1, vec2 = kv1[word], kv2_aligned[word]
        cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        delta_cos = 1 - cos_sim

        neighbors1 = set([w for w, s in kv1.most_similar(word, topn=K_NEIGHBORS)])
        neighbors2 = set([w for w, s in kv2_aligned.most_similar(word, topn=K_NEIGHBORS)])
        jaccard_sim = len(neighbors1 & neighbors2) / len(neighbors1 | neighbors2) if len(neighbors1 | neighbors2) > 0 else 0
        
        metrics.append({"word": word, "delta_cos": delta_cos, "neighbor_jaccard": jaccard_sim})
    return pd.DataFrame(metrics)

# --- 5. 2D PROJECTION VISUALIZATION ---
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
        
    reducer = umap.UMAP(n_components=2, random_state=42, min_dist=0.5, n_neighbors=min(15, len(all_vectors)-1))
    embedding_2d = reducer.fit_transform(np.array(all_vectors))
    
    plot_df = pd.DataFrame(embedding_2d, columns=['x', 'y'])
    plot_df = plot_df.join(pd.DataFrame(vector_labels))
    plot_df['size'] = [30 if w == target_word else 10 for w in plot_df['word']]
    plot_df['symbol'] = ['star' if w == target_word else 'circle' for w in plot_df['word']]

    fig = go.Figure()
    for era_name_plot, color in [(era1_name, 'blue'), (era2_name, 'red')]:
        era_df = plot_df[plot_df['era'] == era_name_plot]
        fig.add_trace(go.Scatter(
            x=era_df['x'], y=era_df['y'],
            mode='markers+text', text=era_df['word'],
            textposition="top center",
            marker=dict(color=color, size=era_df['size'], symbol=era_df['symbol']),
            name=era_name_plot
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
        template="plotly_white", font_family="Kalpurush, Arial"
    )
    fig.update_xaxes(title_text="UMAP Dimension 1", visible=True)
    fig.update_yaxes(title_text="UMAP Dimension 2", visible=True)
    
    output_filename = f"figure_4_drift_{target_word}_{era1_name}_{era2_name}.png"
    fig.write_image(output_filename, width=1200, height=800, scale=2)
    print(f"Drift plot saved as '{output_filename}'")
    fig.show()


if __name__ == "__main__":
    # train_models_per_era()
  
    for i in range(len(ERAS) - 1):
        era1_name = ERAS[i]
        era2_name = ERAS[i+1]
        
        print(f"\n--- Processing Transition: {era1_name} -> {era2_name} ---")
        
        try:
            model1 = Word2Vec.load(os.path.join(MODEL_FOLDER, f"word2vec_{era1_name}.model"))
            model2 = Word2Vec.load(os.path.join(MODEL_FOLDER, f"word2vec_{era2_name}.model"))
        except FileNotFoundError:
            print(f"Error: Could not find model files for transition. Skipping.")
            continue
            
        kv1, kv2_aligned = align_models(model1, model2)
        drift_df = calculate_drift_metrics(kv1, kv2_aligned)
        
        drift_df['change_score'] = drift_df['delta_cos'] + (1 - drift_df['neighbor_jaccard'])
        drift_df_sorted = drift_df.sort_values(by='change_score', ascending=False)
        
        output_csv_path = f"drift_metrics_{era1_name}_vs_{era2_name}.csv"
        drift_df_sorted.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"Drift metrics for this transition saved to '{output_csv_path}'")
        
    print("\n--- Consecutive Analysis Complete ---")

    print(f"\n--- Now processing the TOTAL CHANGE analysis for {ERAS[0]} vs {ERAS[-1]} ---")
    
    total_change_era1, total_change_era2 = ERAS[0], ERAS[-1]
    
    try:
        model1_total = Word2Vec.load(os.path.join(MODEL_FOLDER, f"word2vec_{total_change_era1}.model"))
        model2_total = Word2Vec.load(os.path.join(MODEL_FOLDER, f"word2vec_{total_change_era2}.model"))
        
        kv1_total, kv2_aligned_total = align_models(model1_total, model2_total)
        
        print(f"Calculating drift metrics for {total_change_era1} vs {total_change_era2}...")
        drift_df_total = calculate_drift_metrics(kv1_total, kv2_aligned_total)
        
        drift_df_total['change_score'] = drift_df_total['delta_cos'] + (1 - drift_df_total['neighbor_jaccard'])
        drift_df_total_sorted = drift_df_total.sort_values(by='change_score', ascending=False)

        output_total_csv_path = f"drift_metrics_{total_change_era1}_vs_{total_change_era2}.csv"
        drift_df_total_sorted.to_csv(output_total_csv_path, index=False, encoding='utf-8')
        print(f"Total change drift metrics saved to '{output_total_csv_path}'")
        
        print(f"\nTop 10 words with highest total drift:")
        print(drift_df_total_sorted.head(10).to_string())
        
        if (total_change_era1, total_change_era2) == VISUALIZATION_ERA_PAIR:
            if VISUALIZATION_TARGET_WORD in kv1_total and VISUALIZATION_TARGET_WORD in kv2_aligned_total:
                plot_neighborhood_drift(VISUALIZATION_TARGET_WORD, kv1_total, kv2_aligned_total, total_change_era1, total_change_era2)
            else:
                print(f"Warning: Target word '{VISUALIZATION_TARGET_WORD}' not frequent enough for visualization.")

    except FileNotFoundError:
        print(f"Error: Could not find model files for the total change analysis.")

    print("\n--- Full Process Finished ---")