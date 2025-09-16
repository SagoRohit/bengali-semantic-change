import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.io as pio
from tqdm import tqdm


EARLY_ERA_MODEL = "models/word2vec_1950-1970.model"
DRIFT_METRICS_FILE = "drift_metrics_1950-1970_vs_2010-2025.csv"
K_NEIGHBORS_DISPERSION = 50

def calculate_neighbor_dispersion(word, model, k):
    """
    Calculates the average pairwise cosine distance among a word's top k neighbors.
    This serves as a proxy for polysemy.
    """
    try:
       
        neighbor_words = [n for n, s in model.wv.most_similar(word, topn=k)]
        neighbor_vectors = [model.wv[w] for w in neighbor_words]
        
        if len(neighbor_vectors) < 2:
            return None # Cannot calculate dispersion with fewer than 2 neighbors

        
        sim_matrix = cosine_similarity(neighbor_vectors)
        
       
        avg_similarity = np.mean(sim_matrix[np.triu_indices(len(sim_matrix), k=1)])
        
        return 1 - avg_similarity
        
    except (KeyError, IndexError):
       
        return None

def run_polysemy_law_analysis():
    """Main function to test the law of polysemy."""
    print("--- Testing Law 2: Polysemy vs. Rate of Change ---")

    
    try:
        df_change = pd.read_csv(DRIFT_METRICS_FILE)
    except FileNotFoundError:
        print(f"Error: Drift metrics file not found at '{DRIFT_METRICS_FILE}'")
        return
    try:
        model = Word2Vec.load(EARLY_ERA_MODEL)
    except FileNotFoundError:
        print(f"Error: Word2Vec model not found at '{EARLY_ERA_MODEL}'")
        return

    print(f"Calculating neighbor dispersion for {len(df_change)} words...")
    tqdm.pandas(desc="Dispersion Calculation")
    df_change['polysemy_proxy'] = df_change['word'].progress_apply(
        lambda w: calculate_neighbor_dispersion(w, model, K_NEIGHBORS_DISPERSION)
    )
    
   
    df_analysis = df_change.dropna(subset=['polysemy_proxy', 'change_score']).copy()

   
    correlation, p_value = spearmanr(df_analysis['polysemy_proxy'], df_analysis['change_score'])
    
    print("\n--- Statistical Results ---")
    print(f"Spearman's Correlation (ρ): {correlation:.4f}")
    print(f"P-value: {p_value:.4g}")

    if p_value < 0.05:
        print("The positive correlation is statistically significant.")
    else:
        print("The correlation is not statistically significant.")

    
    print("\nGenerating scatter plot...")
    fig = px.scatter(
        df_analysis,
        x='polysemy_proxy',
        y='change_score',
        title='Rate of Semantic Change vs. Polysemy Proxy',
        labels={
            'polysemy_proxy': 'Polysemy Proxy (Neighbor Dispersion)',
            'change_score': 'Total Change Score (1950-2025)'
        },
        trendline='ols', 
        trendline_color_override='red'
    )
    
    fig.update_layout(template="plotly_white")
    
    output_filename = "figure_law2_polysemy_vs_change.jpg"
    pio.write_image(fig, output_filename, width=1000, height=600, scale=2)
    print(f"Scatter plot saved as '{output_filename}'")
    fig.show()

if __name__ == "__main__":
    run_polysemy_law_analysis()