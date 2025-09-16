import pandas as pd
import plotly.express as px

DRIFT_METRICS_FILE = "drift_metrics_1950-1970_vs_2010-2025.csv"
TOP_N_TO_DISPLAY = 20

def analyze_and_rank_words(filepath):
    """Loads drift metrics, calculates a final change score, and ranks words."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        print("Please run the embedding analysis script first to generate this file.")
        return None
    df['change_score'] = df['delta_cos'] + (1 - df['neighbor_jaccard'])
    df_sorted = df.sort_values(by='change_score', ascending=False).reset_index(drop=True)
    
    return df_sorted
def plot_score_distribution(df):
    """Generates a histogram of the ChangeScore distribution."""
    if df is None:
        return
        
    fig = px.histogram(
        df,
        x='change_score',
        title=f"Distribution of Semantic Change Scores ({DRIFT_METRICS_FILE.split('_')[2].split('.')[0]} vs. {DRIFT_METRICS_FILE.split('_')[4].split('.')[0]})",
        labels={'change_score': 'Calculated Change Score'},
        nbins=100
    )
    
    mean_score = df['change_score'].mean()
    fig.add_vline(x=mean_score, line_width=3, line_dash="dash", line_color="red",
                  annotation_text=f"Mean = {mean_score:.2f}", annotation_position="top left")
                  
    fig.update_layout(
        template="plotly_white",
        font_family="Arial"
    )

    output_filename = f"figure_5_changescore_distribution.png"
    fig.write_image(output_filename, width=1000, height=600, scale=2)
    print(f"\nHistogram saved as '{output_filename}'")
    fig.show()

if __name__ == "__main__":
    ranked_df = analyze_and_rank_words(DRIFT_METRICS_FILE)
    
    if ranked_df is not None:
      
        top_n_changed = ranked_df.head(TOP_N_TO_DISPLAY).reset_index(drop=True)
        top_n_changed.rename(columns={'word': 'Most Changed', 'change_score': 'Change Score (High)'}, inplace=True)
        top_n_stable = ranked_df.tail(TOP_N_TO_DISPLAY).sort_values(by='change_score', ascending=True).reset_index(drop=True)
        top_n_stable.rename(columns={'word': 'Most Stable', 'change_score': 'Change Score (Low)'}, inplace=True)
        
        combined_df = pd.concat([top_n_changed, top_n_stable], axis=1)

        output_filename = "ranked_word_lists.csv"
        combined_df.to_csv(output_filename, index=False, encoding='utf-8')
        
        print("--- Ranked Word Lists ---")
        print(f"Top {TOP_N_TO_DISPLAY} most changed and stable words have been saved to '{output_filename}'")
        
        plot_score_distribution(ranked_df)