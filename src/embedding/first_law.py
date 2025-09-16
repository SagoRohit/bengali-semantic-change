import pandas as pd
import numpy as np
from collections import Counter
import re
from scipy.stats import spearmanr
import plotly.express as px
import plotly.io as pio

EARLY_ERA_FILE = "../../data_cleaned/1950_1970.txt"
DRIFT_METRICS_FILE = "drift_metrics_1950-1970_vs_2010-2025.csv"

def get_frequencies(filepath):
    """Reads a text file and returns a Counter of word frequencies."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = re.split(r'\s+', text.strip())
        return Counter(tokens)
    except FileNotFoundError:
        print(f"Error: Frequency file not found at '{filepath}'")
        return None

def run_frequency_law_analysis():
    """Main function to test the law of frequency."""
    print("--- Testing Law 1: Frequency vs. Rate of Change ---")
    try:
        df_change = pd.read_csv(DRIFT_METRICS_FILE)
    except FileNotFoundError:
        print(f"Error: Drift metrics file not found at '{DRIFT_METRICS_FILE}'")
        return
    freq_counts = get_frequencies(EARLY_ERA_FILE)
    if freq_counts is None:
        return
    df_freq = pd.DataFrame(freq_counts.items(), columns=['word', 'frequency'])
    
   
    df_merged = pd.merge(df_change, df_freq, on='word')

    df_merged = df_merged[df_merged['frequency'] > 0].copy()
    df_merged['log_frequency'] = np.log10(df_merged['frequency'])
    
    correlation, p_value = spearmanr(df_merged['log_frequency'], df_merged['change_score'])
    
    print("\n--- Statistical Results ---")
    print(f"Spearman's Correlation (ρ): {correlation:.4f}")
    print(f"P-value: {p_value:.4g}")

    if p_value < 0.05:
        print("The negative correlation is statistically significant.")
    else:
        print("The correlation is not statistically significant.")

    print("\nGenerating scatter plot...")
    fig = px.scatter(
        df_merged,
        x='log_frequency',
        y='change_score',
        title='Rate of Semantic Change vs. Historical Word Frequency',
        labels={
            'log_frequency': 'Log10 Frequency (1950-1970)',
            'change_score': 'Total Change Score (1950-2025)'
        },
        trendline='ols', 
        trendline_color_override='red'
    )
    
    fig.update_layout(template="plotly_white")
    
    output_filename = "figure_law1_frequency_vs_change.jpg"
    pio.write_image(fig, output_filename, width=1000, height=600, scale=2)
    print(f"Scatter plot saved as '{output_filename}'")
    fig.show()

if __name__ == "__main__":
    run_frequency_law_analysis()