
import os
import re
import pandas as pd
from collections import Counter
import warnings


import plotly.express as px

warnings.filterwarnings("ignore")

DATA_FOLDER = "../../data_cleaned"
SHOWCASE_WORDS = ["স্বাধীনতা", "ডিজিটাল", "কুল"]
ERA_FILES = {
    "1950-1970": "1950_1970.txt",
    "1970-1990": "1970_1990.txt",
    "1990-2010": "1990_2010.txt",
    "2010-2025": "2010_2025.txt"
}
ALPHA = 1.0

def calculate_smoothed_frequencies(words_to_track, data_folder, era_files, alpha):
    """Calculates smoothed word frequencies per million."""
    print("Calculating frequencies...")
    frequency_data = []
    for era_name, filename in era_files.items():
        file_path = os.path.join(data_folder, filename)
        if not os.path.exists(file_path): 
            print(f"Data file not found: {file_path}, skipping.")
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
        except IOError as e:
            print(f"Error reading file {file_path}: {e}")
            continue
            
        tokens = re.split(r'\s+', full_text.strip())
        if not tokens or tokens == ['']: continue
        
        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        vocab_size = len(token_counts)
        
        for word in words_to_track:
            word_count = token_counts.get(word, 0)
            numerator = 1_000_000 * (word_count + alpha)
            denominator = total_tokens + (alpha * vocab_size)
            freq_pm = numerator / denominator if denominator != 0 else 0
            frequency_data.append({"era": era_name, "word": word, "frequency_pm": freq_pm})
            
    return pd.DataFrame(frequency_data)

def plot_frequency_trends_plotly(df):
    """Generates and saves the frequency trend plot using Plotly."""
    print("Generating plot with Plotly...")
    if df.empty: 
        print("DataFrame is empty. Cannot generate plot.")
        return

    high_contrast_colors = px.colors.qualitative.Bold  

    fig = px.line(df, 
                  x='era', 
                  y='frequency_pm', 
                  color='word',
                  markers=True,
                  color_discrete_sequence=high_contrast_colors,  
                  labels={
                      "era": "Era",
                      "frequency_pm": "Frequency (per million tokens, smoothed)",
                      "word": "Showcase Word"
                  },
                  title="Frequency Trends of Showcase Words Across Eras")

    
    fig.update_layout(
        font_family="Times New Roman, serif",
        font_size=15,
        title_font_size=20,
        legend_title_font_size=16,
        template="plotly_white",
      
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        legend=dict(
            bordercolor='black',
            borderwidth=2,
            bgcolor='white'
        )
    )
    

    fig.update_traces(
        line=dict(width=4.0),  
        marker=dict(size=12, line=dict(width=2, color='black'))  
    )
    
   
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1.5, 
        gridcolor='LightGray',
        linecolor='black',
        linewidth=2
    )
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1.5, 
        gridcolor='LightGray',
        linecolor='black',
        linewidth=2
    )

    # --- Save the plot ---
    output_filename_png = "figure_7_frequency_trends_plotly.png"
    try:
        fig.write_image(output_filename_png, width=1200, height=650, scale=2)
        print(f"Plot saved as '{output_filename_png}'")
    except ValueError as e:
        print(f"Error saving image: {e}")
        print("Please ensure you have the 'kaleido' package installed (`pip install kaleido`)")
    

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    freq_df = calculate_smoothed_frequencies(SHOWCASE_WORDS, DATA_FOLDER, ERA_FILES, ALPHA)
    if not freq_df.empty:
        plot_frequency_trends_plotly(freq_df)
    else:
        print("No frequency data was generated. Check your data folder and file paths.")