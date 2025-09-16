
import os
import re
import pandas as pd
from collections import Counter
import math
import warnings
import plotly.graph_objects as go
import plotly.io as pio

# Suppress minor warnings
warnings.filterwarnings("ignore")

# --- 1. CONFIGURATION ---
DATA_FOLDER = "../../data_cleaned"
TARGET_WORD = "স্বাধীনতা" 
WINDOW_SIZE = 6

TOP_N_PER_ERA = 5 
MIN_CO_OCCURRENCE = 5
ERA_FILES = {
    "1950-1970": "1950_1970.txt",
    "1970-1990": "1970_1990.txt",
    "1990-2010": "1990_2010.txt",
    "2010-2025": "2010_2025.txt"
}

def get_era_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        tokens = re.split(r'\s+', full_text.strip())
        return tokens if tokens != [''] else []
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return []

def calculate_ppmi_for_era(target_word, era_tokens, window_size, era_name):
    """Calculates PPMI scores, applying a co-occurrence threshold only for the modern era."""
    if not era_tokens: return {}
    target_indices = [i for i, token in enumerate(era_tokens) if token == target_word]
    if not target_indices: return {}

    all_co_occurrences = []
    for i in target_indices:
        start_bound, end_bound = max(0, i - window_size), min(len(era_tokens), i + window_size + 1)
        true_start = start_bound
        for j in range(i - 1, start_bound - 1, -1):
            if era_tokens[j] == '<s>': true_start = j + 1; break
        true_end = end_bound
        for j in range(i + 1, end_bound):
            if era_tokens[j] == '<s>': true_end = j; break
        window = era_tokens[true_start:i] + era_tokens[i+1:true_end]
        all_co_occurrences.extend(window)

    total_tokens, p_target = len(era_tokens), len(target_indices) / len(era_tokens)
    co_occurrence_counts, token_counts = Counter(all_co_occurrences), Counter(era_tokens)
    
    ppmi_scores = {}
    for collocate, count in co_occurrence_counts.items():
        if era_name != "2010-2025" or count >= MIN_CO_OCCURRENCE:
            if not all_co_occurrences: continue
            p_collocate = token_counts.get(collocate, 1) / total_tokens
            p_co_occurrence = count / len(all_co_occurrences)
            
            if p_target > 0 and p_collocate > 0:
                pmi = math.log2(p_co_occurrence / (p_target * p_collocate))
                ppmi_scores[collocate] = max(0, pmi)
            
    return ppmi_scores

def plot_collocate_heatmap(target_word, data_folder, era_files, window_size, top_n_per_era):
    """Generates and saves a collocate PPMI heatmap for the target word."""
    print(f"Generating heatmap for target word: '{target_word}'")
    
    all_ppmi_data = {}
    master_collocate_set = set()
    print("Step 1: Identifying top collocates from each era...")
    for era_name, filename in era_files.items():
        file_path = os.path.join(data_folder, filename)
        era_tokens = get_era_data(file_path)
        
       
        ppmi_scores = calculate_ppmi_for_era(target_word, era_tokens, window_size, era_name)
        all_ppmi_data[era_name] = ppmi_scores
        
      
        sorted_era_collocates = sorted(ppmi_scores.items(), key=lambda item: item[1], reverse=True)
        
        # Add the top N from this era to our master list
        for collocate, score in sorted_era_collocates[:top_n_per_era]:
            master_collocate_set.add(collocate)

    if not master_collocate_set:
        print(f"No collocates found for '{target_word}'. Cannot generate heatmap.")
        return
    # Sort the final list alphabetically for a consistent y-axis
    y_axis_words = sorted(list(master_collocate_set))
    
    # 2. Build the data matrix for the heatmap using the master list
    print("Step 2: Building the PPMI matrix for the heatmap...")
    heatmap_data = []
    for word in y_axis_words:
        # For each word in our master list, get its PPMI score from each era
        row = [all_ppmi_data[era].get(word, 0) for era in era_files.keys()]
        heatmap_data.append(row)
    # >>>>>>>> LOGIC CHANGE ENDS HERE <<<<<<<<

    # 3. Create the Plotly Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=list(era_files.keys()),
        y=y_axis_words,
        colorscale='RdYlGn', 
        colorbar=dict(title='PPMI Score')
    ))

    # 4. Customize layout
    fig.update_layout(
        title=f"Collocate Heatmap (PPMI) for '{target_word}'",
        xaxis_title="Era",
        yaxis_title="Top Collocates (from any era)",
        font=dict(family="Kalpurush, Arial", size=14),
        yaxis=dict(autorange="reversed"),
        template="plotly_white"
    )
    fig.update_traces(
        text=[[f'{val:.2f}' for val in row] for row in heatmap_data],
        texttemplate="%{text}",
        textfont={"size":10}
    )

    # 5. Save the plot
    output_filename = f"figure_8_heatmap_{target_word}_corrected.jpg"
    try:
        pio.write_image(fig, output_filename, width=1000, height=len(y_axis_words)*35 + 150, scale=2)
        print(f"Heatmap saved as '{output_filename}'")
    except ValueError as e:
        print(f"Error saving image: {e}")
        print("Please ensure 'kaleido' is installed (`pip install kaleido`)")

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    plot_collocate_heatmap(TARGET_WORD, DATA_FOLDER, ERA_FILES, WINDOW_SIZE, TOP_N_PER_ERA)