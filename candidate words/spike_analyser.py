# spiker_analysis.py
import os
from collections import Counter
import re

# --- 1. CONFIGURATION ---
# Update with the correct paths to your single text files
ERA_PATHS = {
    "1950-1970": "../data_cleaned/1950_1970.txt",  # Single file
    "2010-2025": "../data_cleaned/2010_2025.txt",   # Single file
}
# Eras to compare for the analysis
ERA_OLD = "1950-1970"
ERA_NEW = "2010-2025"
TOP_N_CANDIDATES = 200 # Number of top words to list
MIN_TOTAL_FREQUENCY = 10 # Ignore words that are too rare across both eras combined



# --- 3. DATA LOADING AND PROCESSING ---
def load_and_process_corpus(era_paths):
    """Loads text files, tokenizes them, and calculates word frequencies."""
    print("Loading and processing corpus...")
    all_era_words = {}
    
    for era, file_path in era_paths.items():
        if not os.path.exists(file_path):
            print(f"Warning: File not found for era '{era}': {file_path}")
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
            
            # Normalize whitespace and split into tokens
            tokens = re.split(r'\s+', full_text.strip())
            all_era_words[era] = tokens
            print(f"Loaded {len(tokens)} words from {era}")
            
        except Exception as e:
            print(f"Error reading file for era '{era}': {e}")
            continue

    era_total_counts = {era: len(words) for era, words in all_era_words.items()}
    era_freq_counts = {era: Counter(words) for era, words in all_era_words.items()}
    
    return era_total_counts, era_freq_counts

# --- 4. SPIKER ANALYSIS ---
def analyze_frequency_change(era_total_counts, era_freq_counts, old, new):
    """Calculates a change score for each word and ranks them."""
    print(f"Analyzing frequency change between '{old}' and '{new}'...")
    
    counts_old = era_freq_counts.get(old, Counter())
    counts_new = era_freq_counts.get(new, Counter())
    total_old = era_total_counts.get(old, 1)
    total_new = era_total_counts.get(new, 1)

    # Get a combined vocabulary
    vocab = set(counts_old.keys()) | set(counts_new.keys())
    
    word_scores = {}
    for word in vocab:
        # Ignore stop words and single-character tokens
        if len(word) <= 1:
            continue

        raw_count_old = counts_old.get(word, 0)
        raw_count_new = counts_new.get(word, 0)

        # Filter out words that are too rare overall
        if raw_count_old + raw_count_new < MIN_TOTAL_FREQUENCY:
            continue
            
        # Calculate frequency per million
        freq_pm_old = (raw_count_old / total_old) * 1_000_000
        freq_pm_new = (raw_count_new / total_new) * 1_000_000

        # Calculate the spiker score
        # The +1 in denominator prevents division by zero for new words
        score = freq_pm_new / (freq_pm_old + 1)
        word_scores[word] = score

    # Sort words by score in descending order
    sorted_spikers = sorted(word_scores.items(), key=lambda item: item[1], reverse=True)
    
    return sorted_spikers

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    
    total_counts, freq_counts = load_and_process_corpus(ERA_PATHS)
    
    if ERA_OLD not in total_counts or ERA_NEW not in total_counts:
        print("Error: One or both specified eras not found in the processed data. Exiting.")
        print("Available eras:", list(total_counts.keys()))
    else:
        top_spikers = analyze_frequency_change(total_counts, freq_counts, ERA_OLD, ERA_NEW)
        
        # Save results to a TXT file with formatted output
        output_file = "top_spikers_analysis.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Top {TOP_N_CANDIDATES} 'Spiker' Words Analysis\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"{'Rank':<5} {'Word':<20} {'Spiker Score':<15}\n")
            f.write("-" * 40 + "\n")
            
            for i, (word, score) in enumerate(top_spikers[:TOP_N_CANDIDATES], 1):
                f.write(f"{i:<5} {word:<20} {score:<15.6f}\n")
        
        # print(f"\n--- Top {TOP_N_CANDIDATES} 'Spiker' Words ---")
        # print(f"{'Rank':<5} {'Word':<20} {'Spiker Score':<15}")
        # print("-" * 40)
        # for i, (word, score) in enumerate(top_spikers[:TOP_N_CANDIDATES], 1):
        #     print(f"{i:<5} {word:<20} {score:<15.6f}")
        print(f"\nAnalysis complete. Full ranked list saved to '{output_file}'")