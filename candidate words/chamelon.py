# chameleon_analysis.py
import os
from collections import Counter
import re

# --- 1. CONFIGURATION ---
ERA_PATHS = {
    "1950-1970": "../data_cleaned/1950_1970.txt",  # Single file
    "2010-2025": "../data_cleaned/2010_2025.txt",   # Single file
}
ERA_OLD = "1950-1970"
ERA_NEW = "2010-2025"
WINDOW_SIZE = 5  # Symmetric window of +/- 5
TOP_N_COLLOCATES = 20 # Number of collocates to consider for each word
MIN_COLLOCATES = 2 # Word must have at least this many collocates in BOTH eras
TOP_N_CANDIDATES = 200


# --- 3. DATA LOADING AND COLLOCATE EXTRACTION ---
def load_and_tokenize_corpus(era_paths):
    """Loads and tokenizes the corpus, returning a dictionary of token lists."""
    print("Loading and tokenizing corpus...")
    all_era_words = {}
    
    for era, file_path in era_paths.items():
        if not os.path.exists(file_path):
            print(f"Warning: File not found for era '{era}': {file_path}")
            continue
        
        try:
            # Read the single file directly
            with open(file_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
            
            # Tokenize the text
            tokens = re.split(r'\s+', full_text.strip())
            all_era_words[era] = tokens
            print(f"Loaded {len(tokens)} tokens from {era}")
            
        except Exception as e:
            print(f"Error reading file for era '{era}': {e}")
            continue
    
    return all_era_words

def get_collocates(target_word, era_tokens, window_size):
    """Extracts collocates for a target word, respecting '<s>' boundaries."""
    collocates = []
    indices = [i for i, w in enumerate(era_tokens) if w == target_word]
    
    for i in indices:
        # Define potential window start and end
        start = i - window_size
        end = i + window_size
        
        # Find the true start by checking for '<s>' in the left window
        true_start = start
        for j in range(i - 1, start - 1, -1):
            if j < 0 or era_tokens[j] == '<s>':
                true_start = j + 1
                break
        
        # Find the true end by checking for '<s>' in the right window
        true_end = end
        for j in range(i + 1, end + 1):
            if j >= len(era_tokens) or era_tokens[j] == '<s>':
                true_end = j - 1
                break
                
        # Add words within the true window
        for k in range(true_start, true_end + 1):
            # Exclude the target word itself
            if k != i:
                collocates.append(era_tokens[k])
                
    return Counter(collocates)

# --- 4. CHAMELEON ANALYSIS ---
def analyze_collocate_change(era_tokens, old, new):
    """Calculates Jaccard distance of collocates for each word and ranks them."""
    print(f"Analyzing collocate change between '{old}' and '{new}'...")
    
    tokens_old = era_tokens.get(old, [])
    tokens_new = era_tokens.get(new, [])
    
    vocab = set(tokens_old) | set(tokens_new)
    
    change_scores = {}
    
    for i, word in enumerate(vocab):
        if (i + 1) % 500 == 0:
            print(f"  Processing word {i+1}/{len(vocab)}: {word}")

        if len(word) <= 1:
            continue
        
        collocates_old = get_collocates(word, tokens_old, WINDOW_SIZE)
        collocates_new = get_collocates(word, tokens_new, WINDOW_SIZE)

        # Get the sets of top N collocate words
        set_old = set(c[0] for c in collocates_old.most_common(TOP_N_COLLOCATES))
        set_new = set(c[0] for c in collocates_new.most_common(TOP_N_COLLOCATES))
        
        # Filter: ensure the word has enough context in both eras
        if len(set_old) < MIN_COLLOCATES or len(set_new) < MIN_COLLOCATES:
            continue
            
        # Calculate Jaccard distance
        intersection = len(set_old.intersection(set_new))
        union = len(set_old.union(set_new))
        
        if union == 0:
            jaccard_distance = 0
        else:
            jaccard_distance = 1 - (intersection / union)
        
        change_scores[word] = jaccard_distance

    sorted_chameleons = sorted(change_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_chameleons

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    era_tokens = load_and_tokenize_corpus(ERA_PATHS)
    
    if ERA_OLD not in era_tokens or ERA_NEW not in era_tokens:
        print("Error: One or both specified eras not found. Exiting.")
    else:
        top_chameleons = analyze_collocate_change(era_tokens, ERA_OLD, ERA_NEW)
        
        # Save results to a TXT file with formatted output
        output_file = "top_chameleons_analysis.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Top {TOP_N_CANDIDATES} 'Chameleon' Words Analysis\n")
            f.write("=" * 50 + "\n\n")
            f.write("Chameleon words are those with the largest change in collocates\n")
            f.write("(measured by Jaccard distance between eras)\n\n")
            f.write(f"{'Rank':<5} {'Word':<20} {'Jaccard Distance':<18}\n")
            f.write("-" * 45 + "\n")
            
            for i, (word, distance) in enumerate(top_chameleons[:TOP_N_CANDIDATES], 1):
                f.write(f"{i:<5} {word:<20} {distance:<18.6f}\n")
            
            # Save the full list at the end of the file
            f.write(f"\n\nFull Ranked List ({len(top_chameleons)} words):\n")
            f.write("=" * 40 + "\n")
            for i, (word, distance) in enumerate(top_chameleons, 1):
                f.write(f"{i:4d}. {word:<25} {distance:.8f}\n")
        
        # Print to console
        print(f"\n--- Top {TOP_N_CANDIDATES} 'Chameleon' Words ---")
        print("(Words with largest semantic change based on collocate analysis)")
        print(f"{'Rank':<5} {'Word':<20} {'Jaccard Distance':<18}")
        print("-" * 45)
        for i, (word, distance) in enumerate(top_chameleons[:TOP_N_CANDIDATES], 1):
            print(f"{i:<5} {word:<20} {distance:<18.6f}")
        
        print(f"\nAnalysis complete. Full results saved to '{output_file}'")