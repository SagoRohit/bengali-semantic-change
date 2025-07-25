import os
import re
import json
 # Note the class name change
RAW_DATA_DIR = os.path.join("..", "..", "data_raw")
CLEANED_DATA_DIR = os.path.join("..", "..", "data_cleaned")
METADATA_FILE = os.path.join("..", "..", "document_metadata.json")

# Ensure cleaned data folder exists
os.makedirs(CLEANED_DATA_DIR, exist_ok=True)

# Heuristic filters
BENGALI_VOWELS = set('ািীুূেৈোৌঅআইঈউঊএঐওঔঋৠঌৡ')
BANGLA_DIGITS = '০১২৩৪৫৬৭৮৯'

def is_valid_token(token):
    if len(token) <= 1: return False
    if len(token) > 12: return False
    if all(c == token[0] for c in token): return False  # e.g. রররর
    if all(c in BANGLA_DIGITS for c in token): return False
    if not any(c in BENGALI_VOWELS for c in token): return False  # Not pronounceable
    return True

# Aggressive Bangla stemmer

# Clean and tokenize a line
def clean_line(line):
    line = line.strip()

    # Remove English letters and digits
    line = re.sub(r'[A-Za-z0-9]', '', line)

    # Remove English punctuation and symbols (preserve Bangla dari)
    line = re.sub(r'[“”‘’\"\'.,!?()\[\]{};:@#$%^&*_+=~<>\\|/—–\-]', '', line)

    # Normalize whitespace
    line = re.sub(r'\s+', ' ', line)

    # Tokenize and preserve sentence boundary with <s>
    tokens = []
    for part in line.split('।'):
        part = part.strip()
        if part:
            words = part.split()
            filtered = [w for w in words if is_valid_token(w)]
            tokens.extend(filtered)
            tokens.append('<s>')  # mark sentence boundary

    if tokens and tokens[-1] == '<s>':
        tokens = tokens[:-1]

    return tokens

# Process each file
def process_file(file_path, filename):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    original_words = 0
    cleaned_words = 0

    for line in lines:
        original_words += len(line.strip().split())
        cleaned = clean_line(line)
        cleaned_words += len(cleaned)
        cleaned_lines.append(' '.join(cleaned))

    output_path = os.path.join(CLEANED_DATA_DIR, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in cleaned_lines:
            f.write(line + '\n')

    return {
        'filename': filename,
        'original_word_count': original_words,
        'cleaned_word_count': cleaned_words,
        'num_lines': len(cleaned_lines)
    }

# Main execution
def main():
    metadata = []

    for fname in os.listdir(RAW_DATA_DIR):
        if fname.endswith(".txt"):
            full_path = os.path.join(RAW_DATA_DIR, fname)
            print(f"Processing: {fname}")
            stats = process_file(full_path, fname)
            metadata.append(stats)

    # Write metadata to JSON
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print("Preprocessing completed. Metadata saved to JSON.")

if __name__ == "__main__":
    main()
