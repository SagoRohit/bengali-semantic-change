# import os
# import re
# import json
#  # Note the class name change
# RAW_DATA_DIR = os.path.join("..", "..", "data_raw")
# CLEANED_DATA_DIR = os.path.join("..", "..", "data_cleaned")
# METADATA_FILE = os.path.join("..", "..", "document_metadata.json")

# # Ensure cleaned data folder exists
# os.makedirs(CLEANED_DATA_DIR, exist_ok=True)

# # Heuristic filters
# BENGALI_VOWELS = set('ািীুূেৈোৌঅআইঈউঊএঐওঔঋৠঌৡ')
# BANGLA_DIGITS = '০১২৩৪৫৬৭৮৯'

# def is_valid_token(token):
#     if len(token) <= 1: return False
#     if len(token) > 12: return False
#     if all(c == token[0] for c in token): return False  # e.g. রররর
#     if all(c in BANGLA_DIGITS for c in token): return False
#     if not any(c in BENGALI_VOWELS for c in token): return False  # Not pronounceable
#     return True

# # Aggressive Bangla stemmer

# # Clean and tokenize a line
# # Clean and tokenize a line
# def clean_line(line):
#     line = line.strip()

#     # --- NEW NORMALIZATION STEP --- ✅
#     # This new line finds a sentence delimiter (। ? !) followed by any kind of quote
#     # and replaces the entire thing (e.g., ?”) with just the delimiter itself (e.g., ?).
#     # The \1 in the replacement refers back to the character captured in the first group ([।?!]).
#     line = re.sub(r'([।?!])["”\'‘’]+', r'\1', line)

#     # Remove English letters and digits
#     line = re.sub(r'[A-Za-z0-9]', '', line)

#     # Remove other punctuation (now that the edge case is handled)
#     # The pipe character '|' is preserved for splitting.
#     line = re.sub(r'[“”‘’\"\'.,()\[\]{};:@#$%^&*_+=~<>\\/—–\-]', '', line)

#     # Normalize whitespace
#     line = re.sub(r'\s+', ' ', line)

#     # Tokenize and preserve sentence boundary with <s>
#     tokens = []
    
#     # Split the line by dari, pipe, question mark, or exclamation mark.
#     # This now works reliably because the normalization step cleaned up the delimiters.
#     for part in re.split(r'[|।?!]', line):
#         part = part.strip()
#         if part:
#             words = part.split()
#             filtered = [w for w in words if is_valid_token(w)]
#             if filtered:
#                 tokens.extend(filtered)
#                 tokens.append('<s>')  # mark sentence boundary

#     # Remove the trailing <s> if it exists
#     if tokens and tokens[-1] == '<s>':
#         tokens = tokens[:-1]

#     return tokens

# # Process each file
# # Process each file
# def process_file(file_path, filename):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()

#     all_cleaned_content = []
#     original_words = 0
#     cleaned_words = 0

#     for line in lines:
#         original_words += len(line.strip().split())
#         cleaned = clean_line(line)
#         cleaned_words += len(cleaned)
        
#         if cleaned:
#             line_content = ' '.join(cleaned) 
#             all_cleaned_content.append(line_content)

#     final_text = ' '.join(all_cleaned_content)
    
#     # --- FIX #2 HERE ---
#     # Instead of replacing " <s> ", we now replace it with " <s>\n"
#     # This keeps the symbol and adds a newline right after it.
#     final_text_with_newlines = final_text.replace(' <s> ', ' <s>\n')

#     output_path = os.path.join(CLEANED_DATA_DIR, filename)
#     with open(output_path, 'w', encoding='utf-8') as f:
#         f.write(final_text_with_newlines)

#     num_sentences = final_text_with_newlines.count('\n') + 1

#     return {
#         'filename': filename,
#         'original_word_count': original_words,
#         'cleaned_word_count': cleaned_words,
#         'num_lines': num_sentences
#     }

# # Main execution
# def main():
#     metadata = []

#     for fname in os.listdir(RAW_DATA_DIR):
#         if fname.endswith(".txt"):
#             full_path = os.path.join(RAW_DATA_DIR, fname)
#             print(f"Processing: {fname}")
#             stats = process_file(full_path, fname)
#             metadata.append(stats)

#     # Write metadata to JSON
#     with open(METADATA_FILE, 'w', encoding='utf-8') as f:
#         json.dump(metadata, f, ensure_ascii=False, indent=4)

#     print("Preprocessing completed. Metadata saved to JSON.")

# if __name__ == "__main__":
#     main()
import os
import re
import json

# --- 1. Configuration ---
# Make sure these folder paths are correct relative to where you run the script.
RAW_DATA_DIR = os.path.join("..", "..", "data_raw")
CLEANED_DATA_DIR = os.path.join("..", "..", "data_cleaned")
METADATA_FILE = os.path.join("..", "..", "document_metadata.json")

# --- 2. Setup ---
# Ensure the folder for cleaned data exists before starting.
os.makedirs(CLEANED_DATA_DIR, exist_ok=True)
print(f"Raw data will be read from: '{RAW_DATA_DIR}'")
print(f"Cleaned data will be saved to: '{CLEANED_DATA_DIR}'")

# --- 3. Heuristic Filters ---
BENGALI_VOWELS = set('ািীুূেৈোৌঅআইঈউঊএঐওঔঋৠঌৡ')
BANGLA_DIGITS = '০১২৩৪৫৬৭৮৯'

def is_valid_token(token):
    """
    Applies a set of rules to determine if a token is a valid word.
    """
    # Note: This rule is aggressive and will remove common single-character
    # words like 'ও', 'এ', 'বা', 'সে'. You may want to adjust it.
    if len(token) <= 1: return False
    if len(token) > 12: return False  # Removes overly long tokens
    if all(c == token[0] for c in token): return False  # e.g., রররর
    if all(c in BANGLA_DIGITS for c in token): return False # Removes number-only tokens
    if not any(c in BENGALI_VOWELS for c in token): return False  # Assumes words need a vowel
    return True

# --- 4. Main File Processing Logic ---
def process_file(file_path, filename):
    """
    Reads an entire file, cleans it, tokenizes it into sentences,
    and saves the processed text.
    """
    # STEP 1: Read the ENTIRE file content into a single string.
    with open(file_path, 'r', encoding='utf-8') as f:
        full_text = f.read()

    # Calculate original word count from the raw text for metadata.
    original_words = len(full_text.strip().split())

    # STEP 2: Clean and normalize the ENTIRE text block at once.
    # Merge all lines into one by replacing newlines with spaces.
    cleaned_text = full_text.replace('\n', ' ')
    
    # Handle delimiter-quote edge cases first (e.g., ?").
    cleaned_text = re.sub(r'([।?!])["”\'‘’]+', r'\1', cleaned_text)
    # Remove English letters and digits.
    cleaned_text = re.sub(r'[A-Za-z0-9]', '', cleaned_text)
    # Remove other punctuation, preserving delimiters for splitting.
    cleaned_text = re.sub(r'[“”‘’\"\'.,()\[\]{};:@#$%^&*_+=~<>\\/—–\-]', '', cleaned_text)
    # Normalize all whitespace to single spaces.
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    # STEP 3: Split the clean text block into a list of sentences.
    sentences = re.split(r'[|।?!]', cleaned_text)

    # STEP 4: Process each sentence individually.
    processed_sentences = []
    cleaned_words_count = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        words = sentence.split()
        filtered_tokens = [w for w in words if is_valid_token(w)]
        
        if filtered_tokens:
            # Join the valid tokens of the sentence back into a string.
            processed_sentences.append(' '.join(filtered_tokens))
            cleaned_words_count += len(filtered_tokens)

    # STEP 5: Join the processed sentences with the " <s>\n" separator.
    # We use ' <s>\n'.join() to place the separator between sentences.
    final_output = ' <s>\n'.join(processed_sentences)

    # Write the final, correctly formatted text to the output file.
    output_path = os.path.join(CLEANED_DATA_DIR, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_output)

    # Return statistics for this file.
    return {
        'filename': filename,
        'original_word_count': original_words,
        'cleaned_word_count': cleaned_words_count,
        'num_lines': len(processed_sentences) # This is now an accurate sentence count.
    }

# --- 5. Main Execution ---
def main():
    """
    Main function to loop through raw data files, process them,
    and save the combined metadata.
    """
    metadata = []

    for fname in os.listdir(RAW_DATA_DIR):
        if fname.endswith(".txt"):
            full_path = os.path.join(RAW_DATA_DIR, fname)
            print(f"Processing: {fname}...")
            stats = process_file(full_path, fname)
            metadata.append(stats)
            print(f" -> Completed. Found {stats['cleaned_word_count']:,} valid tokens.")

    # Write all metadata to a single JSON file.
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print(f"\nPreprocessing complete. Metadata for {len(metadata)} files saved to '{METADATA_FILE}'.")

if __name__ == "__main__":
    main()