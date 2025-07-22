import math
from pathlib import Path
from collections import Counter, defaultdict
import json

WINDOW_SIZE = 5
TOP_K = 10

def read_tokens(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        # Tokenized file; split by whitespace
        return f.read().split()

def read_target_words(top_words_file):
    target_words = []
    with open(top_words_file, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().split('\t')[0]
            target_words.append(word)
    return target_words

def split_sentences(tokens):
    sentences = []
    current_sentence = []
    for token in tokens:
        if token in {"<s>", "<s/>"}:  # Handle both possible sentence markers
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
        else:
            current_sentence.append(token)
    if current_sentence:
        sentences.append(current_sentence)
    return sentences

def compute_pmi(target_word, sentences, total_counts, total_tokens, window_size=5, min_cooccur=3):
    target_occurrences = 0
    cooccur_counts = Counter()
    for sentence in sentences:
        sent_len = len(sentence)
        for idx, token in enumerate(sentence):
            if token == target_word:
                target_occurrences += 1
                left = max(0, idx - window_size)
                right = min(sent_len, idx + window_size + 1)
                window = sentence[left:idx] + sentence[idx+1:right]
                for collocate in window:
                    if collocate != target_word:
                        cooccur_counts[collocate] += 1

    pmi_scores = {}
    # Total possible windows (number of target occurrences × window size × 2, clipped by sentence boundary)
    total_target_windows = target_occurrences * window_size * 2  # This is an upper bound; exact would consider boundaries

    for collocate, cooccur in cooccur_counts.items():
        if cooccur < min_cooccur:
            continue  # Only include collocates that appear at least `min_cooccur` times
        p_collocate_given_target = cooccur / total_target_windows
        p_collocate = total_counts[collocate] / total_tokens
        if p_collocate_given_target > 0 and p_collocate > 0:
            pmi = math.log2(p_collocate_given_target / p_collocate)
            pmi_scores[collocate] = pmi

    return pmi_scores


def main():
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data_cleaned"
    results_dir = root / "results"

    for data_file in data_dir.glob("*.txt"):
        period = data_file.stem
        tokens = read_tokens(data_file)
        sentences = split_sentences(tokens)
        all_flat = [w for sent in sentences for w in sent]
        total_counts = Counter(all_flat)
        total_tokens = len(all_flat)

        top_words_file = results_dir / f"top_words_{period}.txt"
        if not top_words_file.exists():
            print(f"Skipping {period}: top_words file not found")
            continue
        target_words = read_target_words(top_words_file)

        era_collocates = {}

        for word in target_words:
            pmi_scores = compute_pmi(word, sentences, total_counts, total_tokens, window_size=WINDOW_SIZE)
            top_collocates = sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
            era_collocates[word] = [[collocate, float(f"{score:.4f}")] for collocate, score in top_collocates]

        output_file = results_dir / f"collocates_{period}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(era_collocates, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
