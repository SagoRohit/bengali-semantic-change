import math
import json
from pathlib import Path
from collections import Counter, defaultdict

WINDOW_SIZE = 5
TOP_K = 10
MIN_COOCUR = 3

def read_tokens(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().split()

def split_sentences(tokens):
    sentences = []
    current = []
    for token in tokens:
        if token in {"<s>", "<s/>"}:
            if current:
                sentences.append(current)
                current = []
        else:
            current.append(token)
    if current:
        sentences.append(current)
    return sentences

def read_target_words(top_words_file):
    words = []
    with open(top_words_file, 'r', encoding='utf-8') as f:
        for line in f:
            w = line.strip().split('\t')[0]
            words.append(w)
    return words

def compute_pmi(target_word, sentences, total_counts, total_tokens, window_size=WINDOW_SIZE, min_cooccur=MIN_COOCUR):
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
    total_target_windows = target_occurrences * window_size * 2 if target_occurrences else 1

    pmi_scores = {}
    for collocate, cooccur in cooccur_counts.items():
        if cooccur < min_cooccur:
            continue
        p_collocate_given_target = cooccur / total_target_windows
        p_collocate = total_counts[collocate] / total_tokens if total_tokens > 0 else 0
        if p_collocate_given_target > 0 and p_collocate > 0:
            pmi = math.log2(p_collocate_given_target / p_collocate)
            pmi_scores[collocate] = pmi
    return pmi_scores

def main():
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data_cleaned"
    results_dir = root / "results"

    era_files = sorted(data_dir.glob("*.txt"))
    era_corpora = {}
    era_sentences = {}
    era_counts = {}
    era_tokens = {}

    # Build union of all top words across eras
    all_target_words = set()
    for data_file in era_files:
        period = data_file.stem
        top_words_file = results_dir / f"top_words_{period}.txt"
        if top_words_file.exists():
            all_target_words.update(read_target_words(top_words_file))

    # Pre-load all corpora for speed
    for data_file in era_files:
        period = data_file.stem
        tokens = read_tokens(data_file)
        sentences = split_sentences(tokens)
        all_flat = [w for sent in sentences for w in sent]
        era_sentences[period] = sentences
        era_counts[period] = Counter(all_flat)
        era_tokens[period] = len(all_flat)

    print(f"Total unique target words: {len(all_target_words)}")
    print(f"Total eras: {list(era_sentences.keys())}")

    # For each word, for each era, get top K collocates
    word_collocates_by_era = defaultdict(dict)
    for word in all_target_words:
        for period in era_sentences:
            pmi_scores = compute_pmi(
                word,
                era_sentences[period],
                era_counts[period],
                era_tokens[period],
                window_size=WINDOW_SIZE,
                min_cooccur=MIN_COOCUR
            )
            top_collocates = sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
            word_collocates_by_era[word][period] = [[coll, float(f"{score:.4f}")] for coll, score in top_collocates]

    output_file = results_dir / "semantic_neighbors_drift.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(word_collocates_by_era, f, ensure_ascii=False, indent=2)
    print(f"Saved semantic drift collocates to {output_file}")

if __name__ == "__main__":
    main()
