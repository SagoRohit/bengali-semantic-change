from pathlib import Path
import json
from collections import Counter, defaultdict

def read_tokens(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().split()

def read_top_words(file_path):
    words = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            w = line.strip().split('\t')[0]
            words.append(w)
    return words

def main():
    root = Path(__file__).resolve().parents[1]  # src/ → bengali-semantic-change/
    data_dir = root / "data_cleaned"
    results_dir = root / "results"

    results_dir.mkdir(parents=True, exist_ok=True)

    era_files = sorted(data_dir.glob("*.txt"))
    era_top_word_files = sorted(results_dir.glob("top_words_*.txt"))

    # 1. Get the union of all target words in all eras
    all_target_words = set()
    for top_words_file in era_top_word_files:
        all_target_words.update(read_top_words(top_words_file))

    # 2. For each era, count frequencies of all words
    freq_by_word = defaultdict(dict)
    for data_file in era_files:
        era = data_file.stem
        tokens = read_tokens(data_file)
        counts = Counter(tokens)
        for word in all_target_words:
            freq_by_word[word][era] = counts[word]

    # 3. Save as JSON
    output_path = results_dir / "freqs_by_era.json"
    print(f"Saving to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(freq_by_word, f, ensure_ascii=False, indent=2)
    print("Saved word frequencies per era to results/freqs_by_era.json")

if __name__ == "__main__":
    main()
