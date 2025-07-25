import os
from gensim.models import Word2Vec, FastText

DATA_DIR = "../../data_cleaned"   # or adjust path as needed
OUT_DIR = OUT_DIR = "../../models"   # embeddings will be saved here
EPOCHS = 10                       # can increase for better quality
VECTOR_SIZE = 300                 # embedding dimension
WINDOW = 5                        # context window size
MIN_COUNT = 10                    # ignore words with <10 occurrences
SG = 1                            # 1 = skipgram, 0 = CBOW

def read_sentences(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = f.read()
    # Split on <s> or <s/> (handles both)
    raw_sentences = [s.strip() for s in data.replace("<s/>", "<s>").split("<s>") if s.strip()]
    # Each sentence is space-separated tokens
    sentences = [sent.split() for sent in raw_sentences if sent]
    return sentences

def train_and_save(sentences, out_path, method="w2v"):
    print(f"Training {method.upper()} on {out_path} ...")
    if method == "w2v":
        model = Word2Vec(
            sentences,
            vector_size=VECTOR_SIZE,
            window=WINDOW,
            min_count=MIN_COUNT,
            sg=SG,
            epochs=EPOCHS,
            workers=os.cpu_count()
        )
    elif method == "fasttext":
        model = FastText(
            sentences,
            vector_size=VECTOR_SIZE,
            window=WINDOW,
            min_count=MIN_COUNT,
            sg=SG,
            epochs=EPOCHS,
            workers=os.cpu_count()
        )
    else:
        raise ValueError("method must be 'w2v' or 'fasttext'")
    model.save(out_path)
    print(f"Saved model to {out_path}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # List all txt files for era
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".txt"):
            era = fname.replace(".txt", "")
            in_path = os.path.join(DATA_DIR, fname)
            sentences = read_sentences(in_path)
            # Save both Word2Vec and fastText models for each era
            out_path_w2v = os.path.join(OUT_DIR, f"{era}_w2v.model")
            out_path_ft = os.path.join(OUT_DIR, f"{era}_fasttext.model")
            train_and_save(sentences, out_path_w2v, method="w2v")
            train_and_save(sentences, out_path_ft, method="fasttext")

if __name__ == "__main__":
    main()
