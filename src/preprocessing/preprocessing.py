import os
import re
import json
from bangla_stemmer.stemmer import stemmer

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

# --- 3. Heuristic Filters & Stop Words ---
BENGALI_VOWELS = set('ািীুূেৈোৌঅআইঈউঊএঐওঔঋৠঌৡ')
BANGLA_DIGITS = '০১২৩৪৫৬৭৮৯'

# Comprehensive Bengali stop words list
BENGALI_STOP_WORDS = {
    'অতএব', 'অথচ', 'অথবা', 'অনুযায়ী', 'অনেক', 'অনেকে', 'অন্তত', 'অন্য', 'অবধি', 'অবশ্য', 'অর্থাৎ',
    'আই', 'আছে', 'আজ', 'আপনার', 'আপনি', 'আবার', 'আমরা', 'আমাকে', 'আমাদের', 'আমার', 'আমি',
    'আর', 'আরও', 'ই', 'ইত্যাদি', 'এই', 'এ', 'এঁদের', 'এঁরা', 'এটি', 'এটা', 'এটা', 'এদিকে', 'এবং', 'এবার',
    'এমন', 'এমনকী', 'এরা', 'এল', 'এস', 'এসে', 'ঐ', 'ও', 'ওই', 'ওদের', 'ওর', 'কখনও', 'কত', 'কবে',
    'কমনে', 'কয়েক', 'করবে', 'করবেন', 'করল', 'করলেন', 'করা', 'করানো', 'করায়', 'করার', 'করি', 'করেছে',
    'করেন', 'করে', 'করেই', 'করেনি', 'কিছু', 'কিছুই', 'কিন্তু', 'কী', 'কেউ', 'কে', 'কেমন', 'কোথা', 'কোন',
    'কোনও', 'কি', 'কিভাবে', 'কিছুক্ষণ', 'কেন', 'কেউই', 'গিয়ে', 'গিয়েছে', 'গেল', 'গেলে', 'চলে', 'ছাড়া',
    'ছিল', 'ছিলাম', 'ছিলেন', 'ছিলো', 'জানেন', 'জন্য', 'জানে', 'জানানো', 'জানায়', 'জানিয়ে', 'টা',
    'টি', 'তখন', 'তত', 'তবে', 'তুমি', 'তুলে', 'তেমন', 'তো', 'তোমার', 'তাদের', 'তাহলে', 'তা', 'তাঁদের',
    'তাঁরা', 'তাঁর', 'তাঁরা', 'তাঁকে', 'তাঁরাও', 'তাদেরকে', 'তারা', 'তার', 'তারা', 'তাকে', 'তাহা', 'তিন',
    'তিনি', 'তিনিও', 'তুমি', 'তোমরা', 'থাকবে', 'থাকবেন', 'থাকা', 'থাকায়', 'থাকে', 'থেকেও', 'থেকে',
    'দিকে', 'দিয়ে', 'দিলেন', 'দিতে', 'দিয়ে', 'দুজন', 'দুটি', 'দুটো', 'দেওয়া', 'দেওয়া হয়', 'দেন', 'দেননি',
    'ধরে', 'ধরে', 'নই', 'নয়', 'না', 'নিতে', 'নিজে', 'নিজেই', 'নিজের', 'নিজেদের', 'নেওয়া', 'নেওয়া হয়নি',
    'নেওয়ার', 'নিয়েই', 'নিয়েছে', 'নিয়েছেন', 'নিয়ে', 'নেই', 'পক্ষে', 'পর', 'পরেও', 'পর্যন্ত', 'পরে', 'পাওয়া',
    'পারে', 'পারেন', 'পারি', 'পি', 'পেয়েছি', 'প্রতি', 'প্রভৃতি', 'ফিরে', 'বদলে', 'বরং', 'বললেন', 'বললেন না',
    'বললাম', 'বলতে', 'বলল', 'বললেন', 'বলছি', 'বলেছেন', 'বলে', 'বলেন', 'বলা', 'বলে', 'বসে', 'বা', 'বিনা',
    'বিশেষ', 'ব্যবহার', 'ব্যবহার করে', 'ভাল', 'ভাবেই', 'মতো', 'মধ্যেও', 'মধ্যেই', 'মাধ্যমে', 'মোট', 'যখন',
    'যত', 'যথেষ্ট', 'যদি', 'যাবে', 'যাওয়া', 'যাচ্ছে', 'যাতে', 'যার', 'যারা', 'যিনি', 'যে', 'যেকোনো', 'যেখানে',
    'যেমন', 'যেতে', 'যেন', 'যেহেতু', 'যেয়ে', 'র', 'রকম', 'রেখে', 'রয়েছে', 'শুধু', 'সঙ্গে', 'সব', 'সবার',
    'সবাই', 'সমস্ত', 'সম্পর্কে', 'সহ', 'সবারই', 'সামনে', 'সি', 'সে', 'সেই', 'সেখান', 'সেখানে', 'স্পষ্ট',
    'হইতে', 'হইবে', 'হইয়াছে', 'হইয়েছে', 'হওয়া', 'হওয়াতে', 'হওয়ার', 'হওয়ায়', 'হচ্ছে', 'হত', 'হতে',
    'হবেন', 'হয়ে', 'হয়', 'হয়তো', 'হয়নি', 'হয়েছে', 'হয়েছে', 'হল', 'হলে', 'হলো', 'হিসেবে', 'এক', 'থে', 'আমা',
    'যা', 'হা', 'তাঁ', 'করেছেন', 'ঢাকা', 'বাংলাদেশ', 'ভারত', 'হয়েছে', 'জানা','গেছে', 'দেওয়া','কাজ','জানান', 'হবে', 'দি','রয়েছে',
    'শুরু','প্রা','দেখ','নি','একটু','মধ্যে','কথা','হঠাৎ','তোমা','উঠল','যায়','আমর','ওদ','দেখি','তে','উপর','নাম','এর','শুনে','কিন্ত','খুব',
    'বেশ','থা','জন্যে','এখন','গলায়','ঠিক','কোনো','বাড়ি','একজন','হাত','তাকিয়ে','পার','আচ্ছা','দিন','করছে','খালা','জানি',
    'মুখ','কোথায়','আছেন','তাই','বুঝ','শেষ','উঠে','চেষ্','ফেলে','দেব','এখনো','খেয়ে','যাব','রা','দু',
    "অতএব","অথচ","অথবা","অনুযায়ী","অনেক","অনেকে","অনেকেই","অন্তত","অন্য","অবধি","অবশ্য","অর্থাত","আই","আগামী","আগে","আগেই",
    "আছে","আজ","আদ্যভাগে","আপনার","আপনি","আবার","আমরা","আমাকে","আমাদের","আমার","আমি","আর","আরও","ই","ইত্যাদি","ইহা","উচিত",
    "উত্তর","উনি","উপর","উপরে","এ","এঁদের","এঁরা","এই","একই","একটি","একবার","একে","এক্","এখন","এখনও","এখানে","এখানেই","এটা","এটাই",
    "এটি","এত","এতটাই","এতে","এদের","এব","এবং","এবার","এমন","এমনকী","এমনি","এর","এরা","এল","এস","এসে","ঐ","ও","ওঁদের","ওঁর","ওঁরা",
    "ওই","ওকে","ওখানে","ওদের","ওর","ওরা","কখনও","কত","কবে","কমনে","কয়েক","কয়েকটি","করছে","করছেন","করতে","করবে","করবেন","করলে",
    "করলেন","করা","করাই","করায়","করার","করি","করিতে","করিয়া","করিয়ে","করে","করেই","করেছিলেন","করেছে","করেছেন","করেন","কাউকে","কাছ","কাছে",
    "কাজ","কাজে","কারও","কারণ","কি","কিংবা","কিছু","কিছুই","কিন্তু","কী","কে","কেউ","কেউই","কেখা","কেন","কোটি","কোন","কোনও","কোনো","ক্ষেত্রে","কয়েক",
    "খুব","গিয়ে","গিয়েছে","গিয়ে","গুলি","গেছে","গেল","গেলে","গোটা","চলে","চান","চায়","চার","চালু","চেয়ে","চেষ্টা","ছাড়া","ছাড়াও","ছিল","ছিলেন","জন","জনকে",
    "জনের","জন্য","জন্যওজে","জানতে","জানা","জানানো","জানায়","জানিয়ে","জানিয়েছে","জে","জ্নজন","টি","ঠিক","তখন","তত","তথা","তবু","তবে","তা","তাঁকে",
    "তাঁদের","তাঁর","তাঁরা","তাঁাহারা","তাই","তাও","তাকে","তাতে","তাদের","তার","তারপর","তারা","তারৈ","তাহলে","তাহা","তাহাতে","তাহার","তিনঐ","তিনি","তিনিও","তুমি",
    "তুলে","তেমন","তো","তোমার","থাকবে","থাকবেন","থাকা","থাকায়","থাকে","থাকেন","থেকে","থেকেই","থেকেও","দিকে","দিতে","দিন","দিয়ে","দিয়েছে","দিয়েছেন","দিলেন",
    "দু","দুই","দুটি","দুটো","দেওয়া","দেওয়ার","দেওয়া","দেখতে","দেখা","দেখে","দেন","দেয়","দ্বারা","ধরা","ধরে","ধামার","নতুন","নয়","না","নাই","নাকি","নাগাদ","নানা","নিজে",
    "নিজেই","নিজেদের","নিজের","নিতে","নিয়ে","নিয়ে","নেই","নেওয়া","নেওয়ার","নেওয়া","নয়","পক্ষে","পর","পরে","পরেই","পরেও","পর্যন্ত","পাওয়া","পাচ","পারি","পারে","পারেন",
    "পি","পেয়ে","পেয়্র্","প্রতি","প্রথম","প্রভৃতি","প্রযন্ত","প্রাথমিক","প্রায়","প্রায়","ফলে","ফিরে","ফের","বক্তব্য","বদলে","বন","বরং","বলতে","বলল","বললেন","বলা","বলে","বলেছেন",
    "বলেন","বসে","বহু","বা","বাদে","বার","বি","বিনা","বিভিন্ন","বিশেষ","বিষয়টি","বেশ","বেশি","ব্যবহার","ব্যাপারে","ভাবে","ভাবেই","মতো","মতোই","মধ্যভাগে","মধ্যে","মধ্যেই","মধ্যেও",
    "মনে","মাত্র","মাধ্যমে","মোট","মোটেই","যখন","যত","যতটা","যথেষ্ট","যদি","যদিও","যা","যাঁর","যাঁরা","যাওয়া","যাওয়ার","যাওয়া","যাকে","যাচ্ছে","যাতে","যাদের","যান","যাবে","যায়",
    "যার","যারা","যিনি","যে","যেখানে","যেতে","যেন","যেমন","র","রকম","রয়েছে","রাখা","রেখে","লক্ষ","শুধু","শুরু","সঙ্গে","সঙ্গেও","সব","সবার","সমস্ত","সম্প্রতি","সহ","সহিত","সাধারণ",
    "সামনে","সি","সুতরাং","সে","সেই","সেখান","সেখানে","সেটা","সেটাই","সেটাও","সেটি","স্পষ্ট","স্বয়ং","হইতে","হইবে","হইয়া","হওয়া","হওয়ায়","হওয়ার","হচ্ছে","হত","হতে","হতেই","হন","হবে",
    "হবেন","হয়","হয়তো","হয়নি","হয়ে","হয়েই","হয়েছিল","হয়েছে","হয়েছেন","হল","হলে","হলেই","হলেও","হলো","হাজার","হিসাবে","হৈলে","হোক","হয়","একটা",
    "একট","হয়ে","আছে৷","তুই","না৷","যাই","জি","রূপা","ফুপা","সাহেব","হিমু","আপনাকে","বের","মানুষ","দিয়ে","যায়",'নেয়া','ছাড়া','দায়','দিয়','আব্দুল','হয়নি',
    'মো','অবস্থায়','আহমেদ','দিয়েছেন','দ্বিতীয়','মিয়া','দেয়া','ইত্তেফাকনূহু','দেওয়','যাওয়','বাড়','ছাড়','হওয়া','রায়','মাহমুদ','ইত্তেফাকইউবি','ইত্তেফাকএসআর',
    'নুসরাত','রয়েছেন','খালেদা','পেয়েছেন','ইত্তেফাকঅনি','ইত্তেফাককেক','প্রতিবেদনে','ইত্তেফাকএএম','চায়','ইত্তেফাকটিএস','সেলিম','নিয়ন্ত্রণে','রহমান','মোহাম্মদ',
    'হওয়','জানিয়েছেন','পড়ুনঃ','সাড়ে','এগা','বিয়ে','আবুল','হওয়ায়','থাকায়','যায়নি','এসময়','নেওয়','ইত্তেফাককেআই','সিলেট','বিয়','বিষয়','এছাড়া',
    'ঘটনায়','হয়েছেন','শুক্রব','পড়ে','পাওয়া','বিএনপির','বিজেপি','বিয়','নিয়','ওবায়দুল','মাশরাফি','বৃহস্পতিব','মোদি','ইকবাল','মিয়','পাওয়','উপজেল','উপজেলা',
    'আওয়ামী','বাড়ি','এলাকায়','শুক্রব','জানায়','ওসি','হয়েছিল','উল্লেখ্য','হাসিন','ফায়া','চালায়','গ্রেফত','র্যাব','হোসেনের','এসআ','মির্জা','রুহুল','মুজিবুর',
    'জিয়','ইত্তেফাকএমআই','শীষ','খাতুন','শামীম',''

}

# Initialize Bengali stemmer
bengali_stemmer = stemmer.BanglaStemmer()

# Pre-stem the stop words for consistent filtering
stemmed_stop_words = set()
for stop_word in BENGALI_STOP_WORDS:
    try:
        stemmed_stop_word = bengali_stemmer.stem(stop_word)
        if stemmed_stop_word:
            stemmed_stop_words.add(stemmed_stop_word)
    except:
        stemmed_stop_words.add(stop_word)

def is_valid_token(token):
    """
    Applies a set of rules to determine if a token is a valid word.
    """
    # Basic length checks
    if len(token) <= 1: return False
    if len(token) > 12: return False  # Removes overly long tokens
    
    # Pattern checks
    if all(c == token[0] for c in token): return False  # e.g., রররর
    if all(c in BANGLA_DIGITS for c in token): return False # Removes number-only tokens
    if not any(c in BENGALI_VOWELS for c in token): return False  # Assumes words need a vowel
    
    # Stop word check
    if token in stemmed_stop_words: return False
    
    return True

def stem_token(token):
    """
    Applies stemming to a token with error handling.
    Returns the stemmed token or original if stemming fails.
    """
    try:
        stemmed = bengali_stemmer.stem(token)
        if stemmed and stemmed.strip():  # Only return non-empty stems
            return stemmed
        else:
            return token  # Return original if stemming returns empty
    except Exception as e:
        # If stemming fails, return the original token
        return token

# --- 4. Main File Processing Logic ---
def process_file(file_path, filename):
    """
    Reads an entire file, cleans it, tokenizes it into sentences,
    applies stemming, removes stop words, and saves the processed text.
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

    # STEP 4: Process each sentence individually with stemming and stop word removal
    processed_sentences = []
    cleaned_words_count = 0
    stemmed_words_count = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        words = sentence.split()
        
        # Apply stemming to each word
        stemmed_words = []
        for word in words:
            stemmed_word = stem_token(word)
            stemmed_words.append(stemmed_word)
            if stemmed_word != word:
                stemmed_words_count += 1
        
        # Filter out invalid tokens and stop words
        filtered_tokens = [w for w in stemmed_words if is_valid_token(w)]
        
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
        'stemmed_words_count': stemmed_words_count,
        'num_lines': len(processed_sentences), # This is now an accurate sentence count.
        'stemming_ratio': stemmed_words_count / len(words) if words else 0
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
            print(f" -> Completed. {stats['cleaned_word_count']:,} valid tokens ({stats['stemmed_words_count']:,} stemmed)")

    # Write all metadata to a single JSON file.
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    # Calculate overall statistics
    total_original = sum(item['original_word_count'] for item in metadata)
    total_cleaned = sum(item['cleaned_word_count'] for item in metadata)
    total_stemmed = sum(item['stemmed_words_count'] for item in metadata)
    
    print(f"\nPreprocessing complete!")
    print(f"Files processed: {len(metadata)}")
    print(f"Original words: {total_original:,}")
    print(f"Cleaned words: {total_cleaned:,} ({total_cleaned/total_original*100:.1f}% retained)")
    print(f"Words stemmed: {total_stemmed:,} ({total_stemmed/total_original*100:.1f}% of original)")
    print(f"Metadata saved to '{METADATA_FILE}'")

if __name__ == "__main__":
    main()