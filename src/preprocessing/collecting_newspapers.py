import requests
from bs4 import BeautifulSoup

# Wikisource URL for the book
URL = "https://bn.wikisource.org/wiki/ভোঁদড়_বাহাদুর"

# Bengali keywords to exclude
UNWANTED_KEYWORDS = [
    "প্রথম সংস্করণ",
    "প্রকাশক",
    "মুদ্রক",
    "প্রেস",
    "কোম্পানি",
    "লেন",
    "দাম",
    "সর্বস্বত্ব",
    "বাঁধিয়েছেন",
    "প্রচ্ছদপট",
    "সংরক্ষিত",
    "মুদ্রণ",
    "মূল্য",
    "ঠিকানা"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

def is_unwanted(line):
    for word in UNWANTED_KEYWORDS:
        if word in line:
            return True
    return False

def scrape_book_text(url):
    print(f"Fetching: {url}")
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        print("Failed to fetch the page!")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    content_div = soup.find("div", {"class": "mw-parser-output"})
    if not content_div:
        print("Content not found!")
        return ""

    # Get all text, one line per block
    all_text = content_div.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in all_text.split("\n") if line.strip()]

    # 1️⃣ Filter out lines with known metadata words
    lines = [line for line in lines if not is_unwanted(line)]

    # 2️⃣ Drop lines that are "too short" (likely metadata)
    lines = [line for line in lines if len(line) > 8]

    # 3️⃣ Drop the first few lines (extra cautious)
    lines = lines[3:] if len(lines) > 5 else lines

    # Final cleaned text
    return "\n\n".join(lines)

def main():
    clean_text = scrape_book_text(URL)
    if clean_text:
        with open("bhondor_bahadur_cleaned.txt", "w", encoding="utf-8") as f:
            f.write(clean_text)
        print("✅ Cleaned text saved to bhondor_bahadur_cleaned.txt")
    else:
        print("❌ No text extracted.")

if __name__ == "__main__":
    main()
