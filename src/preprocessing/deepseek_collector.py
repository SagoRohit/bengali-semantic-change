import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
import csv
import os

# Configuration
BASE_URL = "https://bangla.bdnews24.com/archive"
OUTPUT_FOLDER = "bdnews24_articles"
CSV_FILE = "bdnews24_articles.csv"
START_DATE = "2011-01-01"
END_DATE = "2011-01-31"
DELAY = 2  # seconds between requests to be polite

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def format_date_for_url(date_obj):
    """Format date object into the format needed for the URL"""
    return date_obj.strftime("%Y-%m-%d")

def get_article_links(date_obj):
    """Get all article links for a specific date"""
    formatted_date = format_date_for_url(date_obj)
    params = {
        'archive_start_date': formatted_date,
        'archive_end_date': formatted_date,
        'archive_category': 'all',
        'archive_keyword': '',
        'archive_submit': 'খুঁজুন'  # 'Search' in Bengali
    }
    
    try:
        response = requests.post(BASE_URL, data=params)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all article links - you may need to adjust this selector
        links = []
        for link in soup.select('div.article a[href^="https://bangla.bdnews24.com/"]'):
            href = link.get('href')
            if href and href not in links:
                links.append(href)
        
        return links
    
    except Exception as e:
        print(f"Error fetching links for {formatted_date}: {e}")
        return []

def scrape_article(url):
    """Scrape the text content from an article page"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title - adjust selector as needed
        title = soup.find('h1').get_text(strip=True) if soup.find('h1') else "No Title"
        
        # Extract article text - adjust selector as needed
        article_body = soup.find('div', class_='article-body')
        if article_body:
            paragraphs = [p.get_text(strip=True) for p in article_body.find_all('p')]
            content = '\n'.join(paragraphs)
        else:
            content = "Content not found"
        
        return {
            'title': title,
            'url': url,
            'content': content
        }
    
    except Exception as e:
        print(f"Error scraping article {url}: {e}")
        return None

def save_to_csv(data, date_obj):
    """Save scraped data to CSV file"""
    file_exists = os.path.isfile(CSV_FILE)
    
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['date', 'title', 'url', 'content']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for article in data:
            writer.writerow({
                'date': date_obj.strftime("%Y-%m-%d"),
                'title': article['title'],
                'url': article['url'],
                'content': article['content']
            })

def main():
    start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_date = datetime.strptime(END_DATE, "%Y-%m-%d")
    
    current_date = start_date
    while current_date <= end_date:
        print(f"Processing date: {current_date.strftime('%Y-%m-%d')}")
        
        # Get all article links for the current date
        article_links = get_article_links(current_date)
        print(f"Found {len(article_links)} articles")
        
        # Scrape each article
        articles_data = []
        for link in article_links:
            print(f"Scraping: {link}")
            article_data = scrape_article(link)
            if article_data:
                articles_data.append(article_data)
            time.sleep(DELAY)  # Be polite
        
        # Save to CSV
        if articles_data:
            save_to_csv(articles_data, current_date)
            print(f"Saved {len(articles_data)} articles to CSV")
        
        # Move to next day
        current_date += timedelta(days=1)
        time.sleep(DELAY)  # Be polite between date changes
    
    print("Scraping completed!")

if __name__ == "__main__":
    main()