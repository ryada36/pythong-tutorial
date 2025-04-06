from bs4 import BeautifulSoup
import requests
import csv
import os
import time

# Get the directory where current script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

quotes_file_location = os.path.join(BASE_DIR, "quotes.csv")

def get_html(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None
    
def quotes():
    url = 'http://quotes.toscrape.com/'  # Replace with the target URL
    page = 0
    with open(quotes_file_location, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Quote", "Author", "Tags"])
        html_content = get_html(url)
        while html_content and page < 4:
            soup = BeautifulSoup(html_content, 'html.parser')
            quotes = soup.find_all("div", class_="quote")
            for quote in quotes:
                text = quote.find("span", class_="text").get_text(strip=True)
                author = quote.find("small", class_="author").get_text(strip=True)
                tags = [tag.get_text(strip=True) for tag in quote.find_all("a", class_="tag")]
                writer.writerow([text, author, ", ".join(tags)])
            next_button = soup.find("li", class_="next")
            if next_button:
                next_url = next_button.a["href"]
                url = f"http://quotes.toscrape.com{next_url}"
                time.sleep(2)
                html_content = get_html(url)
                page += 1
                print(f"Page {page} scraped.")
            else:
                html_content = None


if __name__ == "__main__":
    quotes()

