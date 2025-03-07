from langchain.tools import Tool
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import requests

# Allowed medical websites
ALLOWED_SITES = [
    "medlineplus.gov", "mayoclinic.org", "fda.gov", "drugs.com",
    "webmd.com", "who.int", "ema.europa.eu", "nih.gov"
]

# Function to search DuckDuckGo
def search_duckduckgo(query):
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=8)  # Get top 8 results
    links = [result['href'] for result in results if 'href' in result]
    return links

# Function to scrape websites from allowed list
def scrape_website(url):
    if any(site in url for site in ALLOWED_SITES):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                paragraphs = soup.find_all('p')  # Extract only paragraphs
                content = " ".join([p.get_text() for p in paragraphs[:5]])  # Limit to 5 paragraphs
                return content if content else None
        except Exception as e:
            print(f"Error scraping {url}: {e}")
    return None

# Function that combines search + scraping
def medical_web_search(query):
    links = search_duckduckgo(query + " site:" + " OR site:".join(ALLOWED_SITES))
    for link in links:
        content = scrape_website(link)
        if content:
            return content
    return "No reliable information found."

# Wrapping it as a LangChain tool
medical_search_tool = Tool(
    name="Medical_Web_Search",
    description="Searches trusted medical websites (NIH, Mayo Clinic, WebMD, etc.) for drug and medical information.",
    func=medical_web_search
)