from langchain.tools import Tool
from Tools.query_faiss import query_vector_store
from duckduckgo_search import DDGS  # Ensure this import is correct
from bs4 import BeautifulSoup
import requests

# FAISS Query Tool
def faiss_query_tool(query):
    """Tool to query the FAISS index."""
    try:
        query_vector_store(query)  # Call the existing FAISS query function
        return "FAISS query executed. Check console for results."
    except Exception as e:
        return f"Error querying FAISS index: {str(e)}"

# Wrap it as a LangChain tool
faiss_tool = Tool(
    name="FAISS_Query",
    description="Queries the FAISS index for medical information.",
    func=faiss_query_tool
)

# Allowed medical websites
ALLOWED_SITES = [
    "medlineplus.gov", "mayoclinic.org", "fda.gov", "drugs.com",
    "webmd.com", "who.int", "ema.europa.eu", "nih.gov"
]

# Function to search DuckDuckGo
def search_duckduckgo(query):
    """Search DuckDuckGo for medical information."""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=8)  # Get top 8 results
        links = [result['href'] for result in results if 'href' in result]
        return links
    except Exception as e:
        print(f"Error searching DuckDuckGo: {e}")
        return []

# Function to scrape websites from allowed list
def scrape_website(url):
    """Scrape content from trusted medical websites."""
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
    """Search and scrape trusted medical websites for information."""
    try:
        links = search_duckduckgo(query + " site:" + " OR site:".join(ALLOWED_SITES))
        for link in links:
            content = scrape_website(link)
            if content:
                return content
        return "No reliable information found."
    except Exception as e:
        return f"Error during medical web search: {str(e)}"

# Wrapping it as a LangChain tool
medical_search_tool = Tool(
    name="Medical_Web_Search",
    description="Searches trusted medical websites (NIH, Mayo Clinic, WebMD, etc.) for drug and medical information.",
    func=medical_web_search
)
