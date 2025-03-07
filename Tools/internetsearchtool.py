import requests
from bs4 import BeautifulSoup
from google import genai
from duckduckgo_search import DDGS
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Gemini API client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# List of medical websites to search
SITES = [
    "https://medlineplus.gov",
    "https://www.mayoclinic.org",
    "https://www.fda.gov",
    "https://www.drugs.com",
    "https://www.webmd.com",
    "https://www.who.int",
    "https://www.ema.europa.eu",
    "https://www.nih.gov"
]

def search_articles(query):
    search_results = []
    with DDGS() as ddgs:
        for site in SITES:
            results = ddgs.text(f"site:{site} {query}", max_results=5)
            search_results.extend([result["href"] for result in results])
    return search_results

def filter_and_summarize_articles(articles):
    prompt = f"Filter and summarize the top three most reliable medical articles from this list, and provide a 5-line summary with their URLs:\n{articles}"
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=prompt
    )
    return response.text

def find_medicine_images(medicine_name):
    with DDGS() as ddgs:
        results = list(ddgs.images(medicine_name, max_results=1))
    return [results[0]["image"]] if results else []

def find_purchase_links(medicine_name):
    purchase_links = []
    with DDGS() as ddgs:
        results = ddgs.text(f"buy {medicine_name}", max_results=3)
        purchase_links = [result["href"] for result in results]
    return purchase_links

if __name__ == "__main__":
    query = input("Enter medical query: ")
    articles = search_articles(query)
    filtered_summary = filter_and_summarize_articles(articles)
    print("\n=========================")
    print(" Top 3 Articles with Summaries ")
    print("=========================")
    print(filtered_summary)
    
    medicine_name = input("\nEnter medicine name: ")
    image = find_medicine_images(medicine_name)
    purchase_links = find_purchase_links(medicine_name)
    print("\n=========================")
    print(" Medicine Image ")
    print("=========================")
    print(image[0] if image else "No image found.")
    
    print("\n=========================")
    print(" Purchase Links ")
    print("=========================")
    for link in purchase_links:
        print(link)