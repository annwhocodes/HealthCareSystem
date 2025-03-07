import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google Custom Search API credentials
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_SE_ID")

# Function to search using Google Custom Search API
def search_google(query):
    try:
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}"
        response = requests.get(url)
        if response.status_code == 200:
            results = response.json().get("items", [])
            links = [result["link"] for result in results]
            return links
        else:
            print(f"Error searching Google: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Error searching Google: {e}")
        return []

# Example usage
if __name__ == "__main__":
    query = input("Enter your query: ")
    links = search_google(query)
    print("Search Results:")
    for link in links:
        print(link)