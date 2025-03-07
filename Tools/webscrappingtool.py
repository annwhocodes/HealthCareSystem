# Import necessary libraries
from google import genai
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import requests
import os
from dotenv import load_dotenv
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


ALLOWED_SITES = [
    "medlineplus.gov", "mayoclinic.org", "fda.gov", "drugs.com", "webmd.com",
    "who.int", "ema.europa.eu", "nih.gov"
]

# Function to query Gemini's LLM
def query_gemini(question):
    try:
        # Send the question to Gemini's LLM
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=question
        )
        
        if response.text:
            return response.text  # Return answer if found
        else:
            return None  # No answer found, trigger scraping
    except Exception as e:
        print(f"Error querying Gemini: {e}")
        return None

# Function to search using DuckDuckGo
def search_duckduckgo(query):
    results = DDGS(query)
    links = [result['href'] for result in results]
    return links

# Function to scrape websites from allowed sites list
def scrape_website(url):
    # Only scrape the allowed sites
    if any(site in url for site in ALLOWED_SITES):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Extract the content based on site structure (this can vary)
                content = soup.get_text()  # For now, we get all the text, but you can refine this
                return content
        except Exception as e:
            print(f"Error scraping {url}: {e}")
    return None

# Main function to serve as AI agent
def ai_agent(question):
    # First, try to get the answer from Gemini's LLM
    answer = query_gemini(question)
    
    if answer:
        return answer
    else:
        # If no answer, perform web scraping using DuckDuckGo
        links = search_duckduckgo(question + " drug information")
        
        # Iterate through the links and scrape content from allowed sites
        for link in links:
            content = scrape_website(link)
            if content:
                return content  # Return the first valid content found
        return "Sorry, I couldn't find relevant information."

# Example usage:
if __name__ == "__main__":
    question = input("Please enter your question: ")  # Taking input from the user
    answer = ai_agent(question)  
    print(f"Answer: {answer}")  