# Import necessary libraries
import google.generativeai as genai
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Google Gemini client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

ALLOWED_SITES = [
    "medlineplus.gov", "mayoclinic.org", "fda.gov", "drugs.com", "webmd.com",
    "who.int", "ema.europa.eu", "nih.gov"
]

class WebScraper:
    def __init__(self):
        pass

    def query_gemini(self, question):
        try:
            # Initialize the Gemini model
            model = genai.GenerativeModel("gemini-1.5-flash-latest")  # Use "gemini-pro" or the correct model name
            response = model.generate_content(question)
            
            if response and response.text:
                return response.text  # Return answer if found
            else:
                return None  # No answer found, trigger scraping
        except Exception as e:
            print(f"Error querying Gemini: {e}")
            return None

    def search_duckduckgo(self, query):
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=5)
            links = [result['href'] for result in results]
        return links

    def scrape_website(self, url):
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

    def scrape_medical_info(self, query):
        # First, try to get the answer from Gemini's LLM
        answer = self.query_gemini(query)
        
        if answer:
            return answer
        else:
            # If no answer, perform web scraping using DuckDuckGo
            links = self.search_duckduckgo(query + " drug information")
            
            # Iterate through the links and scrape content from allowed sites
            for link in links:
                content = self.scrape_website(link)
                if content:
                    return content  # Return the first valid content found
            return "Sorry, I couldn't find relevant information."

# Example usage:
if __name__ == "__main__":
    scraper = WebScraper()
    question = input("Please enter your question: ")  # Taking input from the user
    answer = scraper.scrape_medical_info(question)  
    print(f"Answer: {answer}")
