from crewai import Agent
from langchain_community.tools import DuckDuckGoSearchRun
from bs4 import BeautifulSoup
import requests
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalSearchAgent:
    """Enhanced Medical Search Agent with improved search capabilities"""
    
    ALLOWED_MEDICAL_SITES = [
        "medlineplus.gov",
        "mayoclinic.org",
        "who.int",
        "nih.gov",
        "cdc.gov",
        "webmd.com",
        "healthline.com",
        "drugs.com",
        "fda.gov"
    ]

    def __init__(self, llm):
        """Initialize the Medical Search Agent with necessary tools"""
        self.llm = llm
        self.search_tool = DuckDuckGoSearchRun()
        
    def _filter_medical_urls(self, urls: List[str]) -> List[str]:
        """Filter URLs to only include trusted medical websites"""
        return [url for url in urls if any(site in url.lower() for site in self.ALLOWED_MEDICAL_SITES)]
    
    def _scrape_content(self, url: str) -> str:
        """Safely scrape content from a medical website"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'iframe']):
                element.decompose()
            
            # Extract main content
            main_content = soup.find('main') or soup.find('article') or soup.find('div', {'class': ['content', 'main-content']})
            if main_content:
                text = main_content.get_text(separator=' ', strip=True)
            else:
                text = soup.get_text(separator=' ', strip=True)
            
            # Clean and format the text
            text = ' '.join(text.split())
            return text[:2000]  # Limit content length
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return ""

    def search_medical_info(self, query: str) -> Dict:
        """
        Search for medical information using multiple sources
        """
        try:
            # Enhance the search query for medical context
            medical_query = f"medical {query} site:({' OR site:'.join(self.ALLOWED_MEDICAL_SITES)})"
            
            # Perform the search
            search_results = self.search_tool.run(medical_query)
            
            # Extract URLs from search results
            urls = self._filter_medical_urls([url for url in search_results.split() if url.startswith('http')])
            
            if not urls:
                return {
                    "success": False,
                    "message": "No reliable medical sources found for the query.",
                    "results": []
                }
            
            # Scrape content from each URL
            results = []
            for url in urls[:3]:  # Limit to top 3 results
                content = self._scrape_content(url)
                if content:
                    results.append({
                        "source": url,
                        "content": content
                    })
            
            # Process results with LLM for better formatting
            if results:
                formatted_results = []
                for result in results:
                    summary_prompt = f"""
                    Summarize the following medical information in a clear and concise way:
                    Source: {result['source']}
                    Content: {result['content']}
                    
                    Provide a summary that:
                    1. Highlights key medical information
                    2. Uses bullet points for important facts
                    3. Maintains medical accuracy
                    4. Includes any relevant warnings or disclaimers
                    """
                    
                    summary = self.llm.invoke(summary_prompt)
                    formatted_results.append({
                        "source": result['source'],
                        "summary": summary
                    })
                
                return {
                    "success": True,
                    "message": "Successfully retrieved medical information",
                    "results": formatted_results
                }
            else:
                return {
                    "success": False,
                    "message": "Could not extract content from medical sources.",
                    "results": []
                }
                
        except Exception as e:
            logger.error(f"Error in medical search: {str(e)}")
            return {
                "success": False,
                "message": f"An error occurred during the search: {str(e)}",
                "results": []
            }

    def get_purchase_options(self, medicine_name: str) -> Dict:
        """
        Search for medicine purchase options from legitimate pharmacies
        """
        try:
            # Enhance the search query for pharmacy context
            pharmacy_query = f"buy {medicine_name} pharmacy prescription price site:drugs.com OR site:goodrx.com OR site:pharmacy.ca"
            
            # Perform the search
            search_results = self.search_tool.run(pharmacy_query)
            
            # Extract URLs from search results
            urls = [url for url in search_results.split() if url.startswith('http')][:3]
            
            if not urls:
                return {
                    "success": False,
                    "message": "No reliable pharmacy sources found.",
                    "results": []
                }
            
            results = []
            for url in urls:
                content = self._scrape_content(url)
                if content:
                    # Process pricing information with LLM
                    price_prompt = f"""
                    Extract pricing and pharmacy information from the following content:
                    {content}
                    
                    Provide:
                    1. Price range if available
                    2. Available pharmacies
                    3. Any discount programs mentioned
                    4. Required prescription information
                    """
                    
                    summary = self.llm.invoke(price_prompt)
                    results.append({
                        "source": url,
                        "pricing_info": summary
                    })
            
            return {
                "success": True,
                "message": "Successfully retrieved purchase options",
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error finding purchase options: {str(e)}")
            return {
                "success": False,
                "message": f"An error occurred while searching for purchase options: {str(e)}",
                "results": []
            }

    def process_query(self, query: str) -> Dict:
        """
        Process a user query and return comprehensive medical information
        """
        # Determine query type with LLM
        query_analysis_prompt = f"""
        Analyze this medical query and categorize it:
        Query: {query}
        
        Is this about:
        1. General medical information
        2. Medicine purchase/pricing
        3. Both
        
        Return just the number (1, 2, or 3).
        """
        
        query_type = self.llm.invoke(query_analysis_prompt).strip()
        
        results = {
            "medical_info": None,
            "purchase_options": None
        }
        
        # Get medical information
        if query_type in ["1", "3"]:
            results["medical_info"] = self.search_medical_info(query)
        
        # Get purchase options
        if query_type in ["2", "3"]:
            # Extract medicine name using LLM
            medicine_prompt = f"""
            Extract the medicine name from this query:
            {query}
            
            Return just the medicine name, nothing else.
            """
            medicine_name = self.llm.invoke(medicine_prompt).strip()
            results["purchase_options"] = self.get_purchase_options(medicine_name)
        
        return results