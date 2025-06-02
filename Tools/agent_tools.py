from langchain_community.tools import DuckDuckGoSearchRun, Tool
from Frontend.search_agent import MedicalSearchAgent
from Frontend.PDFSearchTool import PDFSearchTool
from Tools.webscrappingtool import WebScraper
from Frontend.csv_reader_tool import CSVReaderTool
from Frontend.repl_tool import PythonREPLTool

def initialize_tools(llm):
    """Initialize all tools needed for the agents"""
    
    # Initialize base tools
    web_scraper = WebScraper()
    search_tool = DuckDuckGoSearchRun()
    medical_search_agent = MedicalSearchAgent(llm)
    
    # Initialize PDF tool
    pdf_tool = PDFSearchTool(
        pdf='Data/MSR Sample Active Substance Use 2023.pdf',
        config=dict(
            llm=dict(
                provider="ollama",
                config=dict(
                    model="llama2",
                    temperature=0.5,
                    stream=True,
                ),
            ),
            embedder=dict(
                provider="google",
                config=dict(
                    model="models/embedding-001",
                    task_type="retrieval_document",
                ),
            ),
        )
    )
    
    # Initialize CSV and REPL tools
    csv_tool = CSVReaderTool('Data/hospital_records_2021_2024_with_bills.csv')
    repl_tool = PythonREPLTool()
    
    # Create tool wrappers
    tools = {
        "search": [
            Tool(
                name="web_scraper",
                func=web_scraper.scrape_medical_info,
                description="Scrapes medical information from trusted websites"
            ),
            Tool(
                name="search_tool",
                func=search_tool.run,
                description="Searches the internet for medical information"
            ),
            Tool(
                name="medical_search",
                func=medical_search_agent.process_query,
                description="Comprehensive medical information search and processing"
            )
        ],
        "diagnostic": [
            Tool(
                name="pdf_analyzer",
                func=pdf_tool.search,
                description="Analyzes medical PDFs for relevant information"
            )
        ],
        "management": [
            Tool(
                name="csv_reader",
                func=csv_tool.read_csv,
                description="Reads and processes hospital records from CSV"
            ),
            Tool(
                name="python_repl",
                func=repl_tool.run,
                description="Executes Python code for data analysis"
            )
        ]
    }
    
    return tools 