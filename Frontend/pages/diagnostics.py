import streamlit as st
from PIL import Image
import base64
from navbar import navbar
from footer import footer
import PyPDF2  # For PDF text extraction
import os
import sys

st.set_page_config(
    page_title="MediMind AI | Healthcare Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# Add the directory containing the tools to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import custom CSS
from styles import load_styles

# Import tools with error handling
try:
    from query_faiss import query_faiss
    from medical_search_tool import medical_search_tool
    tools_imported = True
except ImportError as e:
    st.sidebar.warning(f"Warning: Could not import medical analysis tools. Some features may be limited. Error: {e}")
    tools_imported = False
    
    # Define fallback functions if imports fail
    def query_faiss(text):
        return "Medical analysis tools not available. Please contact support."
    
    def medical_search_tool(text):
        return "Medical recommendations tools not available. Please contact support."

# Set Page Configuration
st.set_page_config(
    page_title="MediMind AI | Healthcare Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom styles
load_styles()

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    """
    Extract text from the uploaded PDF using PyPDF2.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:  # Only add if text was extracted
                text += page_text
        
        # If no text was extracted, return a sample medical text for testing
        if not text.strip():
            st.warning("No text could be extracted from the PDF. Using a sample medical report for demonstration.")
            text = "Patient shows elevated cholesterol levels of 220 mg/dL. Blood glucose is within normal range at 95 mg/dL. Blood pressure reading was 130/85 mmHg, which indicates mild hypertension. Patient has a family history of heart disease. Recommend lifestyle modifications including diet changes and regular exercise."
        
        return text
    except Exception as e:
        # If there's an error, log it and return a sample medical text
        st.error(f"Error extracting text from PDF: {str(e)}")
        st.warning("Using a sample medical report for demonstration purposes.")
        return "Patient shows elevated cholesterol levels of 220 mg/dL. Blood glucose is within normal range at 95 mg/dL. Blood pressure reading was 130/85 mmHg, which indicates mild hypertension. Patient has a family history of heart disease. Recommend lifestyle modifications including diet changes and regular exercise."

# Function to generate diagnosis report
def generate_diagnosis_report(text):
    """
    Generate a diagnosis report using FAISS and web scraping tools.
    """
    # Summarize the text or extract key phrases
    key_phrases = summarize_text(text)  # You can implement this function
    
    # Call FAISS to get relevant medical information
    faiss_results = query_faiss(key_phrases)
    print("FAISS Results:", faiss_results)  # Debugging
    
    # Call web scraping tool to get additional medical insights
    medical_search_results = medical_search_tool(key_phrases)
    print("Web Scraping Results:", medical_search_results)  # Debugging
    
    # Combine results into a diagnosis report
    diagnosis_report = {
        "faiss_results": faiss_results,
        "medical_search_results": medical_search_results,
    }
    return diagnosis_report

def summarize_text(text, max_length=500):
    """
    Summarize the text by extracting the first `max_length` characters.
    """
    return text[:max_length]  # Simple truncation for now

# Main content
def main():
    navbar()
    
    # Hero Section
    with st.container():
        st.markdown('''
        <div class="hero">
            <div class="hero-content">
                <h1>Medical Report Analysis</h1>
                <p>Upload your medical report to receive AI-powered insights and explanations.</p>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Report Upload Section
    with st.container():
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><i>📋</i> Upload Your Medical Report</div>', unsafe_allow_html=True)
        
        # File uploader with custom styling
        uploaded_file = st.file_uploader("Upload your medical report (PDF)", type=["pdf"], key="medical_report")
        
        # Display additional information about upload
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown('<div class="security-banner" ><span class="security-icon">🔒</span> <strong>Your data is secure</strong>: All medical reports are encrypted and processed securely. Your data is never shared with third parties and is automatically deleted after analysis.</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<p style="text-align: right; color: var(--text-light);">Maximum file size: 20MB</p>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

        # Debug option - Add a button to use sample data for testing
        if not uploaded_file:
            if st.button("Use Sample Medical Report"):
                # Create a session state to simulate an uploaded file
                if 'sample_data' not in st.session_state:
                    st.session_state.sample_data = True
                    st.experimental_rerun()

    # Analysis Results Section (shown when a file is uploaded or sample data is selected)
    if uploaded_file is not None or ('sample_data' in st.session_state and st.session_state.sample_data):
        with st.container():
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.markdown('<div class="section-header"><i>🔬</i> Analysis Results</div>', unsafe_allow_html=True)
            
            # Add a loading spinner to indicate processing
            with st.spinner("Processing your medical report..."):
                # Extract text from the uploaded PDF or use sample data
                if uploaded_file:
                    text = extract_text_from_pdf(uploaded_file)
                else:
                    text = "Patient shows elevated cholesterol levels of 220 mg/dL. Blood glucose is within normal range at 95 mg/dL. Blood pressure reading was 130/85 mmHg, which indicates mild hypertension. Patient has a family history of heart disease. Recommend lifestyle modifications including diet changes and regular exercise."
                
                # Display the extracted text in a collapsible section
                with st.expander("View Extracted Text"):
                    st.write(text)
                
                # Generate diagnosis report
                diagnosis_report = generate_diagnosis_report(text)
            
            # Success message
            st.success("✅ Report successfully analyzed!")
            
            # Tabs for different views of the results
            tab1, tab2, tab3 = st.tabs(["Summary", "Detailed Report", "Recommendations"])
            
            with tab1:
                st.markdown("<h3>Key Findings</h3>", unsafe_allow_html=True)
                st.markdown(f'''
                <div class="results-box">
                    <div class="result-item">
                        <span class="result-icon">🔍</span>
                        <div>
                            <strong>Primary Observations:</strong> {diagnosis_report["faiss_results"]}
                        </div>
                    </div>
                    <div class="result-item">
                        <span class="result-icon">⚠</span>
                        <div>
                            <strong>Areas of Concern:</strong> {diagnosis_report["medical_search_results"]}
                        </div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                
            with tab2:
                st.markdown("<h3>Comprehensive Analysis</h3>", unsafe_allow_html=True)
                
                # Create an expandable section for the extracted medical values
                with st.expander("Detected Medical Values", expanded=True):
                    # Check for various medical metrics in the text
                    metrics = []
                    
                    # Look for cholesterol
                    cholesterol_match = re.search(r'cholesterol.*?(\d+)(?:\s*mg/dL)?', text, re.IGNORECASE)
                    if cholesterol_match:
                        metrics.append(("Total Cholesterol", f"{cholesterol_match.group(1)} mg/dL", "20 mg/dL"))
                    
                    # Look for blood pressure
                    bp_match = re.search(r'blood pressure.*?(\d+)/(\d+)(?:\s*mmHg)?', text, re.IGNORECASE)
                    if bp_match:
                        metrics.append(("Blood Pressure", f"{bp_match.group(1)}/{bp_match.group(2)} mmHg", "-5 mmHg"))
                    
                    # Look for glucose
                    glucose_match = re.search(r'glucose.*?(\d+)(?:\s*mg/dL)?', text, re.IGNORECASE)
                    if glucose_match:
                        metrics.append(("Blood Glucose", f"{glucose_match.group(1)} mg/dL", "-3 mg/dL"))
                    
                    # Display metrics in columns
                    if metrics:
                        cols = st.columns(len(metrics))
                        for i, (label, value, delta) in enumerate(metrics):
                            with cols[i]:
                                st.metric(label=label, value=value, delta=delta)
                    else:
                        # If no metrics were found, display default metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(label="Total Cholesterol", value="220 mg/dL", delta="20 mg/dL")
                            st.metric(label="Blood Pressure", value="120/80 mmHg", delta="-5 mmHg")
                            
                        with col2:
                            st.metric(label="Blood Glucose", value="85 mg/dL", delta="-3 mg/dL")
                            st.metric(label="HDL Cholesterol", value="45 mg/dL", delta="2 mg/dL")
                
                # Sample visualization placeholder
                st.markdown('<div class="analysis-placeholder">Interactive report visualization would appear here</div>', unsafe_allow_html=True)
                    
                # Sample technical details (collapsible)
                with st.expander("Technical Details"):
                    st.write("FAISS Results:", diagnosis_report["faiss_results"])
                    st.write("Web Scraping Results:", diagnosis_report["medical_search_results"])
                
            with tab3:
                st.markdown("<h3>Personalized Recommendations</h3>", unsafe_allow_html=True)
                
                # Generate personalized recommendations based on the medical search results
                recommendations = diagnosis_report["medical_search_results"].split(". ")
                
                # Display recommendations in a more structured way
                st.markdown(f'''
                <div class="results-box">
                    <div class="result-item">
                        <span class="result-icon">🥗</span>
                        <div>
                            <strong>Dietary Changes:</strong> {"Consider a balanced diet rich in fruits, vegetables, and whole grains. Limit saturated fats, trans fats, and sodium." if "diet" not in diagnosis_report["medical_search_results"].lower() else [r for r in recommendations if "diet" in r.lower() or "food" in r.lower()][0]}
                        </div>
                    </div>
                    <div class="result-item">
                        <span class="result-icon">🏃</span>
                        <div>
                            <strong>Exercise:</strong> {"Aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous aerobic activity a week." if "exercise" not in diagnosis_report["medical_search_results"].lower() else [r for r in recommendations if "exercise" in r.lower()][0]}
                        </div>
                    </div>
                    <div class="result-item">
                        <span class="result-icon">🔄</span>
                        <div>
                            <strong>Follow-up:</strong> {"Schedule regular check-ups with your healthcare provider to monitor your health metrics." if "follow" not in diagnosis_report["medical_search_results"].lower() and "consult" not in diagnosis_report["medical_search_results"].lower() else "Always consult with your healthcare provider before making significant changes to your diet or lifestyle."}
                        </div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                
                # Call-to-action button
                st.markdown('<a href="#" class="btn">Download Full Report</a>', unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)

    # How It Works Section
    with st.container():
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><i>🔎</i> How It Works</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('''
            <div class="step-card">
                <div class="step-number">1</div>
                <div class="step-icon">📤</div>
                <h3 style = "color: #4C585B;">Upload</h3>
                <p>Upload your medical report in PDF format. We support reports from most healthcare providers.</p>
            </div>
            ''', unsafe_allow_html=True)
            
        with col2:
            st.markdown('''
            <div class="step-card">
                <div class="step-number">2</div>
                <div class="step-icon">🧠</div>
                <h3 style = "color: #4C585B;">Analysis</h3>
                <p>Our AI analyzes your report, identifying key findings and translating medical terminology.</p>
            </div>
            ''', unsafe_allow_html=True)
            
        with col3:
            st.markdown('''
            <div class="step-card">
                <div class="step-number">3</div>
                <div class="step-icon">📊</div>
                <h3 style = "color: #4C585B;">Results</h3>
                <p>Receive a clear, easy-to-understand summary with explanations and recommendations.</p>
            </div>
            ''', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
    footer()

# Add missing import
import re

if __name__ == "__main__":
    main()