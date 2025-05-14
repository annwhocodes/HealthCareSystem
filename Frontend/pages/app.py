import streamlit as st
from navbar import navbar
from styles import load_styles
from footer import footer
import pandas as pd
from crewai import Agent, Task, Crew
from repl_tool import PythonREPLTool, BaseTool
from csv_reader_tool import CSVReaderTool
from visualiser_tool import VisualiserTool
import os
import tempfile

# Set Page Configuration
st.set_page_config(
    page_title="MediMind AI | Healthcare Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom styles
load_styles()

# Display navbar
navbar()

# Initialize session state variables if they don't exist
if 'hospital_data' not in st.session_state:
    st.session_state.hospital_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'csv_path' not in st.session_state:
    st.session_state.csv_path = None

# Hero Section with Modern Design
st.markdown(
    """
    <div class="hero" id="home">
        <div class="hero-content">
            <h1>
                <span style="color: #2D8CFF;">Medi</span><span style="color: #34C759;">Mind</span> AI
            </h1>
            <h2 style="margin-top: 0; color: #4A5568; font-size: 1.5rem; font-weight: 500;">Transforming Healthcare with Artificial Intelligence</h2>
            <p>Experience the future of healthcare with our AI-powered platform. Get instant insights from patient reports, manage hospital operations efficiently, and access reliable medical information with ease.</p>
            <div style="display: flex; justify-content: center; gap: 15px;">
                <a class="btn" href="#diagnostics">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;">
                        <path d="M22 12h-4l-3 9L9 3l-3 9H2"></path>
                    </svg>
                    Start Diagnosis
                </a>
                <a class="btn btn-outline" href="#operations">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;">
                        <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
                        <line x1="8" y1="21" x2="16" y2="21"></line>
                        <line x1="12" y1="17" x2="12" y2="21"></line>
                    </svg>
                    Hospital Operations
                </a>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize hospital manager and tools
@st.cache_resource
def initialize_tools(csv_path):
    # Create instances of tools
    python_repl_tool = PythonREPLTool()
    csv_reader_tool = CSVReaderTool(csv_path)
    
    class PatientRecordManagerTool(BaseTool):
        def __init__(self, csv_path):
            super().__init__(
                name="Patient Record Manager Tool",
                description="A tool to manage patient records in the CSV dataset."
            )
            self.csv_path = csv_path
        
        def execute(self, record):
            df = pd.read_csv(self.csv_path)
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
            df.to_csv(self.csv_path, index=False)
            return "Patient record added successfully."
    
    patient_record_manager_tool = PatientRecordManagerTool(csv_path)
    
    return python_repl_tool, csv_reader_tool, patient_record_manager_tool

# Function to create hospital manager agent
def create_hospital_manager(tools):
    return Agent(
        role="Hospital Manager",
        goal="Manage hospital data by reading the database, analyzing patterns, and generating insights.",
        backstory="You are an AI specialized in processing and understanding hospital data.",
        tools=tools,
        verbose=True
    )

# Function to execute hospital manager tasks
def run_hospital_analysis(csv_path):
    python_repl_tool, csv_reader_tool, patient_record_manager_tool = initialize_tools(csv_path)
    tools = [csv_reader_tool, python_repl_tool, patient_record_manager_tool]
    
    hospital_manager = create_hospital_manager(tools)
    
    manage_hospital_data_task = Task(
        description=f"""
        Read the hospital database from {csv_path}, analyze the data, and generate insights on:
        1. Patient demographics
        2. Common diagnoses
        3. Treatment effectiveness
        4. Financial trends
        
        Be thorough and comprehensive in your analysis.
        """,
        agent=hospital_manager,
        expected_output="Visualizations and insights from the hospital data.",
        tools=tools
    )
    
    crew = Crew(
        agents=[hospital_manager],
        tasks=[manage_hospital_data_task],
        verbose=True
    )
    
    result = crew.kickoff(inputs={"data_path": csv_path})
    return result

# Function to add a new patient record
def add_patient_record(csv_path, record_data):
    _, _, patient_record_manager_tool = initialize_tools(csv_path)
    result = patient_record_manager_tool.execute(record_data)
    return result

# Create two columns for main content
col1, col2 = st.columns([2, 1])

with col1:
    # Diagnostics Section
    st.markdown(
        """
        <div class="card" id="diagnostics">
            <div class="section-header">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2D8CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M22 12h-4l-3 9L9 3l-3 9H2"></path>
                </svg>
                <h2 style="margin: 0 0 0 10px;">Upload Patient Reports</h2>
            </div>
            <p>Simply upload your medical reports and our AI will analyze them for insights and recommendations.</p>
        """,
        unsafe_allow_html=True
    )
    
    uploaded_file = st.file_uploader("Drag and drop your patient report (PDF)", type="pdf", key="pdf")
    
    if uploaded_file:
        st.markdown(
            """
            <div class="success-message">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#34C759" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                    <polyline points="22 4 12 14.01 9 11.01"></polyline>
                </svg>
                Report uploaded successfully! Processing your document...
            </div>
            
            <div style="margin-top: 20px;">
                <h3 style="color: var(--text-dark); font-size: 1.2rem; font-weight: 600;">AI Analysis:</h3>
                <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <p style="color: var(--text-dark); margin: 0;">
                        <span style="color: var(--primary); font-weight: 500;">Key Findings:</span><br>
                        • Blood pressure: 120/80 mmHg (Normal)<br>
                        • Glucose levels: 5.4 mmol/L (Normal)<br>
                        • Cholesterol: 5.2 mmol/L (Borderline high)<br>
                        <br>
                        <span style="color: var(--primary); font-weight: 500;">Recommendations:</span><br>
                        • Follow-up in 6 months<br>
                        • Consider lifestyle modifications to address cholesterol
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Hospital Operations Section
    st.markdown(
        """
        <div class="card" id="operations">
            <div class="section-header">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2D8CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
                    <line x1="8" y1="21" x2="16" y2="21"></line>
                    <line x1="12" y1="17" x2="12" y2="21"></line>
                </svg>
                <h2 style="margin: 0 0 0 10px;">Hospital Operations</h2>
            </div>
            <p>Upload hospital data for AI-powered insights on resource allocation, patient flow, and more.</p>
        """,
        unsafe_allow_html=True
    )
    
    uploaded_excel = st.file_uploader("Upload hospital data (CSV/Excel)", type=["csv", "xls", "xlsx"], key="excel")
    
    if uploaded_excel:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            temp_file.write(uploaded_excel.getvalue())
            temp_file_path = temp_file.name
        
        # Store the path in session state
        st.session_state.csv_path = temp_file_path
        
        # Display loading spinner
        with st.spinner("Analyzing hospital data..."):
            try:
                # Try to read the data to display a preview
                df = pd.read_csv(temp_file_path)
                st.session_state.hospital_data = df
                
                st.markdown(
                    """
                    <div class="success-message">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#34C759" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 8px;">
                            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                            <polyline points="22 4 12 14.01 9 11.01"></polyline>
                        </svg>
                        Hospital data uploaded successfully!
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Show a preview of the data
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Add a button to run the hospital manager analysis
                if st.button("Run AI Analysis"):
                    with st.spinner("Running comprehensive analysis..."):
                        analysis_result = run_hospital_analysis(temp_file_path)
                        st.session_state.analysis_results = analysis_result
                
                # Display analysis results if available
                if st.session_state.analysis_results:
                    st.markdown(
                        """
                        <div style="margin-top: 20px;">
                            <h3 style="color: var(--text-dark); font-size: 1.2rem; font-weight: 600;">AI Analysis Results:</h3>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.write(st.session_state.analysis_results)
                
                # Add new patient record section
                st.subheader("Add New Patient Record")
                
                # Create input fields for patient data
                col_a, col_b = st.columns(2)
                with col_a:
                    patient_id = st.text_input("Patient ID", placeholder="e.g., 65f49173")
                    patient_name = st.text_input("Patient Name", placeholder="e.g., John Doe")
                    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                    diagnosis = st.text_input("Diagnosis", placeholder="e.g., Flu")
                
                with col_b:
                    treatment = st.text_input("Treatment", placeholder="e.g., Medication")
                    admission_date = st.date_input("Admission Date")
                    discharge_date = st.date_input("Discharge Date")
                    bill_amount = st.number_input("Bill Amount", min_value=0.0, step=100.0)
                
                if st.button("Add Patient Record"):
                    # Create record in the format expected by the tool
                    new_record = [
                        patient_id,
                        patient_name,
                        gender,
                        diagnosis,
                        treatment,
                        admission_date.strftime("%Y-%m-%d"),
                        discharge_date.strftime("%Y-%m-%d"),
                        bill_amount
                    ]
                    
                    # Add the record
                    result = add_patient_record(temp_file_path, new_record)
                    st.success(result)
                    
                    # Refresh the data preview
                    df = pd.read_csv(temp_file_path)
                    st.session_state.hospital_data = df
                    st.experimental_rerun()
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                # Clean up the temporary file in case of error
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # Search Medical Information Section
    st.markdown(
        """
        <div class="card" id="search">
            <div class="section-header">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2D8CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="11" cy="11" r="8"></circle>
                    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                </svg>
                <h2 style="margin: 0 0 0 10px;">Medical Information</h2>
            </div>
            <p>Search for reliable information about medical conditions, treatments, and procedures.</p>
        """,
        unsafe_allow_html=True
    )
    
    search_query = st.text_input("Search medical topics", placeholder="e.g., diabetes, hypertension", key="search")
    
    search_button = st.button("Search", key="search_button")
    
    if search_query and search_button:
        st.markdown(
            f"""
            <div style="margin-top: 20px;">
                <h3 style="color: var(--text-dark); font-size: 1.2rem; font-weight: 600;">Results for "{search_query}":</h3>
                <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 15px;">
                    <h4 style="color: var(--primary); margin-top: 0; font-size: 1rem;">Overview</h4>
                    <p style="color: var(--text-dark); margin: 0;">Information about {search_query} would appear here, sourced from medical databases and literature...</p>
                </div>
                
                <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <h4 style="color: var(--primary); margin-top: 0; font-size: 1rem;">Related Topics</h4>
                    <p style="color: var(--text-dark); margin: 0;">
                        • Topic 1<br>
                        • Topic 2<br>
                        • Topic 3
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Data Statistics Section
    if st.session_state.hospital_data is not None:
        st.markdown(
            """
            <div class="card" id="statistics">
                <div class="section-header">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2D8CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                    </svg>
                    <h2 style="margin: 0 0 0 10px;">Data Statistics</h2>
                </div>
            """,
            unsafe_allow_html=True
        )
        
        df = st.session_state.hospital_data
        
        # Display some basic statistics
        st.write(f"Total Records: {len(df)}")
        
        try:
            # Display gender distribution
            if 'Gender' in df.columns or 'gender' in df.columns:
                gender_col = 'Gender' if 'Gender' in df.columns else 'gender'
                gender_counts = df[gender_col].value_counts()
                st.write("Gender Distribution:")
                st.bar_chart(gender_counts)
            
            # Display common diagnoses
            if 'Diagnosis' in df.columns or 'diagnosis' in df.columns:
                diagnosis_col = 'Diagnosis' if 'Diagnosis' in df.columns else 'diagnosis'
                top_diagnoses = df[diagnosis_col].value_counts().head(5)
                st.write("Top Diagnoses:")
                st.bar_chart(top_diagnoses)
        except Exception as e:
            st.error(f"Error generating statistics: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Quick Contact
    st.markdown(
        """
        <div class="card" id="contact">
            <div class="section-header">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2D8CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"></path>
                </svg>
                <h2 style="margin: 0 0 0 10px;">Contact Us - </h2>
                <h4>9823123139</h4>
            </div><p>Have questions? Reach out to our support team for assistance.</p>
             
        """,
        unsafe_allow_html=True
    )

# Clean up temporary files when the app is closed
def cleanup():
    if st.session_state.csv_path and os.path.exists(st.session_state.csv_path):
        os.unlink(st.session_state.csv_path)

# Register the cleanup function to be called when the app is closed
import atexit
atexit.register(cleanup)

# Display footer
footer()