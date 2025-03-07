from crewai import Agent, Task, Crew
from Tools.repl_tool import PythonREPLTool, BaseTool  # Import the custom tool and BaseTool
from Tools.csv_reader_tool import CSVReaderTool  # Import the CSVReaderTool
from Tools.visualiser_tool import VisualiserTool  # Import the VisualiserTool
import pandas as pd

# Check if the import is successful
print(PythonREPLTool)
print(CSVReaderTool)
print(VisualiserTool)

# Create an instance of the PythonREPLTool
python_repl_tool = PythonREPLTool()

# Create an instance of the CSVReaderTool
csv_reader_tool = CSVReaderTool("C://Users//Ananya//Desktop//Hackathon_Project//Data//hospital_records_2021_2024_with_bills.csv")

# Define the PatientRecordManagerTool
class PatientRecordManagerTool(BaseTool):
    def __init__(self, csv_path):
        super().__init__(
            name="Patient Record Manager Tool",
            description="A tool to manage patient records in the CSV dataset."
        )
        self.csv_path = csv_path

    def execute(self, record):
        df = pd.read_csv(self.csv_path)
        df = df.append(record, ignore_index=True)
        df.to_csv(self.csv_path, index=False)
        return "Patient record added successfully."

# Create an instance of the PatientRecordManagerTool
patient_record_manager_tool = PatientRecordManagerTool("C://Users//Ananya//Desktop//Hackathon_Project//Data//hospital_records_2021_2024_with_bills.csv")

# Define the Hospital Manager Agent
hospital_manager = Agent(
    role="Hospital Manager",
    goal="Manage hospital data by reading the database from {data_path}, thinking which data visualisation will be suitable based on generated insights.",
    backstory="You are an AI specialized in processing and understanding hospital data.",
    tools=[csv_reader_tool, python_repl_tool, patient_record_manager_tool],  # Add the custom tools
    verbose=True
)

# Define a task to read and visualize the hospital database
manage_hospital_data_task = Task(
    description="""
    Read the hospital database from {data_path}, visualize the data using colorful graphs and charts, and generate insights.
    """,
    agent=hospital_manager,
    expected_output="Visualizations and insights from the hospital data.",
    tools=[csv_reader_tool, python_repl_tool, patient_record_manager_tool]  # Ensure the task uses the tools
)

# Define a task to add patient records
add_patient_record_task = Task(
    description="""
    Add a new patient record to the hospital database.
    """,
    agent=hospital_manager,
    expected_output="Patient record added successfully.",
    tools=[patient_record_manager_tool]  # Ensure the task uses the PatientRecordManagerTool
)

# Create a crew
crew = Crew(
    agents=[hospital_manager],
    tasks=[manage_hospital_data_task, add_patient_record_task],
    verbose=True
)

# Run the crew to visualize the data
result = crew.kickoff(inputs={"data_path": "C://Users//Ananya//Desktop//Hackathon_Project//Data//hospital_records_2021_2024_with_bills.csv"})
print(result)

# Example usage to add a patient record
new_record = {
    "Patient ID": 12345,
    "Name": "John Doe",
    "Gender": "Male",
    "Medical Condition": "Flu",
    "Treatments": "Medication",
    "Admit Date": "2024-01-01",
    "Discharge Date": "2024-01-05",
    "Bill Amount": 500.00
}
add_result = patient_record_manager_tool.execute(new_record)  # filepath: c:\Users\Ananya\Desktop\Hackathon_Project\hospital_manager_agent.py
