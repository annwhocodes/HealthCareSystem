# Tools/patient_record_manager_tool.py
import pandas as pd
from base_tool import BaseTool  # Import the BaseTool class

class PatientRecordManagerTool(BaseTool):
    def __init__(self, csv_path: str):
        """
        Initialize the tool with the path to the CSV file.

        Args:
            csv_path (str): The path to the CSV file containing patient records.
        """
        super().__init__(
            name="Patient Record Manager Tool",
            description="A tool to manage patient records in the CSV dataset."
        )
        self.csv_path = csv_path

    def execute(self, record: dict) -> str:
        """
        Adds a new patient record to the CSV dataset.

        Args:
            record (dict): A dictionary containing the patient record details.

        Returns:
            str: A success or error message.
        """
        try:
            df = pd.read_csv(self.csv_path)
            df = df.append(record, ignore_index=True)
            df.to_csv(self.csv_path, index=False)
            return "Patient record added successfully."
        except Exception as e:
            return f"Error: {str(e)}"