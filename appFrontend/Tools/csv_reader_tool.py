import pandas as pd

class BaseTool:
    def __init__(self, name, description):
        self.name = name
        self.description = description

    def execute(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the execute method.")

class CSVReaderTool(BaseTool):
    def __init__(self, csv_path):
        super().__init__(
            name="CSV Reader Tool",
            description="A tool to read CSV files and generate Python code to analyze them."
        )
        self.csv_path = csv_path

    def execute(self):
        df = pd.read_csv(self.csv_path)
        return df