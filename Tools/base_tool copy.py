# Tools/base_tool.py
class BaseTool:
    """
    Base class for all tools. Custom tools should inherit from this class.
    """
    def __init__(self, name: str, description: str):
        """
        Initialize the tool with a name and description.

        Args:
            name (str): The name of the tool.
            description (str): A brief description of the tool's purpose.
        """
        self.name = name
        self.description = description
        self.func = self.execute  # Assign the execute method to the func attribute

    def execute(self, *args, **kwargs):
        """
        Execute the tool's functionality. This method must be implemented by subclasses.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement the execute method.")