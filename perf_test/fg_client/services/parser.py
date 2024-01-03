import pandas as pd


class Parser:
    """
    Parser class for parsing CSV and Excel files.
    """

    @staticmethod
    def load(file_path: str) -> pd.DataFrame:
        """
        Load a file from the given file path and return a pandas DataFrame.

        Args:
            file_path: The path to the file.

        Returns:
            df: The loaded data as a pandas DataFrame.
        """
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format")

        return df
