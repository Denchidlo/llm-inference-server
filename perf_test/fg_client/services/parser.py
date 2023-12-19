import boto3
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
        elif file_path.startswith('s3://'):
            bucket_name, key = file_path[5:].split('/', 1)
            s3 = boto3.client('s3')
            response = s3.get_object(Bucket=bucket_name, Key=key)
            body = response['Body']
            if file_path.endswith('.xlsx'):
                df = pd.read_excel(body)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(body)
        else:
            raise ValueError("Unsupported file format")

        return df
