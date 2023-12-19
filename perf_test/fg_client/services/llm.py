import contextlib
import io

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama


class Llm:
    """
    LLM wrapper for LLM models.
    """

    def __init__(self, model_url: str, model_name: str) -> None:
        """
        Initializes an instance of the class.

        Args:
            model_url: The base URL for the LLM API.
            model_name: The name of the model to use.
        """
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        if model_name == 'llama2':
            try:
                self.client = Ollama(base_url=model_url, model='llama2',
                                     callback_manager=callback_manager)
            except Exception as e:
                print("Connection failed")
                raise e
        else:
            print("Unsupported model")

    def get_answer(self, query: str) -> str:
        """
        Retrieves the answer by querying the LLM.

        Args:
            query: The query to be sent to the LLM.

        Returns:
            The answer returned by the LLM.
        """
        with contextlib.redirect_stdout(io.StringIO()):
            answer = self.client(query)
        return answer
