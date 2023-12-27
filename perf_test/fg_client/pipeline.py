from .services.parser import Parser
from .services.processing import Processing
from .services.prompter import Prompter

from . import config


def load_data(filename: str, num_samples: int=10) -> list[str]:
    """
    Args:
        num_samples (int): amount of string this function will return
    Return:
        inp_batch (list[str])
    """
    df = Parser.load(filename)

    uuid = df[config.uuid].to_list()
    issues = df[config.issue_description].to_list()
    issues = Processing.clean_issues(issues)

    prompt = Prompter.get_prompt()
    result_format = Prompter.get_result_format()


    issues = issues[:min(len(issues), num_samples)]
    inp_batch = [prompt.format(text=issue, result_format=result_format) for issue in issues]
    return inp_batch
