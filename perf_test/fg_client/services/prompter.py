from langchain.prompts import PromptTemplate


class Prompter:
    """
    Prompter class for generating prompts.
    """

    @staticmethod
    def get_result_format() -> str:
        result_format = """
            [{"value": {"start": 26,
                      "end": 36,
                      "text": "flow meter",
                      "labels": ["object"]},
            "from_name": "ner",
            "to_name": "text-1",
            "type": "labels"},
            
            {"value": {"start": 37,
                      "end": 56,
                      "text": "has stopped working",
                      "labels": ["state"]},
            "from_name": "ner",
            "to_name": "text-1",
            "type": "labels"},
            
            {"value": {"start": 26,
                      "end": 56,
                      "text": "flow meter has stopped working",
                      "labels": ["issue"]},
            "from_name": "ner",
            "to_name": "text-1",
            "type": "labels"},
            
            {"value": {"start": 18,
                      "end": 25,
                      "text": "CRAH 10",
                      "labels": ["equipment"]},
            "from_name": "ner",
            "to_name": "text-1",
            "type": "labels"},
            
            {"value": {"start": 8,
                      "end": 17,
                      "text": "2nd Floor",
                      "labels": ["location"]},
            "from_name": "ner",
            "to_name": "text-1",
            "type": "labels"},
            
            {"value": {"start": 94,
                      "end": 136,
                      "text": "this needs to be investigated and resolved",
                      "labels": ["solution"]},
            "from_name": "ner",
            "to_name": "text-1",
            "type": "labels"},
            {"value": {"start": 59,
                      "end": 89,
                      "text": "The valve is still controlling",
                      "labels": ["out_of_category"]},
            "from_name": "ner",
            "to_name": "text-1",
            "type": "labels"}]
        """

        return result_format

    @staticmethod
    def get_prompt() -> PromptTemplate:
        """
        Generate the PromptTemplate.

        :return: PromptTemplate.
        """
        template = """
        <s>[INST] <<SYS>>
        Act as a civil engineer, who labels issues for NER in the construction field into the following components:
        
        object - object specified in the ‘Issue description’ and with which the issue is associated
        state - state of the object that is associated with the issue
        issue = Object + State - direct description of the issue
        case - cause of current problem
        solution - possible solution that can solve the current issue
        issue Type - type of issue from key words
        equipment - equipment type, abbreviation or equipment number
        equipment_type - equipment type
        location - location where the system is located
        attachment - Indication of whether there are additions in the form of photos or other files
        project_phase – Project phase and task
        date - Date of the issue
        out_of_category - Data that does not fit into any category is marked here.
        <</SYS>>
        
        issue: {text}
        Example: Stulz - 2nd Floor CRAH 10 flow meter has stopped working.  The valve is still controlling but this needs to be investigated and resolved.
        The resulting format should be as follows: {result_format}
        [/INST]
        """

        prompt_template = PromptTemplate(input_variables=['text', 'result_format'],
                                         template=template)

        return prompt_template
