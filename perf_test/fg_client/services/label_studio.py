from label_studio_sdk import Client, Project


class LabelStudio:
    """
    A class for interacting with the Label Studio API.
    """

    def __init__(self, url: str, api_key: str) -> None:
        """
        Initializes a new instance of the class.

        Args:
            url: The URL of the API.
            api_key: The API key for authentication.

        Raises:
            Exception: If the connection to the API fails.
        """
        try:
            self.client = Client(url=url, api_key=api_key)
            print("Connection successful")
        except Exception as e:
            print("Connection failed")
            raise e

    @staticmethod
    def _get_label_config() -> str:
        """
        Returns the label config template as a string.
        """
        label_config = """
        <View>
          <Text name="text-1" value="$text" granularity="word" />
          <Labels name="ner" toName="text-1">
            <Label value="object" background="#FFA39E"/>
            <Label value="state" background="#D4380D"/>
            <Label value="issue" background="#FFC069"/>
            <Label value="case" background="#AD8B00"/>
            <Label value="solution" background="#D3F261"/>
            <Label value="issue_type" background="#389E0D"/>
            <Label value="equipment" background="#5CDBD3"/>
            <Label value="equipment_type" background="#096DD9"/>
            <Label value="location" background="#ADC6FF"/>
            <Label value="attachment" background="#9254DE"/>
            <Label value="project_phase" background="#F759AB"/>
            <Label value="date" background="#fb00ff"/>
            <Label value="out_of_category" background="#000000"/>
          </Labels>
        </View>
        """
        return label_config

    def create_project(self, project_name: str) -> Project:
        """
        Creates a project with the given project name.

        Args:
            project_name: The name of the project.

        Returns:
            The created project object.
        """
        project_template = self._get_label_config()
        project = self.client.start_project(title=project_name,
                                            description='Labeling for Facility Grid project',
                                            label_config=project_template)

        return project

    @staticmethod
    def import_task(project: Project, text: str, meta_info: dict[str, str], results: list[dict[str, str]]) -> None:
        """
            Import tasks into a project.

            Args:
                project: The project to import tasks into.
                text: The text for the task.
                meta_info: Additional information about the task.
                results: The NER results for the task.
            """
        project.import_tasks([{'data': {'text': text,
                                        'meta_info': meta_info},
                               'annotations': [{'result': results}],
                               'predictions': []}])

    @staticmethod
    def export_tasks(project: Project) -> list[dict[str, str]]:
        """
        Export tasks from a project.

        Args:
            project: The project to export tasks from.
        """
        tasks = project.export_tasks(export_type='JSON_MIN',
                                     download_all_tasks=True)

        return tasks
