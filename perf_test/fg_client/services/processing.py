import re


class Processing:
    """
    A class for processing data.
    """

    @staticmethod
    def clean_issues(issues: list[str]) -> list[str]:
        """
        Cleans up the issues list by removing unnecessary whitespace characters, etc.

        Args:
            issues: The list of issues to be cleaned up.

        Returns:
            The cleaned up list of issues.
        """

        for i, issue in enumerate(issues):
            # delete '\n', '\xa0', ' '
            issues[i] = re.sub(r'\s+', ' ', issue).strip()
        return issues
