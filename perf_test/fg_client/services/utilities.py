import datetime


def current_timestamp():
    """
    Current timestamp without timezone (0:00).

    Returns:
        Current timestamp.
    """
    datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
