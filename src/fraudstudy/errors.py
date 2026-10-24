"""Project exception hierarchy. Messages should always say what the user needs to fix."""


class FraudStudyError(Exception):
    """Base class for every error raised by fraudstudy."""


class ConfigError(FraudStudyError):
    """A configuration file is missing, unreadable, or lacks required keys."""


class DataValidationError(FraudStudyError):
    """Input data does not match the expected schema or contents."""


class LeakageError(FraudStudyError):
    """An operation would leak information from validation/test data into training."""
