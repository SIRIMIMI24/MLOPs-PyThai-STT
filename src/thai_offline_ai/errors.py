"""Errors safe to display without user inputs or model outputs."""


class WorkbenchError(RuntimeError):
    """A user-actionable application failure."""


class ConfigurationError(WorkbenchError, ValueError):
    """Invalid configuration or environment override."""
