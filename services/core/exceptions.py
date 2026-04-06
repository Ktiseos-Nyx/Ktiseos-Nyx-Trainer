"""
Custom exception hierarchy for service layer.

All service exceptions inherit from ServiceError for easy catching.
Use specific exceptions to provide meaningful error messages to the API layer.
"""


class ServiceError(Exception):
    """
    Base exception for all service layer errors.

    Catch this to handle any service-related error.
    """
    pass


class ValidationError(ServiceError):
    """
    Invalid input data.

    Raised when:
    - User input fails validation (e.g., learning rate out of range)
    - Config parameters are invalid
    - File types are incorrect

    HTTP equivalent: 400 Bad Request
    """
    pass


class NotFoundError(ServiceError):
    """
    Resource not found.

    Raised when:
    - Dataset doesn't exist
    - Model file not found
    - Job ID doesn't exist

    HTTP equivalent: 404 Not Found
    """
    pass


class ProcessError(ServiceError):
    """
    Subprocess execution failed.

    Raised when:
    - Training process crashes
    - Tagging process fails
    - External script returns non-zero exit code

    HTTP equivalent: 500 Internal Server Error
    """
    pass


class ConfigError(ServiceError):
    """
    Invalid configuration.

    Raised when:
    - TOML generation fails
    - Required config fields missing
    - Config conflicts detected

    HTTP equivalent: 400 Bad Request
    """
    pass
