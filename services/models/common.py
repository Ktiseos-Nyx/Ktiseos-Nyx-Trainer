"""
Common Pydantic models shared across services.
"""

from typing import Generic, TypeVar, Optional
from pydantic import BaseModel

# Generic type variable for pagination
T = TypeVar('T')


class PaginatedResponse(BaseModel, Generic[T]):
    """
    Generic paginated response.

    Used for any list endpoints that need pagination.

    Example:
        PaginatedResponse[Dataset](
            items=[...datasets...],
            total=100,
            page=1,
            page_size=20
        )
    """
    items: list[T]
    total: int
    page: int
    page_size: int

    @property
    def pages(self) -> int:
        """Calculate total number of pages"""
        return (self.total + self.page_size - 1) // self.page_size

    @property
    def has_next(self) -> bool:
        """Check if there's a next page"""
        return self.page < self.pages

    @property
    def has_prev(self) -> bool:
        """Check if there's a previous page"""
        return self.page > 1


class ErrorResponse(BaseModel):
    """
    Standard error response format.

    All API errors return this format for consistency.
    """
    error: str
    detail: Optional[str] = None
    field: Optional[str] = None  # For validation errors

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "detail": "Learning rate must be between 0 and 1",
                "field": "learning_rate"
            }
        }
