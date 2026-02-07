"""
Pydantic models for caption editing operations.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class CaptionFile(BaseModel):
    """Caption file information."""
    image_path: str
    caption_path: str
    caption_text: str
    exists: bool = True


class AddTriggerWordRequest(BaseModel):
    """Request to add trigger word to captions."""
    dataset_dir: str = Field(..., description="Dataset directory path")
    trigger_word: str = Field(..., min_length=1, description="Trigger word to add")
    position: str = Field("start", description="'start' or 'end'")
    caption_extension: str = Field(".txt", description="Caption file extension")


class RemoveTagsRequest(BaseModel):
    """Request to remove tags from captions."""
    dataset_dir: str = Field(..., description="Dataset directory path")
    tags_to_remove: List[str] = Field(..., min_items=1, description="Tags to remove")
    caption_extension: str = Field(".txt", description="Caption file extension")


class ReplaceTextRequest(BaseModel):
    """Request to replace text in captions."""
    dataset_dir: str = Field(..., description="Dataset directory path")
    find_text: str = Field(..., min_length=1, description="Text to find")
    replace_text: str = Field("", description="Replacement text")
    caption_extension: str = Field(".txt", description="Caption file extension")
    use_regex: bool = Field(False, description="Use regex pattern matching")


class ReadCaptionRequest(BaseModel):
    """Request to read a single caption."""
    image_path: str = Field(..., description="Path to image file")
    caption_extension: str = Field(".txt", description="Caption file extension")


class WriteCaptionRequest(BaseModel):
    """Request to write a single caption."""
    image_path: str = Field(..., description="Path to image file")
    caption_text: str = Field(..., description="Caption content")
    caption_extension: str = Field(".txt", description="Caption file extension")


class CaptionOperationResponse(BaseModel):
    """Response from caption operation."""
    success: bool
    message: str
    files_modified: int = 0
    errors: List[str] = Field(default_factory=list)


class CaptionReadResponse(BaseModel):
    """Response from reading caption."""
    success: bool
    image_path: str
    caption_path: str
    caption_text: Optional[str] = None
    exists: bool = False
