"""
Caption service for caption editing operations.

Handles:
- Add trigger words to captions
- Remove specific tags
- Replace text (with regex support)
- Read/write individual captions
- Bulk operations on entire datasets
"""

import re
import logging
from pathlib import Path
from typing import List, Optional

from services.models.caption import (
    AddTriggerWordRequest,
    RemoveTagsRequest,
    ReplaceTextRequest,
    ReadCaptionRequest,
    WriteCaptionRequest,
    CaptionOperationResponse,
    CaptionReadResponse,
)
from services.core.exceptions import ValidationError, NotFoundError
from services.core.validation import validate_dataset_path, ALLOWED_IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


class CaptionService:
    """
    High-level service for caption editing.

    Responsibilities:
    - Bulk caption operations
    - Individual caption read/write
    - Tag manipulation
    - Text replacement
    """

    async def add_trigger_word(self, request: AddTriggerWordRequest) -> CaptionOperationResponse:
        """
        Add a trigger word to all captions in a dataset.

        Args:
            request: Trigger word addition request

        Returns:
            CaptionOperationResponse with count of modified files
        """
        try:
            dataset_path = validate_dataset_path(request.dataset_dir)

            if not dataset_path.exists():
                raise NotFoundError(f"Dataset not found: {request.dataset_dir}")

            files_modified = 0
            errors = []

            # Process all caption files
            for image_file in dataset_path.iterdir():
                if not image_file.is_file():
                    continue
                if image_file.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
                    continue

                caption_file = image_file.with_suffix(request.caption_extension)

                try:
                    # Read existing caption or create empty
                    if caption_file.exists():
                        caption_text = caption_file.read_text(encoding='utf-8').strip()
                    else:
                        caption_text = ""

                    # Add trigger word
                    if request.position == "start":
                        new_caption = f"{request.trigger_word}, {caption_text}" if caption_text else request.trigger_word
                    else:  # end
                        new_caption = f"{caption_text}, {request.trigger_word}" if caption_text else request.trigger_word

                    # Write back
                    caption_file.write_text(new_caption, encoding='utf-8')
                    files_modified += 1

                except Exception as e:
                    errors.append(f"{image_file.name}: {str(e)}")

            logger.info(f"Added trigger word '{request.trigger_word}' to {files_modified} captions")

            return CaptionOperationResponse(
                success=True,
                message=f"Added trigger word to {files_modified} captions",
                files_modified=files_modified,
                errors=errors
            )

        except (ValidationError, NotFoundError) as e:
            return CaptionOperationResponse(
                success=False,
                message=str(e),
                files_modified=0,
                errors=[str(e)]
            )

    async def remove_tags(self, request: RemoveTagsRequest) -> CaptionOperationResponse:
        """
        Remove specific tags from all captions in a dataset.

        Args:
            request: Tag removal request

        Returns:
            CaptionOperationResponse with count of modified files
        """
        try:
            dataset_path = validate_dataset_path(request.dataset_dir)

            if not dataset_path.exists():
                raise NotFoundError(f"Dataset not found: {request.dataset_dir}")

            files_modified = 0
            errors = []

            # Process all caption files
            for caption_file in dataset_path.glob(f"*{request.caption_extension}"):
                try:
                    caption_text = caption_file.read_text(encoding='utf-8').strip()

                    # Split into tags
                    tags = [tag.strip() for tag in caption_text.split(',')]

                    # Remove unwanted tags (case-insensitive)
                    tags_lower = {tag.lower() for tag in request.tags_to_remove}
                    filtered_tags = [tag for tag in tags if tag.lower() not in tags_lower]

                    # Write back if changed
                    if len(filtered_tags) != len(tags):
                        new_caption = ', '.join(filtered_tags)
                        caption_file.write_text(new_caption, encoding='utf-8')
                        files_modified += 1

                except Exception as e:
                    errors.append(f"{caption_file.name}: {str(e)}")

            logger.info(f"Removed tags from {files_modified} captions")

            return CaptionOperationResponse(
                success=True,
                message=f"Removed tags from {files_modified} captions",
                files_modified=files_modified,
                errors=errors
            )

        except (ValidationError, NotFoundError) as e:
            return CaptionOperationResponse(
                success=False,
                message=str(e),
                files_modified=0,
                errors=[str(e)]
            )

    async def replace_text(self, request: ReplaceTextRequest) -> CaptionOperationResponse:
        """
        Replace text in all captions (with optional regex support).

        Args:
            request: Text replacement request

        Returns:
            CaptionOperationResponse with count of modified files
        """
        try:
            dataset_path = validate_dataset_path(request.dataset_dir)

            if not dataset_path.exists():
                raise NotFoundError(f"Dataset not found: {request.dataset_dir}")

            files_modified = 0
            errors = []

            # Process all caption files
            for caption_file in dataset_path.glob(f"*{request.caption_extension}"):
                try:
                    caption_text = caption_file.read_text(encoding='utf-8')

                    # Replace text
                    if request.use_regex:
                        new_caption = re.sub(
                            request.find_text,
                            request.replace_text,
                            caption_text
                        )
                    else:
                        new_caption = caption_text.replace(
                            request.find_text,
                            request.replace_text
                        )

                    # Write back if changed
                    if new_caption != caption_text:
                        caption_file.write_text(new_caption, encoding='utf-8')
                        files_modified += 1

                except Exception as e:
                    errors.append(f"{caption_file.name}: {str(e)}")

            logger.info(f"Replaced text in {files_modified} captions")

            return CaptionOperationResponse(
                success=True,
                message=f"Replaced text in {files_modified} captions",
                files_modified=files_modified,
                errors=errors
            )

        except (ValidationError, NotFoundError) as e:
            return CaptionOperationResponse(
                success=False,
                message=str(e),
                files_modified=0,
                errors=[str(e)]
            )

    async def read_caption(self, request: ReadCaptionRequest) -> CaptionReadResponse:
        """
        Read a single caption file.

        Args:
            request: Caption read request

        Returns:
            CaptionReadResponse with caption content
        """
        try:
            image_path = Path(request.image_path)
            caption_path = image_path.with_suffix(request.caption_extension)

            if caption_path.exists():
                caption_text = caption_path.read_text(encoding='utf-8')
                return CaptionReadResponse(
                    success=True,
                    image_path=str(image_path),
                    caption_path=str(caption_path),
                    caption_text=caption_text,
                    exists=True
                )
            else:
                return CaptionReadResponse(
                    success=True,
                    image_path=str(image_path),
                    caption_path=str(caption_path),
                    caption_text=None,
                    exists=False
                )

        except Exception as e:
            logger.error(f"Failed to read caption: {e}")
            return CaptionReadResponse(
                success=False,
                image_path=request.image_path,
                caption_path=str(Path(request.image_path).with_suffix(request.caption_extension)),
                caption_text=None,
                exists=False
            )

    async def write_caption(self, request: WriteCaptionRequest) -> CaptionOperationResponse:
        """
        Write a single caption file.

        Args:
            request: Caption write request

        Returns:
            CaptionOperationResponse
        """
        try:
            image_path = Path(request.image_path)
            caption_path = image_path.with_suffix(request.caption_extension)

            # Ensure parent directory exists
            caption_path.parent.mkdir(parents=True, exist_ok=True)

            # Write caption
            caption_path.write_text(request.caption_text, encoding='utf-8')

            logger.info(f"Wrote caption: {caption_path.name}")

            return CaptionOperationResponse(
                success=True,
                message=f"Caption written to {caption_path.name}",
                files_modified=1,
                errors=[]
            )

        except Exception as e:
            logger.error(f"Failed to write caption: {e}")
            return CaptionOperationResponse(
                success=False,
                message=f"Failed to write caption: {e}",
                files_modified=0,
                errors=[str(e)]
            )


# Global service instance
caption_service = CaptionService()
