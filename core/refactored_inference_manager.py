
# core/refactored_inference_manager.py
"""
Compatibility wrapper for the inference system.
"""

import logging

from .kohya_inference_manager import KohyaInferenceManager

logger = logging.getLogger(__name__)

class RefactoredInferenceManager:
    """
    This class provides a unified interface for inference, delegating the work
    to the new KohyaInferenceManager. It ensures that existing widgets and
    code can operate without modification.
    """

    def __init__(self):
        self.kohya_manager = KohyaInferenceManager()
        logger.info("RefactoredInferenceManager initialized, using Kohya backend.")

    def generate(self, params):
        """
        A single, unified method to handle image generation requests.
        It passes the request directly to the KohyaInferenceManager.

        Args:
            params (dict): A dictionary of parameters for image generation,
                           including 'model_path', 'prompt', etc.
        """
        logger.info("Delegating generation request to KohyaInferenceManager.")
        try:
            self.kohya_manager.generate_image(params)
        except Exception as e:
            logger.error(f"An error occurred in the inference backend: {e}")

# Alias for backward compatibility if any old code specifically imports it.
InferenceManager = RefactoredInferenceManager
