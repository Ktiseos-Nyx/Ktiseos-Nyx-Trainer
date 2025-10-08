# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Ktiseos Nyx
# Contributors: See README.md Credits section for full acknowledgements

# shared_managers.py
# This file provides shared manager instances for use across widgets
# Uses lazy loading - only creates managers when actually needed

# Suppress annoying startup warnings
import warnings
import logging
import os

# Suppress FutureWarnings from diffusers
warnings.filterwarnings('ignore', category=FutureWarning, module='diffusers')
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')

# Reduce logging verbosity for startup
logging.getLogger('diffusers').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('huggingface_hub').setLevel(logging.WARNING)

# Global storage for manager instances (lazy loaded)
_setup_manager = None
_model_manager = None
_dataset_manager = None
_training_manager = None
_utilities_manager = None
_config_manager = None
_inference_manager = None

def get_setup_manager():
    """Get or create the setup manager"""
    global _setup_manager
    if _setup_manager is None:
        from core.managers import SetupManager
        _setup_manager = SetupManager()
    return _setup_manager

def get_model_manager():
    """Get or create the model manager"""
    global _model_manager
    if _model_manager is None:
        from core.managers import ModelManager
        _model_manager = ModelManager()
    return _model_manager

def get_dataset_manager():
    """Get or create the dataset manager (no forced ModelManager dependency!)"""
    global _dataset_manager
    if _dataset_manager is None:
        from core.dataset_manager import DatasetManager
        _dataset_manager = DatasetManager()  # ModelManager loaded lazily when needed
    return _dataset_manager

def get_training_manager():
    """Get or create the training manager (heavy imports - only load when needed!)"""
    global _training_manager
    if _training_manager is None:
        from core.refactored_training_manager import RefactoredTrainingManager
        _training_manager = RefactoredTrainingManager()
    return _training_manager

def get_utilities_manager():
    """Get or create the utilities manager"""
    global _utilities_manager
    if _utilities_manager is None:
        from core.utilities_manager import UtilitiesManager
        _utilities_manager = UtilitiesManager()
    return _utilities_manager

def get_config_manager():
    """Get or create the config manager"""
    global _config_manager
    if _config_manager is None:
        from core.config_manager import ConfigManager
        _config_manager = ConfigManager()
    return _config_manager

def get_inference_manager():
    """Get or create the inference manager"""
    global _inference_manager
    if _inference_manager is None:
        from core.refactored_inference_manager import RefactoredInferenceManager
        _inference_manager = RefactoredInferenceManager()
    return _inference_manager

# File manager instance (lazy loaded)
_file_manager = None

def get_file_manager():
    """Get or create FileManager instance"""
    global _file_manager
    if _file_manager is None:
        from core.file_manager import FileManagerUtility
        _file_manager = FileManagerUtility()
    return _file_manager

# Lazy widget imports - only import when actually needed!

def show_recent_logs(lines=50):
    """Show recent log entries for debugging"""
    from core.logging_config import tail_log_file
    tail_log_file(lines)

def get_log_file_path():
    """Get the current log file path for manual viewing"""
    from core.logging_config import get_latest_log_file
    return get_latest_log_file()

def create_widgets():
    """
    Create all widgets with shared manager instances (lazy loaded)
    This ensures all widgets use the same state
    """
    return {
        'setup': create_widget('setup'),
        'dataset': create_widget('dataset'),
        'training': create_widget('training'),
        'utilities': create_widget('utilities'),
        'calculator': create_widget('calculator'),
        'file_manager': create_widget('file_manager'),
        'image_curation': create_widget('image_curation')
    }

def create_widget(widget_name):
    """Create a single widget with proper lazy-loaded dependency injection"""
    if widget_name == 'setup':
        from widgets.setup_widget import SetupWidget
        return SetupWidget(get_setup_manager(), get_model_manager())
    elif widget_name == 'setup_simple':
        from widgets.setup_widget import SetupWidget
        widget = SetupWidget(get_setup_manager(), get_model_manager())
        widget.simplified_mode = True
        return widget
    elif widget_name == 'dataset':
        from widgets.dataset_widget import DatasetWidget
        return DatasetWidget(get_dataset_manager())
    elif widget_name == 'training':
        from widgets.training_widget import TrainingWidget
        return TrainingWidget(get_training_manager())
    elif widget_name == 'utilities':
        from widgets.utilities_widget import UtilitiesWidget
        return UtilitiesWidget(get_utilities_manager())
    elif widget_name == 'calculator':
        from widgets.calculator_widget import CalculatorWidget
        return CalculatorWidget()  # No manager needed - pure math widget!
    elif widget_name == 'file_manager':
        from widgets.file_manager_widget import create_file_manager_widget
        return create_file_manager_widget()  # Returns widget directly
    elif widget_name == 'image_curation':
        from widgets.image_curation_widget import ImageCurationWidget
        return ImageCurationWidget(shared_managers=None)  # Self-contained widget
    else:
        available = ['setup', 'dataset', 'training', 'utilities', 'calculator', 'file_manager', 'image_curation']
        raise ValueError(f"Unknown widget: {widget_name}. Available: {available}")