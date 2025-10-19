'''
nfeloml - Portable machine learning models for NFL analytics

This package provides trained XGBoost models for NFL play-by-play predictions,
including Expected Points (EP) and Win Probability (WP).
'''
from .models.expected_points.model import ExpectedPointsModel
from .models.expected_points.types import EPFeatures, EPPrediction, EPTrainingConfig
from .models.win_probability.model import WinProbabilityModel
from .models.win_probability.types import WPFeatures, WPPrediction, WPTrainingConfig, WPSpreadTrainingConfig
from .core.types import ModelMetadata
from .utils import calculate_epa

__version__ = "0.1.1"

__all__ = [
    ##  Expected Points
    'ExpectedPointsModel',
    'EPFeatures',
    'EPPrediction',
    'EPTrainingConfig',
    ##  Win Probability
    'WinProbabilityModel',
    'WPFeatures',
    'WPPrediction',
    'WPTrainingConfig',
    'WPSpreadTrainingConfig',
    ##  Core
    'ModelMetadata',
    ##  Utilities
    'calculate_epa',
]

