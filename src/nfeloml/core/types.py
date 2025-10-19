'''
Core type definitions and dataclasses used across all models
'''
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from abc import ABC

@dataclass
class TrainingConfig:
    '''
    Base configuration for model training
    
    Params:
    * seasons: list[int] - seasons to include in training data
    * validation_strategy: str - cross-validation strategy (e.g., 'loso')
    * random_seed: int - random seed for reproducibility
    * verbose: bool - whether to print training progress
    '''
    seasons: list[int]
    validation_strategy: str = "loso"
    random_seed: int = 2013
    verbose: bool = True

@dataclass
class ModelMetadata:
    '''
    Metadata about a trained model
    
    Params:
    * model_name: str - name of the model
    * version: str - model version
    * trained_date: datetime - when the model was trained
    * training_seasons: list[int] - seasons used in training
    * calibration_error: Optional[float] - calibration error metric
    * additional_metrics: Dict[str, Any] - any other performance metrics
    '''
    model_name: str
    version: str
    trained_date: datetime
    training_seasons: list[int]
    calibration_error: Optional[float] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        '''
        Convert metadata to dictionary for serialization
        
        Returns:
        * Dict[str, Any]: metadata as dictionary
        '''
        import numpy as np
        ##  Convert numpy types to native Python types for JSON serialization
        def convert_value(val):
            if isinstance(val, (np.integer, np.int32, np.int64)):
                return int(val)
            elif isinstance(val, (np.floating, np.float32, np.float64)):
                return float(val)
            elif isinstance(val, list):
                return [convert_value(v) for v in val]
            elif isinstance(val, dict):
                return {k: convert_value(v) for k, v in val.items()}
            return val
        return {
            'model_name': self.model_name,
            'version': self.version,
            'trained_date': self.trained_date.isoformat(),
            'training_seasons': convert_value(self.training_seasons),
            'calibration_error': convert_value(self.calibration_error),
            'additional_metrics': convert_value(self.additional_metrics)
        }

@dataclass
class PredictionInput(ABC):
    '''
    Base class for model prediction inputs
    '''
    pass

@dataclass
class PredictionOutput(ABC):
    '''
    Base class for model prediction outputs
    '''
    pass

