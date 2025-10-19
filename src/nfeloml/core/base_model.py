'''
Abstract base class for all models
'''
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import json
import xgboost as xgb
import pandas as pd
from .types import ModelMetadata, PredictionInput, PredictionOutput

class BaseModel(ABC):
    '''
    Abstract base class for machine learning models
    
    Provides common functionality for loading, saving, and making predictions
    '''
    
    def __init__(self, model_path: Optional[Path] = None):
        '''
        Initialize the model
        
        Params:
        * model_path: Optional[Path] - path to saved model file (if None, uses package default)
        '''
        if model_path is None:
            ##  Use default path in model's trained_model directory
            model_dir = Path(self.__class__.__module__.replace('.', '/')).parent
            package_root = Path(__file__).parent.parent.parent
            model_path = package_root / model_dir / 'trained_model' / 'model.ubj'
        self.model_path = model_path
        self.xgb_model: Optional[xgb.Booster] = None
        self.metadata: Optional[ModelMetadata] = None
        if model_path and model_path.exists():
            self._load_model()
    
    @classmethod
    def load(cls, model_path: Path) -> 'BaseModel':
        '''
        Load a trained model from disk
        
        Params:
        * model_path: Path - path to the model file
        
        Returns:
        * BaseModel: loaded model instance
        '''
        return cls(model_path=model_path)
    
    def _load_model(self) -> None:
        '''
        Internal method to load model and metadata from disk
        '''
        if not self.model_path:
            raise ValueError("model_path must be set to load a model")
        ##  Load the XGBoost model
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(str(self.model_path))
        ##  Load metadata if it exists
        metadata_path = self.model_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
                self.metadata = self._dict_to_metadata(metadata_dict)
    
    @abstractmethod
    def _dict_to_metadata(self, metadata_dict: dict) -> ModelMetadata:
        '''
        Convert dictionary to ModelMetadata (subclass specific)
        
        Params:
        * metadata_dict: dict - metadata dictionary
        
        Returns:
        * ModelMetadata: metadata object
        '''
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, features: PredictionInput, include_probabilities: bool = False):
        '''
        Make predictions using the model
        
        Params:
        * features: PredictionInput - input features
        * include_probabilities: bool - if False, returns summary float; if True, returns full dataclass
        
        Returns:
        * float or PredictionOutput: summary value or full predictions
        '''
        raise NotImplementedError
    
    @abstractmethod
    def predict_df(self, df: 'pd.DataFrame', include_probabilities: bool = False) -> 'pd.DataFrame':
        '''
        Make predictions on a DataFrame
        
        Params:
        * df: pd.DataFrame - dataframe with required columns
        * include_probabilities: bool - if False, adds one summary column; if True, adds all probability columns
        
        Returns:
        * pd.DataFrame: dataframe with prediction column(s) added
        '''
        raise NotImplementedError
    
    def get_metadata(self) -> Optional[ModelMetadata]:
        '''
        Get model metadata
        
        Returns:
        * Optional[ModelMetadata]: model metadata if available
        '''
        return self.metadata
    
    def save(self, save_path: Path) -> None:
        '''
        Save the model and metadata to disk
        
        Params:
        * save_path: Path - path to save the model
        '''
        if not self.xgb_model:
            raise ValueError("No model to save")
        ##  Save the XGBoost model
        self.xgb_model.save_model(str(save_path))
        self.model_path = save_path
        ##  Save metadata if available
        if self.metadata:
            metadata_path = save_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata.to_dict(), f, indent=2)

