'''
Abstract base class for model training
'''
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import time
from .types import TrainingConfig, ModelMetadata
from .base_data_loader import BaseDataLoader

class BaseTrainer(ABC):
    '''
    Abstract base class for training models with cross-validation
    '''
    
    def __init__(self, config: TrainingConfig, data_loader: BaseDataLoader):
        '''
        Initialize the trainer
        
        Params:
        * config: TrainingConfig - training configuration
        * data_loader: BaseDataLoader - data loader instance
        '''
        self.config = config
        self.data_loader = data_loader
        self.trained_model: Optional[xgb.Booster] = None
        self.cv_results: Optional[pd.DataFrame] = None
    
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        '''
        Load and prepare training data
        
        Returns:
        * pd.DataFrame: prepared training data
        '''
        raise NotImplementedError
    
    @abstractmethod
    def prepare_features(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        '''
        Prepare features, labels, and optional weights for training
        
        Params:
        * data: pd.DataFrame - training data
        
        Returns:
        * tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]: features, labels, and optional weights
        '''
        raise NotImplementedError
    
    @abstractmethod
    def get_xgb_params(self) -> Dict[str, Any]:
        '''
        Get XGBoost parameters for training
        
        Returns:
        * Dict[str, Any]: XGBoost parameters
        '''
        raise NotImplementedError
    
    @abstractmethod
    def get_num_rounds(self) -> int:
        '''
        Get number of boosting rounds
        
        Returns:
        * int: number of rounds
        '''
        raise NotImplementedError
    
    def train(self) -> xgb.Booster:
        '''
        Train the model using leave-one-season-out cross-validation
        
        Returns:
        * xgb.Booster: trained model
        '''
        data = self.load_data()
        seasons = self.config.seasons
        ##  Perform LOSO cross-validation if multiple seasons
        if self.config.validation_strategy == "loso" and len(seasons) > 1:
            cv_results = []
            for holdout_season in seasons:
                if self.config.verbose:
                    print(f"Training with season {holdout_season} held out...")
                cv_model, cv_preds = self._train_single_fold(data, holdout_season)
                cv_results.append(cv_preds)
            self.cv_results = pd.concat(cv_results, ignore_index=True)
        ##  Train final model on all data
        if self.config.verbose:
            print("Training final model on all seasons...")
        features, labels, weights = self.prepare_features(data)
        self.trained_model = self._fit_model(features, labels, weights)
        return self.trained_model
    
    def _train_single_fold(
        self, 
        data: pd.DataFrame, 
        holdout_season: int
    ) -> tuple[xgb.Booster, pd.DataFrame]:
        '''
        Train model for a single CV fold
        
        Params:
        * data: pd.DataFrame - all training data
        * holdout_season: int - season to hold out for validation
        
        Returns:
        * tuple[xgb.Booster, pd.DataFrame]: trained model and predictions
        '''
        train_data = data[data['season'] != holdout_season]
        test_data = data[data['season'] == holdout_season]
        train_features, train_labels, train_weights = self.prepare_features(train_data)
        test_features, test_labels, _ = self.prepare_features(test_data)
        ##  Train model
        model = self._fit_model(train_features, train_labels, train_weights)
        ##  Get predictions
        predictions = self._predict(model, test_features)
        ##  Combine with test data for evaluation
        result = test_data[['season']].copy()
        result['label'] = test_labels.values  ## Use values to avoid index mismatch
        ##  Reset indices before concatenating to avoid NaN values
        result = result.reset_index(drop=True)
        predictions = predictions.reset_index(drop=True)
        result = pd.concat([result, predictions], axis=1)
        return model, result
    
    def _fit_model(
        self, 
        features: pd.DataFrame, 
        labels: pd.Series,
        weights: Optional[pd.Series] = None
    ) -> xgb.Booster:
        '''
        Fit XGBoost model
        
        Params:
        * features: pd.DataFrame - feature matrix
        * labels: pd.Series - target labels
        * weights: Optional[pd.Series] - sample weights
        
        Returns:
        * xgb.Booster: trained model
        '''
        dtrain = xgb.DMatrix(features, label=labels, weight=weights)
        params = self.get_xgb_params()
        num_rounds = self.get_num_rounds()
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_rounds,
            verbose_eval=self.config.verbose
        )
        return model
    
    @abstractmethod
    def _predict(self, model: xgb.Booster, features: pd.DataFrame) -> pd.DataFrame:
        '''
        Generate predictions from model
        
        Params:
        * model: xgb.Booster - trained model
        * features: pd.DataFrame - feature matrix
        
        Returns:
        * pd.DataFrame: predictions
        '''
        raise NotImplementedError
    
    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        '''
        Evaluate model performance using CV results
        
        Returns:
        * Dict[str, float]: evaluation metrics
        '''
        raise NotImplementedError
    
    def save_model(self, save_path: Path, metadata: ModelMetadata) -> None:
        '''
        Save trained model and metadata
        
        Params:
        * save_path: Path - path to save model
        * metadata: ModelMetadata - model metadata
        '''
        if not self.trained_model:
            raise ValueError("No trained model to save")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.trained_model.save_model(str(save_path))
        ##  Save metadata
        metadata_path = save_path.parent / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
    
    def save_training_run(
        self, 
        training_runs_dir: Path,
        metrics: Dict[str, Any],
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Path:
        '''
        Save training run record with timestamp
        
        Params:
        * training_runs_dir: Path - directory to save training run records
        * metrics: Dict[str, Any] - all metrics from evaluation including calibration, log loss, bins
        * additional_info: Optional[Dict[str, Any]] - any additional info to save (notes, etc.)
        
        Returns:
        * Path: path to saved training run file
        '''
        training_runs_dir.mkdir(parents=True, exist_ok=True)
        ##  Create filename with unix timestamp
        timestamp = int(time.time())
        filename = f"{timestamp}.json"
        filepath = training_runs_dir / filename
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
        ##  Gather all training info
        run_record = {
            'timestamp': timestamp,
            'config': {
                'seasons': convert_value(self.config.seasons),
                'validation_strategy': self.config.validation_strategy,
                'random_seed': self.config.random_seed
            },
            'xgb_params': convert_value(self.get_xgb_params()),
            'num_rounds': int(self.get_num_rounds()),
            'metrics': convert_value(metrics)
        }
        if additional_info:
            run_record['additional_info'] = convert_value(additional_info)
        ##  Save to file
        with open(filepath, 'w') as f:
            json.dump(run_record, f, indent=2)
        return filepath

