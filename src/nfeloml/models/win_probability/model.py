'''
Win Probability model for inference
'''
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from typing import Optional
from datetime import datetime
from nfeloml.core.base_model import BaseModel
from nfeloml.core.types import ModelMetadata
from .types import WPFeatures, WPPrediction

class WinProbabilityModel(BaseModel):
    '''
    Win Probability model for predicting game outcomes
    '''
    
    def __init__(self, model_path: Optional[Path] = None, use_spread: bool = True):
        '''
        Initialize Win Probability model
        
        Params:
        * model_path: Optional[Path] - path to saved model file (if None, loads from package)
        * use_spread: bool - deprecated, model always uses all features including spread
        
        Note: The nflfastr WP model expects all 12 features including spread_time.
        If spread_time is not available, it will default to 0.
        '''
        self.use_spread = use_spread  # Kept for backward compatibility
        super().__init__(model_path)
    
    def _dict_to_metadata(self, metadata_dict: dict) -> ModelMetadata:
        '''
        Convert dictionary to ModelMetadata
        
        Params:
        * metadata_dict: dict - metadata dictionary
        
        Returns:
        * ModelMetadata: metadata object
        '''
        return ModelMetadata(
            model_name=metadata_dict['model_name'],
            version=metadata_dict['version'],
            trained_date=datetime.fromisoformat(metadata_dict['trained_date']),
            training_seasons=metadata_dict['training_seasons'],
            calibration_error=metadata_dict.get('calibration_error'),
            additional_metrics=metadata_dict.get('additional_metrics', {})
        )
    
    def predict(self, features: WPFeatures, include_probabilities: bool = False):
        '''
        Predict win probability for a play
        
        Params:
        * features: WPFeatures - input features for the play
        * include_probabilities: bool - if False (default), returns just win_probability float;
                                        if True, returns WPPrediction dataclass
        
        Returns:
        * float or WPPrediction: win probability value or dataclass
        '''
        if not self.xgb_model:
            raise ValueError("Model not loaded")
        ##  Convert features to DataFrame
        ##  IMPORTANT: Feature order must match nflfastr model training order
        feature_dict = {
            'receive_2h_ko': [features.receive_2h_ko],
            'spread_time': [features.spread_time if features.spread_time is not None else 0.0],
            'home': [features.home],
            'half_seconds_remaining': [features.half_seconds_remaining],
            'game_seconds_remaining': [features.game_seconds_remaining],
            'Diff_Time_Ratio': [features.diff_time_ratio],
            'score_differential': [features.score_differential],
            'down': [features.down],
            'ydstogo': [features.ydstogo],
            'yardline_100': [features.yardline_100],
            'posteam_timeouts_remaining': [features.posteam_timeouts_remaining],
            'defteam_timeouts_remaining': [features.defteam_timeouts_remaining]
        }
        df = pd.DataFrame(feature_dict)
        ##  Create DMatrix and predict
        dmatrix = xgb.DMatrix(df)
        pred = float(self.xgb_model.predict(dmatrix)[0])
        if not include_probabilities:
            return pred
        ##  Return dataclass
        return WPPrediction(win_probability=pred)
    
    def predict_df(self, df: pd.DataFrame, include_probabilities: bool = False) -> pd.DataFrame:
        '''
        Make predictions on a DataFrame of plays
        
        Params:
        * df: pd.DataFrame - dataframe with columns: receive_2h_ko, home, half_seconds_remaining,
                            game_seconds_remaining, Diff_Time_Ratio, score_differential, down,
                            ydstogo, yardline_100, posteam_timeouts_remaining, 
                            defteam_timeouts_remaining
                            Note: spread_time will be auto-added as 0 if not present
        * include_probabilities: bool - (not used for WP, kept for API consistency)
        
        Returns:
        * pd.DataFrame: original df with added win_probability column
        '''
        if not self.xgb_model:
            raise ValueError("Model not loaded")
        ##  Select feature columns in nflfastr model order
        ##  spread_time must be 2nd feature (after receive_2h_ko)
        if 'spread_time' not in df.columns:
            ##  If spread_time missing, add it as 0 for compatibility
            df = df.copy()
            df['spread_time'] = 0.0
        
        feature_cols = [
            'receive_2h_ko', 'spread_time', 'home', 'half_seconds_remaining', 'game_seconds_remaining',
            'Diff_Time_Ratio', 'score_differential', 'down', 'ydstogo', 'yardline_100',
            'posteam_timeouts_remaining', 'defteam_timeouts_remaining'
        ]
        X = df[feature_cols]
        ##  Predict
        dmatrix = xgb.DMatrix(X)
        preds = self.xgb_model.predict(dmatrix)
        ##  Add prediction to original dataframe
        result = df.copy()
        result['win_probability'] = preds
        return result

