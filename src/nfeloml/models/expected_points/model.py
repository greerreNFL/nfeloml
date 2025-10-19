'''
Expected Points model for inference
'''
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from typing import Optional
from datetime import datetime
from nfeloml.core.base_model import BaseModel
from nfeloml.core.types import ModelMetadata
from .types import EPFeatures, EPPrediction

class ExpectedPointsModel(BaseModel):
    '''
    Expected Points model for making predictions on NFL plays
    '''
    
    def __init__(self, model_path: Optional[Path] = None):
        '''
        Initialize Expected Points model
        
        Params:
        * model_path: Optional[Path] - path to saved model file (if None, loads from package)
        '''
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
    
    def predict(self, features: EPFeatures, include_probabilities: bool = False):
        '''
        Predict expected points for a play
        
        Params:
        * features: EPFeatures - input features for the play
        * include_probabilities: bool - if False (default), returns just expected_points float;
                                        if True, returns full EPPrediction dataclass
        
        Returns:
        * float or EPPrediction: expected points value or full probabilities
        '''
        if not self.xgb_model:
            raise ValueError("Model not loaded")
        ##  Convert features to DataFrame
        feature_dict = {
            'half_seconds_remaining': [features.half_seconds_remaining],
            'yardline_100': [features.yardline_100],
            'home': [features.home],
            'retractable': [features.retractable],
            'dome': [features.dome],
            'outdoors': [features.outdoors],
            'ydstogo': [features.ydstogo],
            'era0': [1 if features.era == 0 else 0],
            'era1': [1 if features.era == 1 else 0],
            'era2': [1 if features.era == 2 else 0],
            'era3': [1 if features.era == 3 else 0],
            'era4': [1 if features.era == 4 else 0],
            'down1': [1 if features.down == 1 else 0],
            'down2': [1 if features.down == 2 else 0],
            'down3': [1 if features.down == 3 else 0],
            'down4': [1 if features.down == 4 else 0],
            'posteam_timeouts_remaining': [features.posteam_timeouts_remaining],
            'defteam_timeouts_remaining': [features.defteam_timeouts_remaining]
        }
        df = pd.DataFrame(feature_dict)
        ##  Create DMatrix and predict
        dmatrix = xgb.DMatrix(df)
        preds = self.xgb_model.predict(dmatrix)[0]
        ##  Calculate expected points
        ep = (
            7 * float(preds[0]) -
            7 * float(preds[1]) +
            3 * float(preds[2]) -
            3 * float(preds[3]) +
            2 * float(preds[4]) -
            2 * float(preds[5])
        )
        if not include_probabilities:
            return ep
        ##  Return full dataclass
        return EPPrediction(
            touchdown=float(preds[0]),
            opp_touchdown=float(preds[1]),
            field_goal=float(preds[2]),
            opp_field_goal=float(preds[3]),
            safety=float(preds[4]),
            opp_safety=float(preds[5]),
            no_score=float(preds[6])
        )
    
    def predict_df(self, df: pd.DataFrame, include_probabilities: bool = False, include_epa: bool = False) -> pd.DataFrame:
        '''
        Make predictions on a DataFrame of plays
        
        Params:
        * df: pd.DataFrame - dataframe with columns: half_seconds_remaining, yardline_100, 
                            home, retractable, dome, outdoors, down, ydstogo, era,
                            posteam_timeouts_remaining, defteam_timeouts_remaining
        * include_probabilities: bool - if False (default), adds only expected_points column;
                                        if True, adds all probability columns
        * include_epa: bool - if True, also calculates EPA (Expected Points Added) for each play
        
        Returns:
        * pd.DataFrame: original df with added expected_points column (and probability/EPA columns if requested)
        '''
        if not self.xgb_model:
            raise ValueError("Model not loaded")
        ##  Create feature columns if they don't exist using shared utility
        from .utils import add_ep_features
        feature_df = add_ep_features(df)
        ##  Select feature columns
        feature_cols = [
            'half_seconds_remaining', 'yardline_100', 'home', 'retractable', 'dome',
            'outdoors', 'ydstogo', 'era0', 'era1', 'era2', 'era3', 'era4',
            'down1', 'down2', 'down3', 'down4', 'posteam_timeouts_remaining',
            'defteam_timeouts_remaining'
        ]
        X = feature_df[feature_cols]
        ##  Predict
        dmatrix = xgb.DMatrix(X)
        preds = self.xgb_model.predict(dmatrix)
        ##  Add predictions to original dataframe
        result = df.copy()
        ##  Calculate expected points
        result['expected_points'] = (
            7 * preds[:, 0] -
            7 * preds[:, 1] +
            3 * preds[:, 2] -
            3 * preds[:, 3] +
            2 * preds[:, 4] -
            2 * preds[:, 5]
        )
        ##  Identify plays with missing critical features - these should have NaN EP
        ##  Critical raw features that must be present for valid EP calculation
        invalid_mask = (
            df['down'].isna() |
            df['yardline_100'].isna() |
            df['ydstogo'].isna() |
            df['half_seconds_remaining'].isna() |
            df['posteam_timeouts_remaining'].isna() |
            df['defteam_timeouts_remaining'].isna()
        )
        ##  Set EP to NaN for plays with missing critical features
        result.loc[invalid_mask, 'expected_points'] = np.nan
        ##  Add probability columns if requested
        if include_probabilities:
            result['ep_touchdown'] = preds[:, 0]
            result['ep_opp_touchdown'] = preds[:, 1]
            result['ep_field_goal'] = preds[:, 2]
            result['ep_opp_field_goal'] = preds[:, 3]
            result['ep_safety'] = preds[:, 4]
            result['ep_opp_safety'] = preds[:, 5]
            result['ep_no_score'] = preds[:, 6]
            ##  Also set probabilities to NaN for invalid plays
            prob_cols = ['ep_touchdown', 'ep_opp_touchdown', 'ep_field_goal', 
                        'ep_opp_field_goal', 'ep_safety', 'ep_opp_safety', 'ep_no_score']
            for col in prob_cols:
                result.loc[invalid_mask, col] = np.nan
        ##  Calculate EPA if requested
        if include_epa:
            from ...utils.epa import calculate_epa
            result = calculate_epa(result)
        return result

