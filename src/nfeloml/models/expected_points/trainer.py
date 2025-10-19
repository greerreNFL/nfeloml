'''
Trainer for Expected Points model
'''
import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Dict, Any
from pathlib import Path
from nfeloml.core.base_trainer import BaseTrainer
from nfeloml.core.types import TrainingConfig
from .data_loader import EPDataLoader
from .types import EPTrainingConfig

class EPTrainer(BaseTrainer):
    '''
    Trainer for Expected Points model using XGBoost multi-class classification
    '''
    
    def __init__(self, config: EPTrainingConfig, data_loader: EPDataLoader):
        '''
        Initialize EP trainer
        
        Params:
        * config: EPTrainingConfig - training configuration
        * data_loader: EPDataLoader - data loader instance
        '''
        super().__init__(config, data_loader)
        self.ep_config = config
    
    def load_data(self) -> pd.DataFrame:
        '''
        Load and prepare EP training data
        
        Returns:
        * pd.DataFrame: prepared training data
        '''
        return self.data_loader.load_data()
    
    def prepare_features(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        '''
        Prepare features, labels, and weights for EP model
        
        Params:
        * data: pd.DataFrame - training data
        
        Returns:
        * tuple[pd.DataFrame, pd.Series, pd.Series]: feature matrix, labels, and weights
        '''
        ##  Define feature columns
        feature_cols = [
            'half_seconds_remaining',
            'yardline_100',
            'home',
            'retractable',
            'dome',
            'outdoors',
            'ydstogo',
            'era0', 'era1', 'era2', 'era3', 'era4',
            'down1', 'down2', 'down3', 'down4',
            'posteam_timeouts_remaining',
            'defteam_timeouts_remaining'
        ]
        ##  Extract features
        X = data[feature_cols].copy()
        ##  Create numeric labels from Next_Score_Half
        ##  0: Touchdown, 1: Opp_Touchdown, 2: Field_Goal, 
        ##  3: Opp_Field_Goal, 4: Safety, 5: Opp_Safety, 6: No_Score
        label_map = {
            'Touchdown': 0,
            'Opp_Touchdown': 1,
            'Field_Goal': 2,
            'Opp_Field_Goal': 3,
            'Safety': 4,
            'Opp_Safety': 5,
            'No_Score': 6
        }
        y = data['Next_Score_Half'].map(label_map)
        ##  Extract weights
        weights = data['Total_W_Scaled']
        return X, y, weights
    
    def get_xgb_params(self) -> Dict[str, Any]:
        '''
        Get XGBoost parameters for EP model
        
        Returns:
        * Dict[str, Any]: XGBoost parameters
        '''
        return {
            'booster': 'gbtree',
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'num_class': 7,
            'eta': self.ep_config.eta,
            'gamma': self.ep_config.gamma,
            'subsample': self.ep_config.subsample,
            'colsample_bytree': self.ep_config.colsample_bytree,
            'max_depth': self.ep_config.max_depth,
            'min_child_weight': self.ep_config.min_child_weight,
            'seed': self.config.random_seed
        }
    
    def get_num_rounds(self) -> int:
        '''
        Get number of boosting rounds
        
        Returns:
        * int: number of rounds
        '''
        return self.ep_config.nrounds
    
    def _predict(self, model: xgb.Booster, features: pd.DataFrame) -> pd.DataFrame:
        '''
        Generate predictions from EP model
        
        Params:
        * model: xgb.Booster - trained model
        * features: pd.DataFrame - feature matrix
        
        Returns:
        * pd.DataFrame: predictions with columns for each outcome
        '''
        dtest = xgb.DMatrix(features)
        preds = model.predict(dtest)
        ##  Convert to DataFrame with proper column names
        pred_df = pd.DataFrame(preds, columns=[
            'Touchdown', 'Opp_Touchdown', 'Field_Goal', 
            'Opp_Field_Goal', 'Safety', 'Opp_Safety', 'No_Score'
        ])
        return pred_df
    
    def evaluate(self) -> Dict[str, Any]:
        '''
        Evaluate EP model with multiple metrics
        
        Returns:
        * Dict[str, Any]: evaluation metrics including calibration error, log loss, accuracy, and bins
        '''
        if self.cv_results is None:
            raise ValueError("No CV results available. Run train() first.")
        
        outcome_cols = ['Touchdown', 'Opp_Touchdown', 'Field_Goal', 
                       'Opp_Field_Goal', 'Safety', 'Opp_Safety', 'No_Score']
        
        ##  Calculate calibration error and bins for each outcome type
        calibration_errors = {}
        calibration_bins = {}
        n_scoring_events = {}
        
        for outcome in outcome_cols:
            if outcome in self.cv_results.columns:
                ##  Bin predictions
                binned = self.cv_results.copy()
                binned['bin_pred_prob'] = (binned[outcome] / 0.05).round() * 0.05
                binned['correct'] = (binned['label'] == outcome_cols.index(outcome)).astype(int)
                ##  Calculate calibration error
                grouped = binned.groupby('bin_pred_prob').agg(
                    n_plays=('correct', 'count'),
                    n_outcome=('correct', 'sum')
                )
                grouped['bin_actual_prob'] = grouped['n_outcome'] / grouped['n_plays']
                grouped['cal_diff'] = abs(grouped.index - grouped['bin_actual_prob'])
                ##  Weighted calibration error (weighted by n_plays per bin)
                cal_error = np.average(grouped['cal_diff'], weights=grouped['n_plays'])
                calibration_errors[outcome] = cal_error
                ##  Track total scoring events for this outcome type
                n_scoring_events[outcome] = grouped['n_outcome'].sum()
                ##  Store bins for visualization
                calibration_bins[outcome] = {
                    f"{bin_val:.2f}": {
                        'n_plays': int(row['n_plays']),
                        'n_outcome': int(row['n_outcome']),
                        'actual': float(row['bin_actual_prob']),
                        'predicted': float(bin_val)
                    }
                    for bin_val, row in grouped.iterrows()
                }
        
        ##  Overall weighted calibration error
        weights = np.array([n_scoring_events[k] for k in calibration_errors.keys()])
        errors = np.array([calibration_errors[k] for k in calibration_errors.keys()])
        overall_cal_error = np.average(errors, weights=weights)
        
        ##  Calculate log loss (cross-entropy)
        pred_matrix = self.cv_results[outcome_cols].values
        true_labels = self.cv_results['label'].values.astype(int)
        ##  Clip predictions to avoid log(0)
        pred_matrix = np.clip(pred_matrix, 1e-15, 1 - 1e-15)
        ##  Get probability of true class for each sample
        true_class_probs = pred_matrix[np.arange(len(true_labels)), true_labels]
        log_loss = -np.mean(np.log(true_class_probs))
        
        ##  Calculate overall accuracy
        predicted_labels = np.argmax(pred_matrix, axis=1)
        accuracy = np.mean(predicted_labels == true_labels)
        
        ##  Compile all metrics
        metrics = {
            'calibration_error': {
                'overall': float(overall_cal_error),
                'by_outcome': {k: float(v) for k, v in calibration_errors.items()}
            },
            'log_loss': float(log_loss),
            'accuracy': float(accuracy),
            'calibration_bins': calibration_bins
        }
        
        return metrics

