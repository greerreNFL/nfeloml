'''
Trainer for Win Probability model
'''
import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Dict, Any
from pathlib import Path
from nfeloml.core.base_trainer import BaseTrainer
from .data_loader import WPDataLoader
from .types import WPTrainingConfig

class WPTrainer(BaseTrainer):
    '''
    Trainer for Win Probability model using XGBoost binary classification
    '''
    
    def __init__(self, config: WPTrainingConfig, data_loader: WPDataLoader):
        '''
        Initialize WP trainer
        
        Params:
        * config: WPTrainingConfig - training configuration
        * data_loader: WPDataLoader - data loader instance
        '''
        super().__init__(config, data_loader)
        self.wp_config = config
    
    def load_data(self) -> pd.DataFrame:
        '''
        Load and prepare WP training data
        
        Returns:
        * pd.DataFrame: prepared training data
        '''
        return self.data_loader.load_data()
    
    def prepare_features(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, None]:
        '''
        Prepare features and labels for WP model
        
        Params:
        * data: pd.DataFrame - training data
        
        Returns:
        * tuple[pd.DataFrame, pd.Series, None]: feature matrix, labels, and no weights
        '''
        ##  Define feature columns in nflfastr model order
        ##  IMPORTANT: spread_time must be 2nd feature (after receive_2h_ko)
        if self.wp_config.use_spread:
            feature_cols = [
                'receive_2h_ko',
                'spread_time',  # 2nd position to match nflfastr
                'home',
                'half_seconds_remaining',
                'game_seconds_remaining',
                'Diff_Time_Ratio',
                'score_differential',
                'down',
                'ydstogo',
                'yardline_100',
                'posteam_timeouts_remaining',
                'defteam_timeouts_remaining'
            ]
        else:
            ##  Non-spread model (11 features)
            feature_cols = [
                'receive_2h_ko',
                'home',
                'half_seconds_remaining',
                'game_seconds_remaining',
                'Diff_Time_Ratio',
                'score_differential',
                'down',
                'ydstogo',
                'yardline_100',
                'posteam_timeouts_remaining',
                'defteam_timeouts_remaining'
            ]
        ##  Extract features
        X = data[feature_cols].copy()
        ##  Extract labels
        y = data['label']
        ##  WP model doesn't use weights
        return X, y, None
    
    def get_xgb_params(self) -> Dict[str, Any]:
        '''
        Get XGBoost parameters for WP model
        
        Returns:
        * Dict[str, Any]: XGBoost parameters
        '''
        params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'eta': self.wp_config.eta,
            'gamma': self.wp_config.gamma,
            'subsample': self.wp_config.subsample,
            'colsample_bytree': self.wp_config.colsample_bytree,
            'max_depth': self.wp_config.max_depth,
            'min_child_weight': self.wp_config.min_child_weight,
            'seed': self.config.random_seed
        }
        ##  Add monotone constraints for spread model
        if self.wp_config.use_spread:
            ##  Monotone constraints matching nflfastr feature order:
            ##  0=receive_2h_ko, 1=spread_time, 2=home, 3=half_sec, 4=game_sec, 5=Diff_Time_Ratio,
            ##  6=score_diff, 7=down, 8=ydstogo, 9=yardline_100, 10=posteam_to, 11=defteam_to
            params['monotone_constraints'] = "(0, -1, 0, 0, 0, 0, 1, 1, -1, -1, -1, 1)"
        return params
    
    def get_num_rounds(self) -> int:
        '''
        Get number of boosting rounds
        
        Returns:
        * int: number of rounds
        '''
        return self.wp_config.nrounds
    
    def _train_single_fold(
        self,
        data: pd.DataFrame,
        holdout_season: int
    ) -> tuple[xgb.Booster, pd.DataFrame]:
        '''
        Train model for a single CV fold (override to include qtr in results)
        
        Params:
        * data: pd.DataFrame - all training data
        * holdout_season: int - season to hold out for validation
        
        Returns:
        * tuple[xgb.Booster, pd.DataFrame]: trained model and predictions with qtr
        '''
        train_data = data[data['season'] != holdout_season]
        test_data = data[data['season'] == holdout_season]
        train_features, train_labels, train_weights = self.prepare_features(train_data)
        test_features, test_labels, _ = self.prepare_features(test_data)
        ##  Train model
        model = self._fit_model(train_features, train_labels, train_weights)
        ##  Get predictions
        predictions = self._predict(model, test_features)
        ##  Combine with test data for evaluation (include qtr for quarter-based calibration)
        result = test_data[['season', 'qtr']].copy()
        result['label'] = test_labels.values  ## Use values to avoid index mismatch
        ##  Reset indices before concatenating to avoid NaN values
        result = result.reset_index(drop=True)
        predictions = predictions.reset_index(drop=True)
        result = pd.concat([result, predictions], axis=1)
        return model, result
    
    def _predict(self, model: xgb.Booster, features: pd.DataFrame) -> pd.DataFrame:
        '''
        Generate predictions from WP model
        
        Params:
        * model: xgb.Booster - trained model
        * features: pd.DataFrame - feature matrix
        
        Returns:
        * pd.DataFrame: predictions with wp column
        '''
        dtest = xgb.DMatrix(features)
        preds = model.predict(dtest)
        ##  Convert to DataFrame
        pred_df = pd.DataFrame({'wp': preds})
        return pred_df
    
    def evaluate(self) -> Dict[str, Any]:
        '''
        Evaluate WP model with calibration error by quarter, log loss, and accuracy
        
        Returns:
        * Dict[str, Any]: evaluation metrics including calibration error, log loss, accuracy, and bins
        '''
        if self.cv_results is None:
            raise ValueError("No CV results available. Run train() first.")
        
        ##  Calculate calibration error by quarter (matching nflfastR approach)
        calibration_errors = {}
        calibration_bins = {}
        n_wins_by_qtr = {}
        
        for qtr in [1, 2, 3, 4]:
            qtr_data = self.cv_results[self.cv_results['qtr'] == qtr].copy()
            if len(qtr_data) == 0:
                continue
            
            ##  Bin predictions
            qtr_data['bin_pred_prob'] = (qtr_data['wp'] / 0.05).round() * 0.05
            
            ##  Calculate calibration metrics per bin
            grouped = qtr_data.groupby('bin_pred_prob').agg(
                n_plays=('label', 'count'),
                n_wins=('label', 'sum')
            )
            grouped['bin_actual_prob'] = grouped['n_wins'] / grouped['n_plays']
            grouped['cal_diff'] = abs(grouped.index - grouped['bin_actual_prob'])
            
            ##  Weighted calibration error for this quarter
            cal_error = np.average(grouped['cal_diff'], weights=grouped['n_plays'])
            calibration_errors[f'Q{qtr}'] = cal_error
            n_wins_by_qtr[f'Q{qtr}'] = grouped['n_wins'].sum()
            
            ##  Store bins for visualization
            calibration_bins[f'Q{qtr}'] = {
                f"{bin_val:.2f}": {
                    'n_plays': int(row['n_plays']),
                    'n_wins': int(row['n_wins']),
                    'actual': float(row['bin_actual_prob']),
                    'predicted': float(bin_val)
                }
                for bin_val, row in grouped.iterrows()
            }
        
        ##  Overall weighted calibration error (weighted by n_wins per quarter)
        weights = np.array([n_wins_by_qtr[k] for k in calibration_errors.keys()])
        errors = np.array([calibration_errors[k] for k in calibration_errors.keys()])
        overall_cal_error = np.average(errors, weights=weights)
        
        ##  Calculate log loss
        preds = self.cv_results['wp'].values
        labels = self.cv_results['label'].values
        preds_clipped = np.clip(preds, 1e-15, 1 - 1e-15)
        log_loss = -np.mean(labels * np.log(preds_clipped) + (1 - labels) * np.log(1 - preds_clipped))
        
        ##  Calculate accuracy
        predicted_labels = (preds >= 0.5).astype(int)
        accuracy = np.mean(predicted_labels == labels)
        
        ##  Compile all metrics
        metrics = {
            'calibration_error': {
                'overall': float(overall_cal_error),
                'by_quarter': {k: float(v) for k, v in calibration_errors.items()}
            },
            'log_loss': float(log_loss),
            'accuracy': float(accuracy),
            'calibration_bins': calibration_bins
        }
        
        return metrics

