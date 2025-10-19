'''
Type definitions for Win Probability model
'''
from dataclasses import dataclass
from typing import Optional
from nfeloml.core.types import PredictionInput, PredictionOutput, TrainingConfig

@dataclass
class WPFeatures(PredictionInput):
    '''
    Input features for Win Probability model
    
    Params:
    * receive_2h_ko: int - 1 if team will receive 2nd half kickoff, 0 otherwise
    * home: int - 1 if possession team is home, 0 otherwise
    * half_seconds_remaining: int - seconds remaining in current half
    * game_seconds_remaining: int - seconds remaining in game
    * diff_time_ratio: float - score differential adjusted for time remaining
    * score_differential: int - current score differential (positive = winning)
    * down: int - current down (1-4)
    * ydstogo: int - yards to go for first down
    * yardline_100: int - yards to opponent endzone
    * posteam_timeouts_remaining: int - timeouts remaining for possession team
    * defteam_timeouts_remaining: int - timeouts remaining for defense
    * spread_time: Optional[float] - vegas spread adjusted for time (defaults to 0 if not provided)
    '''
    receive_2h_ko: int
    home: int
    half_seconds_remaining: int
    game_seconds_remaining: int
    diff_time_ratio: float
    score_differential: int
    down: int
    ydstogo: int
    yardline_100: int
    posteam_timeouts_remaining: int
    defteam_timeouts_remaining: int
    spread_time: Optional[float] = None

@dataclass
class WPPrediction(PredictionOutput):
    '''
    Output prediction from Win Probability model
    
    Params:
    * win_probability: float - probability of possession team winning (0-1)
    '''
    win_probability: float

@dataclass
class WPTrainingConfig(TrainingConfig):
    '''
    Training configuration for Win Probability model
    
    Params:
    * nrounds: int - number of boosting rounds
    * eta: float - learning rate
    * gamma: float - minimum loss reduction for split
    * subsample: float - subsample ratio of training instances
    * colsample_bytree: float - subsample ratio of columns
    * max_depth: int - maximum tree depth
    * min_child_weight: int - minimum sum of instance weight in child
    * use_spread: bool - whether to include Vegas spread in model
    '''
    nrounds: int = 65
    eta: float = 0.2
    gamma: float = 0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    max_depth: int = 4
    min_child_weight: int = 1
    use_spread: bool = False

@dataclass
class WPSpreadTrainingConfig(WPTrainingConfig):
    '''
    Training configuration for Win Probability model with spread
    '''
    nrounds: int = 534
    eta: float = 0.05
    gamma: float = 0.79012017
    subsample: float = 0.9224245
    colsample_bytree: float = 5/12
    max_depth: int = 5
    min_child_weight: int = 7
    use_spread: bool = True

