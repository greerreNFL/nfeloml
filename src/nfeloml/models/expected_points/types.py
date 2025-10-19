'''
Type definitions for Expected Points model
'''
from dataclasses import dataclass
from nfeloml.core.types import PredictionInput, PredictionOutput, TrainingConfig

@dataclass
class EPFeatures(PredictionInput):
    '''
    Input features for Expected Points model
    
    Params:
    * half_seconds_remaining: int - seconds remaining in half
    * yardline_100: int - yards to opponent endzone (0-100)
    * home: int - 1 if possession team is home, 0 otherwise
    * retractable: int - 1 if retractable roof, 0 otherwise
    * dome: int - 1 if dome, 0 otherwise
    * outdoors: int - 1 if outdoors, 0 otherwise
    * down: int - current down (1-4)
    * ydstogo: int - yards to go for first down
    * era: int - era indicator (0-4)
    * posteam_timeouts_remaining: int - timeouts remaining for possession team
    * defteam_timeouts_remaining: int - timeouts remaining for defense
    '''
    half_seconds_remaining: int
    yardline_100: int
    home: int
    retractable: int
    dome: int
    outdoors: int
    down: int
    ydstogo: int
    era: int
    posteam_timeouts_remaining: int
    defteam_timeouts_remaining: int

@dataclass
class EPPrediction(PredictionOutput):
    '''
    Output predictions from Expected Points model
    
    Params:
    * touchdown: float - probability of next score being touchdown for offense
    * opp_touchdown: float - probability of next score being touchdown for defense
    * field_goal: float - probability of next score being field goal for offense
    * opp_field_goal: float - probability of next score being field goal for defense
    * safety: float - probability of next score being safety for offense
    * opp_safety: float - probability of next score being safety for defense
    * no_score: float - probability of no score in the half
    '''
    touchdown: float
    opp_touchdown: float
    field_goal: float
    opp_field_goal: float
    safety: float
    opp_safety: float
    no_score: float
    
    def expected_points(self) -> float:
        '''
        Calculate expected points value
        
        Returns:
        * float: expected points
        '''
        return (
            7 * self.touchdown +
            -7 * self.opp_touchdown +
            3 * self.field_goal +
            -3 * self.opp_field_goal +
            2 * self.safety +
            -2 * self.opp_safety
        )

@dataclass
class EPTrainingConfig(TrainingConfig):
    '''
    Training configuration for Expected Points model
    
    Params:
    * nrounds: int - number of boosting rounds
    * eta: float - learning rate
    * gamma: float - minimum loss reduction for split
    * subsample: float - subsample ratio of training instances
    * colsample_bytree: float - subsample ratio of columns
    * max_depth: int - maximum tree depth
    * min_child_weight: int - minimum sum of instance weight in child
    '''
    nrounds: int = 525
    eta: float = 0.025
    gamma: float = 1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    max_depth: int = 5
    min_child_weight: int = 1

