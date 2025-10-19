# nfeloml

**Portable machine learning models for NFL analytics**

`nfeloml` provides distributed python ports of nflfastr models so they can be used in web services or other python based applications.

## Features

- **Expected Points (EP)** - Predict the next scoring outcome and expected points for any play
- **Win Probability (WP)** - Predict the probability of the possession team winning the game
- **Type-Safe** - Full type hints and dataclass-based inputs/outputs
- **Portable** - Pre-trained models bundled with package, auto-loaded on import
- **DataFrame-Native** - Bulk predictions on entire DataFrames


## Installation

```bash
pip install nfeloml
```

**For training models** (not required for inference):
```bash
pip install nfeloml[training]
```

This installs `nfelodcm` which is only needed for training, not for using the pre-trained models.

## Quick Start

### Expected Points - DataFrame Inference

The most common use case is enriching a DataFrame of plays with predictions:

```python
from nfeloml import ExpectedPointsModel
import nfelodcm as dcm

##  Load the model (automatically loads from package)
model = ExpectedPointsModel()

##  Load some plays - as an example, this loads nflfastr, which already has EPA values
##  though it can be used with any df that has compatible columns
db = dcm.load(['pbp'])
plays = db['pbp'].copy()

##  Add EP predictions to the entire DataFrame (optionally with EPA)
enriched = model.predict_df(plays, include_epa=True)

##  Now you have: expected_points and epa (EPA auto-calculated)
print(enriched[['desc', 'expected_points', 'epa']].head())
```

### Expected Points - Single Play

For type-safe single predictions:

```python
from nfeloml import ExpectedPointsModel, EPFeatures

model = ExpectedPointsModel()

features = EPFeatures(
    half_seconds_remaining=1800,
    yardline_100=75,
    home=1,
    retractable=0,
    dome=0,
    outdoors=1,
    down=1,
    ydstogo=10,
    era=4,
    posteam_timeouts_remaining=3,
    defteam_timeouts_remaining=3
)

##  Simple usage - returns float
ep = model.predict(features)
print(f"Expected Points: {ep:.2f}")

##  Full prediction with probabilities - returns EPPrediction object
prediction = model.predict(features, include_probabilities=True)
print(f"Expected Points: {prediction.expected_points():.2f}")
print(f"TD Probability: {prediction.touchdown:.1%}")
```

### Win Probability - DataFrame Inference

```python
from nfeloml import WinProbabilityModel
import nfelodcm as dcm

##  Load model
model = WinProbabilityModel()

##  Load plays
db = dcm.load(['pbp'])
plays = db['pbp'].copy()

##  Add WP predictions
enriched = model.predict_df(plays)

##  Now you have: win_probability, again note that these already exist in nflfastr
print(enriched[['desc', 'win_probability']].head())
```

### Win Probability - Single Play

```python
from nfeloml import WinProbabilityModel, WPFeatures
import numpy as np

model = WinProbabilityModel()

features = WPFeatures(
    receive_2h_ko=1,
    home=1,
    half_seconds_remaining=300,
    game_seconds_remaining=2100,
    diff_time_ratio=7 * np.exp(4 * (3600 - 2100) / 3600),
    score_differential=7,
    down=2,
    ydstogo=5,
    yardline_100=45,
    posteam_timeouts_remaining=2,
    defteam_timeouts_remaining=3
)

prediction = model.predict(features)
print(f"Win Probability: {prediction.win_probability:.1%}")
```

## Calculating EPA (Expected Points Added)

EPA can be calculated automatically when generating EP predictions:

```python
from nfeloml import ExpectedPointsModel
import nfelodcm as dcm

model = ExpectedPointsModel()

##  Load data
db = dcm.load(['pbp'])
plays = db['pbp'].copy()

##  Add EP and EPA in one call
plays = model.predict_df(plays, include_epa=True)

##  Now you have both EP and EPA!
print(plays[['desc', 'expected_points', 'epa']].head())
```

Alternatively, you can calculate EPA separately on data that already has EP:

```python
from nfeloml import calculate_epa

##  If you already have expected_points in your dataframe
plays = calculate_epa(plays)
```

### How EPA is Calculated

EPA measures the change in expected points from the start to the end of a play:

- **Regular plays**: EPA = EP_end - EP_start
- **Scoring plays**: 
  - Touchdown: EP_end = 7
  - Field Goal: EP_end = 3
  - Safety: EP_end = -2 (for offense)
- **Possession changes**: When the next play has a different `posteam`, EP is negated (opponent's perspective)
- **Invalid plays**: Plays without required features result in null EPA

The function automatically:
- Skips to the next valid play with EP (ignoring timeouts, announcements, etc.)
- Handles scoring plays
- Accounts for possession changes

## Model Training

The package provides a complete training pipeline. Models are trained using data from `nfelodcm`.

**Note:** Training requires the optional `nfelodcm` dependency:
```bash
pip install nfeloml[training]
```

### Training Expected Points Model

```python
from nfeloml.models.expected_points import EPTrainer, EPDataLoader, EPTrainingConfig
from nfeloml.core.types import ModelMetadata
from pathlib import Path
from datetime import datetime

##  Configure training
config = EPTrainingConfig(
    seasons=list(range(2000, 2024)),
    validation_strategy="loso",
    random_seed=2013
)

##  Initialize data loader and trainer
data_loader = EPDataLoader()
trainer = EPTrainer(config, data_loader)

##  Train the model
model = trainer.train()

##  Evaluate
metrics = trainer.evaluate()
print(f"Calibration Error: {metrics['calibration_error']['overall']:.4f}")

##  Save the trained model (saves to package directory automatically)
metadata = ModelMetadata(
    model_name="ExpectedPoints",
    version="1.0.0",
    trained_date=datetime.now(),
    training_seasons=config.seasons,
    calibration_error=metrics['calibration_error']['overall']
)

package_dir = Path(__file__).parent / 'trained_models'
trainer.save_model(package_dir / 'ep_model.ubj', metadata)
```

### Training Win Probability Model

```python
from nfeloml.models.win_probability import WPTrainer, WPDataLoader, WPTrainingConfig

##  Configure training
config = WPTrainingConfig(
    seasons=list(range(2000, 2024)),
    use_spread=False  ##  Set to True for spread-adjusted model
)

##  Train
data_loader = WPDataLoader()
trainer = WPTrainer(config, data_loader)
model = trainer.train()

##  Evaluate
metrics = trainer.evaluate()
print(f"Calibration Error: {metrics['overall']:.4f}")
```

## Layout
```
nfeloml/
├── core/                   # Shared abstractions
│   ├── base_model.py       # Model abstraction
│   ├── base_trainer.py     # Training abstraction
│   ├── base_data_loader.py # Data loading abstraction
│   └── types.py            # Common types
├── models/
│   ├── expected_points/    # EP model implementation
│   └── win_probability/    # WP model implementation
└── utils/                  # Utilities
    └── validation.py       # Data validation
```

Each model consists of:
- **types.py** - Dataclass definitions for inputs/outputs
- **data_loader.py** - Data fetching via nfelodcm
- **trainer.py** - Model training with cross-validation
- **model.py** - Inference interface

## Data Source

Training data comes from `nfelodcm`, which provides:
- Automatic caching and freshness checks
- Simple interface: `dcm.load(['pbp'])`

## Models

### Expected Points (EP)

Predicts the expected points value of the next scoring play. Uses a 7-class XGBoost model to predict probabilities for:
- Touchdown (7 points)
- Opponent Touchdown (-7 points)
- Field Goal (3 points)
- Opponent Field Goal (-3 points)
- Safety (2 points)
- Opponent Safety (-2 points)
- No Score (0 points)

**Features:**
- Game situation (down, distance, yardline)
- Time remaining in half
- Timeouts remaining
- Home field advantage
- Stadium type (dome, outdoors, retractable)
- Era adjustments for rule changes

### Win Probability (WP)

Predicts the probability that the possession team will win the game. Uses binary XGBoost classification.

**Features:**
- Score differential
- Time remaining (game and half)
- Game situation (down, distance, yardline)
- Timeouts remaining
- Which team receives 2nd half kickoff
- Home field advantage
- Optional: Vegas point spread (spread-adjusted model)

## Development

### Adding New Models

To add a new model (e.g., Completion Probability):

1. Create directory: `src/nfeloml/models/completion_probability/`
2. Implement `types.py` with input/output dataclasses
3. Implement `data_loader.py` extending `BaseDataLoader`
4. Implement `trainer.py` extending `BaseTrainer`
5. Implement `model.py` extending `BaseModel` with `get_model_filename()` and `predict_df()`
6. Export in `__init__.py` files

The abstract base classes handle common functionality like LOSO cross-validation, model persistence, and evaluation.

## File Formats

Models are stored in XGBoost's native binary format (`.ubj`) for optimal size and performance. Metadata is stored in JSON alongside the model.


## Credits

The models are intended to be ports of the existing nflfastr models developed by Ben Baldwin ([@benbbaldwin](https://twitter.com/benbbaldwin)) and thus should be credited to him and the nflfastr team.
