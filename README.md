## README

# Taxi Trip Duration Prediction

This project predicts the duration of taxi trips using machine learning models. It reads trip data, processes it, and trains a model to predict the trip duration based on various features. The implementation uses Ridge regression and stores the trained model using `pickle`. Additionally, experimental tracking and model registry are handled using MLflow.

## Features

- **Data Reading**: Reads trip data from parquet files.
- **Data Processing**: Processes the data to compute trip duration and create relevant features.
- **Model Training**: Trains a Ridge regression model to predict trip duration.
- **Model Persistence**: Saves the trained model using `pickle`.
- **MLflow Integration**: Tracks experiments and manages models using MLflow.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/taxi-trip-duration-prediction.git
   cd taxi-trip-duration-prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Reading and Processing

The `read_dataframe` function reads trip data from a given URL, computes trip duration, and creates relevant features.

```python
def read_dataframe(fileurl):
    df = pd.read_parquet(fileurl)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '-' + df['DOLocationID']

    return df
```

### Model Training

The `training_model` function trains a Ridge regression model on the processed data and saves the trained model.


## MLflow Integration

This project uses MLflow for experiment tracking and model registry. Please make sure that MLflow is set up and running before starting your experiments.


## Dependencies

- pandas
- seaborn
- matplotlib
- sklearn
- pickle
- mlflow


Feel free to customize the content based on your specific implementation and details.
