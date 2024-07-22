import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error





def read_dataframe(fileurl):
    df = pd.read_parquet(fileurl)

    df['duration']= df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td : td.total_seconds() / 60)
    df= df[(df.duration>=1) & (df.duration <=60)]

    

    categorical=['PULocationID', 'DOLocationID']
    df[categorical]= df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '-' + df['DOLocationID']

    return df

def training_model(df_train, df_val):
    #categorical=['PULocationID', 'DOLocationID']
    categorical= ['PU_DO']


    dv = DictVectorizer()
    train_dicts= df_train[categorical + ['trip_distance']].to_dict(orient='records')

    X_train = dv.fit_transform(train_dicts)
    y_train = df_train['duration'].values

    val_dicts= df_val[categorical + ['trip_distance']].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_val = df_val['duration'].values


    lr = Ridge()
    lr.fit(X_train,y_train)

    y_pred = lr.predict(X_val)

    with open('Ridge.bin', 'wb') as f:
        pickle.dump((dv, lr), f)
        
    return mean_squared_error(y_val, y_pred, squared=False)

    

if __name__ == '__main__':
    df_train = read_dataframe('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-04.parquet')
    df_val = read_dataframe('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-03.parquet')
    print(training_model(df_train, df_val))
























