import argparse
import pickle
import pandas as pd
import os

# Ensure the 'output' directory exists
os.makedirs("output", exist_ok=True)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def main(year, month):
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    # Load data for given year and month
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df = read_data(url)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    # Print the mean predicted duration
    mean_pred = y_pred.mean()
    print(f"Mean predicted duration for {year}-{month}: {mean_pred:.2f}")

    # Prepare output DataFrame (optional)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict trip durations")
    parser.add_argument("--year", type=int, required=True, help="Year of data (e.g., 2023)")
    parser.add_argument("--month", type=int, required=True, help="Month of data (1-12)")

    args = parser.parse_args()
    main(args.year, args.month)