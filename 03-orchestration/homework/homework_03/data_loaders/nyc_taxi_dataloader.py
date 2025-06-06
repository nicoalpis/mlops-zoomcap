import pandas as pd
if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_yellow_taxi_data(*args, **kwargs):
    """
    Load March 2023 Yellow Taxi trip data from online Parquet file.
    """
    file_path = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet'

    df = pd.read_parquet(file_path)
    print(f"Number of rows loaded: {len(df)}")

    return df


@test
def test_output(output, *args) -> None:
    """
    Test to ensure data was loaded correctly.
    """
    assert output is not None, 'The output is undefined'
    assert len(output) > 0, 'No data was loaded'
