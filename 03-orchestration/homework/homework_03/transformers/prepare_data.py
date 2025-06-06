if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def prepare_data(df, *args, **kwargs):
    """
    Clean NYC Yellow Taxi data:
    - Calculate duration
    - Filter to 1-60 min trips
    - Convert location IDs to strings
    """
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df['duration'].dt.total_seconds() / 60

    # Keep only trips between 1 and 60 minutes
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    # Convert categorical columns to string
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    print(f"Cleaned data size: {len(df)}")

    return df


@test
def test_output(output, *args) -> None:
    assert output is not None, 'The output is undefined'
    assert 'duration' in output.columns, 'Missing duration column'
    assert (output.duration >= 1).all() and (output.duration <= 60).all(), 'Duration filtering failed'
