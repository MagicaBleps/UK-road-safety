import pandas as pd

def add_time_columns(df):
    df["date"] = pd.to_datetime(df["date"])
    df["IsWeekend"] = df['date'].dt.weekday > 5
    df['month'] = df['date'].dt.month
    return df
