import pandas as pd

def add_time_columns(df):
    df["date"] = pd.to_datetime(df["date"])
    df["IsWeekend"] = df['date'].dt.weekday > 5
    df['month'] = df['date'].dt.month
    df['Daytime'] = df['time'].apply(lambda x: "morning rush (5-10)" if '05:00' < x <= '10:00'
                                   else ("office hours (10-15)" if '10:00' < x <= '15:00'
                                         else( "afternoon rush (15-19)" if '15:00' < x <= '19:00'
                                              else ("evening (19-23)" if '19:00' < x <= '23:00'
                                                    else("night (23-5")))))
    return df
