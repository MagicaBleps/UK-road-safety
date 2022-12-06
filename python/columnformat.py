def to_datetime(df):
   df['date']= pd.to_datetime(df['date'], format="%d/%m/%Y")
   return df

# define a function that turns the hours into daytime groups
def when_was_it(hour):
    if hour >= 5 and hour < 10:
        return "morning rush (5-10)"
    elif hour >= 10 and hour < 15:
        return "office hours (10-15)"
    elif hour >= 15 and hour < 19:
        return "afternoon rush (15-19)"
    elif hour >= 19 and hour < 23:
        return "evening (19-23)"
    else:
        return "night (23-5)"

def daytime_groups(df):
    # slice first and second string from time column
    df['Hour'] = df['time'].str[0:2]
    # convert new column to numeric datetype
    df['Hour'] = pd.to_numeric(df['Hour'])
    # drop null values in our new column
    df = df.dropna(subset=['Hour'])
    # cast to integer values
    df['Hour'] = df['Hour'].astype('int')
    df['Daytime'] = df['Hour'].apply(when_was_it)
    return df