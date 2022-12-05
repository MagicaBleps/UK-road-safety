import pygeohash as gh

def add_geohash(df):
    df['geohash']=df.apply(lambda x: gh.encode(x.latitude, x.longitude, precision=5), axis=1)
    return df
