import pygeohash as gh

def add_geohash(df,p=5):
    df['geohash']=df.apply(lambda x: gh.encode(x.latitude, x.longitude, precision=p), axis=1)
    return df
