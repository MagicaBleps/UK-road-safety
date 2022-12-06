import pygeohash as gh

def add_geohash(df,preci=5):
    df['geohash']=df.apply(lambda x: gh.encode(x.latitude, x.longitude, precision=preci), axis=1)
    return df
