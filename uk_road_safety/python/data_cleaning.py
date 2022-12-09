import pandas as pd

import pygeohash as gh


def delete_columns(df):
    columns_to_delete=['Unnamed: 0','accident_reference', 'accident_index',
                       'location_easting_osgr','location_northing_osgr',
                       'police_force','local_authority_district',
                       'local_authority_ons_district','local_authority_highway',
                       'first_road_class', 'first_road_number',
                       'road_type', 'speed_limit',
                       'junction_detail', 'junction_control',
                       'pedestrian_crossing_human_control',
                       'pedestrian_crossing_physical_facilities',
                       'light_conditions', 'weather_conditions',
                       'road_surface_conditions','special_conditions_at_site',
                       'second_road_class', 'second_road_number',
                       'carriageway_hazards', 'urban_or_rural_area',
                       'did_police_officer_attend_scene_of_accident',
                       'trunk_road_flag','lsoa_of_accident_location']
    df_new=df.drop(columns_to_delete,axis=1)
    return df_new

def fix_missing_values(df):
    #dropping rows without lat and lon
    df_new=df.dropna(axis=0,subset=['longitude','latitude'])

    return df_new

def add_geohash(df,p=5):
    df['geohash']=df.apply(lambda x: gh.encode(x.latitude, x.longitude, precision=p), axis=1)
    return df

def add_time_columns(df):
    df["date"] = pd.to_datetime(df["date"],format='%d/%m/%Y')
    df["IsWeekend"] = df['date'].dt.weekday > 5
    df['month'] = df['date'].dt.month
    df['Daytime'] = df['time'].apply(lambda x: "morning rush (5-10)" if '05:00' < x <= '10:00'
                                   else ("office hours (10-15)" if '10:00' < x <= '15:00'
                                         else( "afternoon rush (15-19)" if '15:00' < x <= '19:00'
                                              else ("evening (19-23)" if '19:00' < x <= '23:00'
                                                    else("night (23-5")))))
    return df

def prepare_data_for_groupby(df,precision):
    '''The function takes the list of accidents in time and prepares it for the groupby step.
    The precision parameter is used for the geohash step.'''
    df_new=delete_columns(df)
    df_new=fix_missing_values(df_new)
    df_new=add_time_columns(df_new)
    df_new=add_geohash(df_new,p=precision)

    return df_new
