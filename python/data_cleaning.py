import pandas as pd


def delete_columns(df):
    columns_to_delete=['accident_reference',
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
