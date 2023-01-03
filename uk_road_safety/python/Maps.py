import pygeohash as gh
import pandas as pd
import folium
#import Geohash
from folium.plugins import HeatMapWithTime
from folium.plugins import MarkerCluster, HeatMap
from folium import Marker, Circle, CircleMarker, Choropleth
from geopandas.tools import geocode
from collections import defaultdict, OrderedDict
import ipywidgets as widgets
from ipywidgets import interact, fixed, interact_manual, interactive_output
from folium.features import DivIcon

def city_geocoder(city):
        result = geocode(city, provider='nominatim').geometry.iloc[0]
        return(result)

def visualize_data_UK(year, accident):
    city='London'
    point = city_geocoder(city)

    m = folium.Map(location=[point.y, point.x], tiles='OpenStreetMap', zoom_start=10)


    accident = accident[accident.accident_year == year]



    mc = MarkerCluster()
    for idx, row in accident.iterrows():
        mc.add_child(Marker([row['latitude'], row['longitude']]))

    m.add_child(mc)

    return(m)

def visualize_data_gh():
    city='London'   # UK capital
    point = city_geocoder(city)

    m_1 = folium.Map(location=[point.y, point.x], tiles='OpenStreetMap', zoom_start=12)

    geohash_list=['gcpvj0','gcpvj1','gcpvj4','gcpvhc','gcpuv2']

    for gh in geohash_list:

        lat, long = gh.decode(gh)

        decoded = gh.bbox(gh) # decode the geohash

        W = decoded["w"]
        E = decoded["e"]
        N = decoded["n"]
        S = decoded["s"]

        # create each point of the rectangle
        upper_left = (N, W)
        upper_right = (N, E)
        lower_right = (S, E)
        lower_left = (S, W)
        edges = [upper_left, upper_right, lower_right, lower_left]

        # create rectangle object and add it to the map canvas
        folium.Rectangle(
        bounds=edges,
        color="red",
        fill_color="red",
        weight=5,

        ).add_to(m_1)

        folium.map.Marker(
        [lat, long],
        icon=DivIcon(
            icon_size=(250,20),
            icon_anchor=(0,0),
            html='<div style="font-size: 20pt">'+ gh +'</div>',)).add_to(m_1)
    return(m_1)
