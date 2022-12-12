import pygeohash as gh
import pandas as pd
import folium
import geohash
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
    accident = accident.copy()
    accident = accident[accident.accident_year == year] 
    m = folium.Map(location=[point.y, point.x], tiles='OpenStreetMap', zoom_start=10)
    mc = MarkerCluster()
    for idx, row in accident.iterrows():
        mc.add_child(Marker([row['latitude'], row['longitude']]))
    m.add_child(mc)
    return(m)