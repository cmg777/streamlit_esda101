import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from branca.colormap import LinearColormap

# Load the GeoJSON data
data = gpd.read_file('https://github.com/quarcs-lab/project2021o-notebook/raw/main/map_and_data.geojson')

# Calculate the centroid of the entire GeoDataFrame
centroid = data.geometry.unary_union.centroid

# Create a map centered on the centroid of the data
m = folium.Map(location=[centroid.y, centroid.x], zoom_start=6, tiles='CartoDB positron')

# Create a colormap
colormap = LinearColormap(colors=['blue', 'white', 'red'], vmin=data['imds'].min(), vmax=data['imds'].max())

# Add the choropleth layer
folium.Choropleth(
    geo_data=data,
    name='choropleth',
    data=data,
    columns=['name', 'imds'],
    key_on='feature.properties.name',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='IMDS'
).add_to(m)

# Add GeoJson layer with tooltips
folium.GeoJson(
    data,
    style_function=lambda feature: {
        'fillColor': colormap(feature['properties']['imds']),
        'color': 'gray',
        'weight': 0.5,
        'fillOpacity': 0.7
    },
    tooltip=folium.GeoJsonTooltip(
        fields=['mun', 'imds', 'rank_imds'],
        aliases=['Municipality', 'IMDS', 'IMDS Rank'],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
    )
).add_to(m)

# Add colormap to the map
colormap.add_to(m)

# Display the map in Streamlit
st.title('Choropleth Map')
folium_static(m)
