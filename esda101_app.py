import streamlit as st
import geopandas as gpd
import pandas as pd
from streamlit_folium import folium_static
import folium
import matplotlib.pyplot as plt
import requests
import tempfile
import os

# Set page configuration
st.set_page_config(
    page_title="Simple Choropleth Map",
    page_icon="🗺️",
    layout="wide"
)

# Add title and description
st.title("Simple Choropleth Map with GeoPandas")
st.write("This app demonstrates how to create a simple choropleth map using GeoPandas explore functionality.")

# Function to load data
@st.cache_data
def load_data():
    # Instead of directly reading from URL (which can cause Fiona errors),
    # we'll download the file first and then read it locally
    url = 'https://github.com/quarcs-lab/project2021o-notebook/raw/main/map_and_data.geojson'
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.geojson') as tmp:
        # Download the data
        response = requests.get(url)
        # Write to the temporary file
        tmp.write(response.content)
        # Get the file path
        filepath = tmp.name
    
    try:
        # Read the GeoJSON from the local file
        data = gpd.read_file(filepath)
        # Keep only relevant columns
        gdf = data[['mun', 'rank_imds', 'imds', 'geometry']]
        
        # Clean up the temporary file
        os.unlink(filepath)
        
        return gdf
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Clean up the temporary file even if there's an error
        os.unlink(filepath)
        return None

# Load data
with st.spinner("Loading data..."):
    gdf = load_data()

if gdf is None:
    st.error("Failed to load data. Please check your internet connection and try again.")
    st.stop()

# Display basic information about the dataset
st.subheader("Dataset Information")
st.write(f"Number of municipalities: {len(gdf)}")
st.write("First few rows of the data:")
st.dataframe(gdf.drop(columns=['geometry']).head())

# Map configuration options
st.subheader("Map Configuration")

col1, col2 = st.columns(2)

with col1:
    # Choose the variable to map
    variable = st.selectbox(
        "Select variable to map:",
        ["imds", "rank_imds"]
    )
    
    # Choose the color scheme
    color_scheme = st.selectbox(
        "Select color scheme:",
        ["coolwarm", "viridis", "plasma", "YlOrRd", "Blues", "Greens"]
    )

with col2:
    # Choose the classification scheme
    scheme = st.selectbox(
        "Select classification scheme:",
        ["fisherjenks", "quantiles", "equal_interval", "std_mean"]
    )
    
    # Choose the number of classes
    k_classes = st.slider(
        "Number of classes:",
        min_value=2,
        max_value=10,
        value=5
    )

# Create the map using Folium (alternative to GeoPandas explore)
st.subheader("Choropleth Map")

# Create a Folium map
m = folium.Map(
    location=[gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()],
    zoom_start=8,
    tiles="CartoDB positron"
)

# Add choropleth layer
choropleth = folium.Choropleth(
    geo_data=gdf.__geo_interface__,
    name=variable,
    data=gdf,
    columns=['mun', variable],
    key_on='feature.properties.mun',
    fill_color=color_scheme,
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name=variable,
    highlight=True,
    line_color="#0000",
    scheme=scheme,
    k=k_classes
)
choropleth.add_to(m)

# Add tooltips
tooltip = folium.features.GeoJsonTooltip(
    fields=['mun', 'imds', 'rank_imds'],
    aliases=['Municipality', 'IMDS', 'IMDS Rank'],
    style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
)

# Add GeoJson layer with tooltip
folium.GeoJson(
    gdf,
    style_function=lambda x: {'fillColor': 'transparent', 'color': 'transparent'},
    tooltip=tooltip
).add_to(m)

# Display the map in Streamlit
folium_static(m, width=1000, height=600)

# Explanation of GeoPandas explore() method
st.subheader("About GeoPandas explore()")
st.write("""
The GeoPandas `explore()` method creates interactive maps directly from a GeoDataFrame. 
In this app, we're using Folium to create a similar interactive experience within Streamlit.

The equivalent code with GeoPandas explore would look like this:
```python
gdf.explore(
    column=variable,
    tooltip=['mun', 'imds', 'rank_imds'],
    scheme=scheme,
    k=k_classes,
    cmap=color_scheme,
    legend=True,
    tiles='CartoDB positron',
    style_kwds=dict(color="gray", weight=0.5),
    legend_kwds=dict(colorbar=False)
)
```

This approach would work in a notebook environment, but for Streamlit we need to use `streamlit_folium` 
to render the map correctly in the web application.
""")

# Statistics for the selected variable
st.subheader(f"Statistics for {variable}")
stats = gdf[variable].describe()
st.dataframe(pd.DataFrame(stats).T)

# Histogram of the selected variable
st.subheader(f"Distribution of {variable}")
fig, ax = plt.subplots(figsize=(10, 6))
gdf[variable].plot.hist(bins=20, ax=ax)
ax.set_xlabel(variable)
ax.set_ylabel("Frequency")
ax.set_title(f"Histogram of {variable}")
st.pyplot(fig)