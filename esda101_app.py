import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import requests
import tempfile
import os
import logging
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Interactive Choropleth Map",
    page_icon="🗺️",
    layout="wide"
)

# Add title and description
st.title("Interactive Choropleth Map with GeoPandas")
st.write("Visualize geographical data with customizable choropleth maps.")

# Function to load data
@st.cache_data
def load_data(url: str = 'https://github.com/quarcs-lab/project2021o-notebook/raw/main/map_and_data.geojson') -> Optional[gpd.GeoDataFrame]:
    """
    Load GeoJSON data from a URL into a GeoDataFrame.
    
    Parameters:
    -----------
    url : str
        URL to the GeoJSON file
        
    Returns:
    --------
    gpd.GeoDataFrame or None
        Processed GeoDataFrame containing geographical data or None if loading fails
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.geojson') as tmp:
        try:
            # Download the data with error handling
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Write to the temporary file
            tmp.write(response.content)
            filepath = tmp.name
            
            logger.info(f"Downloaded data to temporary file: {filepath}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download data: {e}")
            os.unlink(tmp.name)
            return None
    
    try:
        # Read the GeoJSON from the local file
        data = gpd.read_file(filepath)
        logger.info(f"Successfully loaded GeoJSON with {len(data)} features")
        
        # Keep only relevant columns
        gdf = data[['mun', 'rank_imds', 'imds', 'geometry']]
        
        return gdf
    except Exception as e:
        logger.error(f"Error processing GeoJSON data: {e}")
        return None
    finally:
        # Clean up the temporary file
        if os.path.exists(filepath):
            os.unlink(filepath)
            logger.info(f"Removed temporary file: {filepath}")

def create_choropleth_map(gdf: gpd.GeoDataFrame, 
                         variable: str, 
                         color_scheme: str, 
                         scheme: str, 
                         k_classes: int) -> folium.Map:
    """
    Create a choropleth map using Folium.
    
    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing data to be visualized
    variable : str
        Column name in gdf to use for choropleth coloring
    color_scheme : str
        Color scheme for choropleth map
    scheme : str
        Classification scheme for data binning
    k_classes : int
        Number of classes for data binning
        
    Returns:
    --------
    folium.Map
        Folium map object with choropleth layer
    """
    # Calculate map center from data
    center_y = gdf.geometry.centroid.y.mean()
    center_x = gdf.geometry.centroid.x.mean()
    
    # Create a Folium map
    m = folium.Map(
        location=[center_y, center_x],
        zoom_start=8,
        tiles="CartoDB positron"
    )
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
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
    
    # Add tooltips with enhanced information
    tooltip = folium.features.GeoJsonTooltip(
        fields=['mun', 'imds', 'rank_imds'],
        aliases=['Municipality', 'IMDS Score', 'IMDS Rank'],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; "
              "padding: 10px; border-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,0.3);")
    )
    
    # Add GeoJson layer with tooltip
    folium.GeoJson(
        gdf,
        style_function=lambda x: {'fillColor': 'transparent', 'color': 'transparent'},
        tooltip=tooltip
    ).add_to(m)
    
    return m

def display_statistics(gdf: gpd.GeoDataFrame, variable: str) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Generate statistics and histogram for the selected variable.
    
    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing data
    variable : str
        Column name for which to generate statistics
        
    Returns:
    --------
    Tuple[pd.DataFrame, plt.Figure]
        DataFrame with statistics and matplotlib Figure with histogram
    """
    # Calculate statistics
    stats = gdf[variable].describe()
    stats_df = pd.DataFrame(stats).T
    
    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    gdf[variable].plot.hist(bins=20, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel(variable, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Distribution of {variable}", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return stats_df, fig

# Main application flow
def main():
    # Load data with progress indicator
    with st.spinner("Loading geographical data..."):
        gdf = load_data()

    if gdf is None:
        st.error("⚠️ Failed to load data. Please check your internet connection and try again.")
        st.stop()
    
    # Success message with data overview
    st.success(f"✅ Successfully loaded data for {len(gdf)} municipalities")
    
    # Add tabs for better organization
    tab1, tab2, tab3 = st.tabs(["Map Visualization", "Data Explorer", "About"])
    
    with tab1:
        st.subheader("Choropleth Map Configuration")
        
        # Create two columns for map controls
        col1, col2 = st.columns(2)
        
        with col1:
            # Choose the variable to map
            variable = st.selectbox(
                "Select variable to map:",
                options=["imds", "rank_imds"],
                help="IMDS = Index of Multiple Deprivation Score, Rank = Position in ranking"
            )
            
            # Choose the color scheme
            color_scheme = st.selectbox(
                "Select color scheme:",
                options=["YlOrRd", "Blues", "Greens", "coolwarm", "viridis", "plasma"],
                help="Color palette to use for visualization"
            )
        
        with col2:
            # Choose the classification scheme
            scheme = st.selectbox(
                "Select classification scheme:",
                options=["quantiles", "equal_interval", "fisherjenks", "std_mean"],
                help="Method to classify data into color bins"
            )
            
            # Choose the number of classes
            k_classes = st.slider(
                "Number of classes:",
                min_value=2,
                max_value=10,
                value=5,
                help="Number of color categories"
            )
        
        # Create and display the map
        st.subheader("Interactive Choropleth Map")
        map_obj = create_choropleth_map(gdf, variable, color_scheme, scheme, k_classes)
        folium_static(map_obj, width=1000, height=600)
        
        # Display statistics for the selected variable
        st.subheader(f"Statistics for {variable}")
        stats_df, hist_fig = display_statistics(gdf, variable)
        st.dataframe(stats_df, use_container_width=True)
        st.pyplot(hist_fig)
    
    with tab2:
        st.subheader("Data Explorer")
        
        # Search functionality
        search_term = st.text_input("Search municipalities:", "")
        if search_term:
            filtered_df = gdf[gdf['mun'].str.contains(search_term, case=False)]
            st.write(f"Found {len(filtered_df)} municipalities matching '{search_term}'")
        else:
            filtered_df = gdf
        
        # Display dataframe with pagination
        st.dataframe(filtered_df.drop(columns=['geometry']), use_container_width=True)
        
        # Option to download the data
        st.download_button(
            label="Download data as CSV",
            data=filtered_df.drop(columns=['geometry']).to_csv(index=False),
            file_name="choropleth_data.csv",
            mime="text/csv"
        )
    
    with tab3:
        st.subheader("About This Application")
        st.write("""
        This application demonstrates how to create interactive choropleth maps using GeoPandas and Folium
        within a Streamlit web application. The data visualized represents municipalities with their 
        corresponding Index of Multiple Deprivation Score (IMDS) and rankings.
        
        ### Key Features:
        - Interactive choropleth mapping with customizable parameters
        - Data exploration tools
        - Statistical analysis and visualization
        
        ### GeoPandas explore() Alternative
        
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
        """)

if __name__ == "__main__":
    main()