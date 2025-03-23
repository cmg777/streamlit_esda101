import streamlit as st
import geopandas as gpd
import plotly.express as px
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Choropleth Map Visualization",
    layout="wide"
)

# Add a title to the app
st.title("Interactive Choropleth Map")

# Add a description
st.markdown("""
This application displays a choropleth map showing the Index of Sustanable Development at the municipal level.
""")

# Create sidebar for controls
st.sidebar.header("Map Controls")

# Load data function with caching to improve performance
@st.cache_data
def load_data():
    # Load data from the GitHub repository
    data = gpd.read_file('https://github.com/quarcs-lab/project2021o-notebook/raw/main/map_and_data.geojson')
    data = data.to_crs(epsg=4326)
    
    # Ensure a unique ID column
    data["id"] = data.index.astype(str)
    
    return data

# Load the data
try:
    with st.spinner("Loading geographic data..."):
        data = load_data()
    
    # Show data info in the sidebar
    st.sidebar.subheader("Dataset Information")
    st.sidebar.info(f"Number of regions: {len(data)}")
    
    # Display available columns for coloring
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Map style options
    map_styles = ["carto-positron", "open-street-map", "white-bg", "carto-darkmatter"]
    map_style = st.sidebar.selectbox("Map style", options=map_styles, index=0)
    
    # Opacity slider
    opacity = st.sidebar.slider("Map opacity", min_value=0.0, max_value=1.0, value=0.6, step=0.1)
    
    # Zoom level slider
    zoom = st.sidebar.slider("Zoom level", min_value=3.0, max_value=10.0, value=4.5, step=0.5)
    
    # Create two columns for display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Convert GeoDataFrame to GeoJSON dictionary
        geojson_dict = data.__geo_interface__
        
        # Create the choropleth map with updated configuration
        fig = px.choropleth_mapbox(
            data_frame=data,
            geojson=geojson_dict,
            locations="id",
            color="imds",
            hover_name="mun",
            hover_data=["imds", "rank_imds"],
            color_continuous_scale="viridis",
            mapbox_style=map_style,
            zoom=zoom,
            center={"lat": data.geometry.centroid.y.mean(), "lon": data.geometry.centroid.x.mean()},
            opacity=opacity
        )
        
        # Improve the figure layout with explicit dimensions
        fig.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            width=630,
            height=600
        )
        
        # Display the figure
        st.plotly_chart(fig, use_container_width=False)
    
    with col2:
        # Data summary
        st.subheader("Data Summary")
        
        # Show statistics for IMDS variable
        st.write("**IMDS Statistics:**")
        stats = data["imds"].describe().reset_index()
        stats.columns = ["Statistic", "Value"]
        st.dataframe(stats, use_container_width=True)
        
        # Option to show raw data
        if st.checkbox("Show raw data"):
            st.dataframe(data.drop(columns='geometry'), use_container_width=True)
        
        # Download option
        csv_data = data.drop(columns='geometry').to_csv(index=False)
        st.download_button(
            label="Download data as CSV",
            data=csv_data,
            file_name="map_data.csv",
            mime="text/csv"
        )

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.error("Please check the data source or report this issue.")

# Footer
st.markdown("---")
st.markdown("Data source: [GitHub Project Repository](https://github.com/quarcs-lab/project2021o-notebook)")