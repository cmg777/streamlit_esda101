# IMPORT SECTION - Libraries needed for this application
import streamlit as st             # Core Streamlit library for creating web applications
import geopandas as gpd            # Extends pandas to work with geospatial data
import plotly.express as px        # High-level interface for creating interactive visualizations
import pandas as pd                # For data manipulation and analysis
import json                        # For GeoJSON processing and export

# PAGE CONFIGURATION - Sets up how the application appears in the browser
st.set_page_config(
    page_title="Choropleth Map Visualization",  # Sets the browser tab title
    layout="wide"                               # Uses the full width of the browser window
)

# APPLICATION TITLE AND DESCRIPTION - Main user-facing elements at the top of the page
st.title("Interactive Choropleth Map")          # Creates the main heading

# Adds explanatory text using Markdown formatting
st.markdown("""
This application displays a choropleth map showing the Index of Sustanable Development at the municipal level.
""")

# SIDEBAR SETUP - Creates a navigation panel on the left side of the screen
st.sidebar.header("Map Controls")               # Adds a header to the sidebar

# DATA LOADING FUNCTION - Defines how to fetch and prepare the geospatial data
@st.cache_data                                  # Caching decorator prevents reloading on every interaction
def load_data():
    # Loads geographic data directly from a GitHub repository URL
    data = gpd.read_file('https://github.com/quarcs-lab/project2021o-notebook/raw/main/map_and_data.geojson')
    
    # Converts the coordinate system to standard WGS84 (latitude/longitude) for web mapping
    data = data.to_crs(epsg=4326)
    
    # Creates a unique identifier column for each region to link data to map features
    data["id"] = data.index.astype(str)
    
    return data

# MAIN APPLICATION BLOCK - Wrapped in try-except for error handling
try:
    # Shows a loading indicator while fetching data
    with st.spinner("Loading geographic data..."):
        data = load_data()
    
    # SIDEBAR INFORMATION - Displays metadata about the dataset
    st.sidebar.subheader("Dataset Information")
    st.sidebar.info(f"Number of regions: {len(data)}")  # Shows how many geographic areas are in the dataset
    
    # Identifies numeric columns that could potentially be used for choropleth coloring
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # USER CONTROLS - Interactive elements for customizing the visualization
    
    # Dropdown for selecting which column to visualize on the map
    color_column = st.sidebar.selectbox(
        "Select data to visualize", 
        options=numeric_columns,
        index=numeric_columns.index("imds") if "imds" in numeric_columns else 0,
        help="Choose which data column to display on the map"
    )
    
    # Dropdown menu for selecting the base map style
    map_styles = ["carto-positron", "open-street-map", "white-bg", "carto-darkmatter"]
    map_style = st.sidebar.selectbox("Map style", options=map_styles, index=0)
    
    # Slider to control the transparency of the choropleth layer
    opacity = st.sidebar.slider("Map opacity", min_value=0.0, max_value=1.0, value=0.6, step=0.1)
    
    # Slider to adjust the initial zoom level of the map
    zoom = st.sidebar.slider("Zoom level", min_value=3.0, max_value=10.0, value=4.5, step=0.5)
    
    # PAGE LAYOUT - Creates a two-column structure for the main content
    col1, col2 = st.columns([2, 1])  # Left column is twice as wide as the right column
    
    # LEFT COLUMN - Contains the interactive map visualization
    with col1:
        # Converts the GeoDataFrame to a GeoJSON format that Plotly can interpret
        geojson_dict = data.__geo_interface__
        
        # Creates the interactive choropleth map with Plotly Express
        fig = px.choropleth_mapbox(
            data_frame=data,                  # The source dataset with both geometry and attributes
            geojson=geojson_dict,             # The geographic boundaries in GeoJSON format
            locations="id",                   # Column that links data rows to geographic features
            color=color_column,               # Selected column to determine the color of each region
            hover_name="mun",                 # Municipality name shown in hover tooltip
            hover_data=[color_column] + [col for col in numeric_columns[:3] if col != color_column], # Dynamic hover data
            color_continuous_scale="viridis", # Color gradient palette for the choropleth
            mapbox_style=map_style,           # Base map style from user selection
            zoom=zoom,                        # Initial zoom level from user selection
            center={"lat": data.geometry.centroid.y.mean(), 
                   "lon": data.geometry.centroid.x.mean()},  # Centers map on the data
            opacity=opacity                   # Transparency level from user selection
        )
        
        # Adjusts the figure layout for better display
        fig.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},  # Removes unnecessary margins
            width=630,                                 # Sets a fixed width for the map
            height=600                                 # Sets a fixed height for the map
        )
        
        # Renders the interactive map in the app
        st.plotly_chart(fig, use_container_width=False)
    
    # RIGHT COLUMN - Contains data summary and exploration tools
    with col2:
        # Creates a heading for the data section
        st.subheader("Data Summary")
        
        # Displays statistical summary of the main variable (IMDS)
        st.write("**IMDS Statistics:**")
        stats = data["imds"].describe().reset_index()  # Generates descriptive statistics and resets index
        stats.columns = ["Statistic", "Value"]         # Renames columns for clarity
        st.dataframe(stats, use_container_width=True)  # Displays the statistics in a table
        
        # Optional display of the full dataset (without geometry column to save space)
        if st.checkbox("Show raw data"):
            st.dataframe(data.drop(columns='geometry'), use_container_width=True)
        
        # DATA EXPORT - Creates download buttons for both CSV and GeoJSON formats
        
        # CSV download option (without geometry)
        csv_data = data.drop(columns='geometry').to_csv(index=False)  # Converts data to CSV format
        st.download_button(
            label="Download data as CSV",     # Button text
            data=csv_data,                    # Data to be downloaded
            file_name="map_data.csv",         # Default filename for the download
            mime="text/csv",                  # MIME type for CSV files
            key="csv_download"                # Unique key for this button
        )
        
        # GeoJSON download option (includes geometry)
        import json
        geojson_data = json.dumps(data.__geo_interface__, indent=2)  # Converts to GeoJSON with formatting
        st.download_button(
            label="Download data as GeoJSON", # Button text
            data=geojson_data,                # GeoJSON data to be downloaded
            file_name="map_data.geojson",     # Default filename for the download
            mime="application/geo+json",      # MIME type for GeoJSON files
            key="geojson_download"            # Unique key for this button
        )

# ERROR HANDLING - Catches and displays any exceptions that occur
except Exception as e:
    # Shows user-friendly error messages
    st.error(f"An error occurred: {e}")                          # Displays the specific error
    st.error("Please check the data source or report this issue.")  # Provides general guidance

# FOOTER - Adds attribution and source information at the bottom of the page
st.markdown("---")  # Horizontal divider
st.markdown("Data source: [GitHub Project Repository](https://github.com/quarcs-lab/project2021o-notebook)")