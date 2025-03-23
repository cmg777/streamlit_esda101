# IMPORT SECTION - Libraries needed for this application
import streamlit as st             # Core Streamlit library for creating web applications
import geopandas as gpd            # Extends pandas to work with geospatial data
import plotly.express as px        # High-level interface for creating interactive visualizations
import pandas as pd                # For data manipulation and analysis
import json                        # For GeoJSON processing and export
import csv                         # For reading the data dictionary

# PAGE CONFIGURATION - Sets up how the application appears in the browser
st.set_page_config(
    page_title="Choropleth Map Visualization",  # Sets the browser tab title
    layout="wide"                               # Uses the full width of the browser window
)

# APPLICATION TITLE AND DESCRIPTION - Main user-facing elements at the top of the page
st.title("Interactive Choropleth Map")          # Creates the main heading

# Adds explanatory text using Markdown formatting
st.markdown("""
This application displays a choropleth map showing the Index of Sustainable Development at the municipal level. 
The application includes a data dictionary with 139 variables, allowing you to explore different dimensions of municipal development.
""")

# SIDEBAR SETUP - Creates a navigation panel on the left side of the screen
st.sidebar.header("Map Controls")               # Adds a header to the sidebar

# DATA LOADING FUNCTIONS - Define how to fetch and prepare the geospatial data
@st.cache_data                                  # Caching decorator prevents reloading on every interaction
def load_data():
    # Loads geographic data directly from a GitHub repository URL
    data = gpd.read_file('https://github.com/quarcs-lab/project2021o-notebook/raw/main/map_and_data.geojson')
    
    # Converts the coordinate system to standard WGS84 (latitude/longitude) for web mapping
    data = data.to_crs(epsg=4326)
    
    # Creates a unique identifier column for each region to link data to map features
    data["id"] = data.index.astype(str)
    
    return data

@st.cache_data
def load_data_dictionary():
    """Load and parse the data dictionary CSV file."""
    try:
        # Create a dictionary mapping variable names to their human-readable labels
        variable_labels = {}
        
        # Read the data dictionary file
        with open('dataDefinitions.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'Variable' in row and 'Label' in row:
                    variable_labels[row['Variable']] = row['Label']
        
        return variable_labels
    except Exception as e:
        st.warning(f"Could not load data dictionary: {e}. Using original column names.")
        return {}

# MAIN APPLICATION BLOCK - Wrapped in try-except for error handling
try:
    # Shows a loading indicator while fetching data
    with st.spinner("Loading geographic data..."):
        data = load_data()
        variable_labels = load_data_dictionary()
    
    # SIDEBAR INFORMATION - Displays metadata about the dataset
    st.sidebar.subheader("Dataset Information")
    st.sidebar.info(f"Number of regions: {len(data)}")  # Shows how many geographic areas are in the dataset
    
    # Identifies numeric columns that could potentially be used for choropleth coloring
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Create a list of column options with labels for the dropdown
    column_options = []
    for col in numeric_columns:
        # Use the label from the dictionary if available, otherwise use the column name
        display_name = variable_labels.get(col, col)
        column_options.append({"label": display_name, "value": col})
    
    # Sort options alphabetically by display name for easier navigation
    column_options.sort(key=lambda x: x["label"])
    
    # USER CONTROLS - Interactive elements for customizing the visualization
    
    # Dropdown for selecting which column to visualize on the map
    # First find the index of the IMDS option
    default_index = 0
    for i, option in enumerate(column_options):
        if option["value"] == "imds":
            default_index = i
            break
    
    # Use format_func to display the label while storing the actual column name
    color_column = st.sidebar.selectbox(
        "Select data to visualize",
        options=[option["value"] for option in column_options],
        index=default_index,
        format_func=lambda x: variable_labels.get(x, x),
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
        # Get the human-readable label for the selected column
        column_label = variable_labels.get(color_column, color_column)
        
        # Prepare hover data with labels
        hover_cols = [color_column] + [col for col in numeric_columns[:3] if col != color_column]
        
        # Create a labels dictionary for better hover information
        labels = {col: variable_labels.get(col, col) for col in hover_cols + ["mun"]}
        
        fig = px.choropleth_mapbox(
            data_frame=data,                  # The source dataset with both geometry and attributes
            geojson=geojson_dict,             # The geographic boundaries in GeoJSON format
            locations="id",                   # Column that links data rows to geographic features
            color=color_column,               # Selected column to determine the color of each region
            hover_name="mun",                 # Municipality name shown in hover tooltip
            hover_data=hover_cols,            # Dynamic hover data
            color_continuous_scale="viridis", # Color gradient palette for the choropleth
            mapbox_style=map_style,           # Base map style from user selection
            zoom=zoom,                        # Initial zoom level from user selection
            center={"lat": data.geometry.centroid.y.mean(), 
                   "lon": data.geometry.centroid.x.mean()},  # Centers map on the data
            opacity=opacity,                  # Transparency level from user selection
            labels=labels,                    # Use dictionary of labels for better hover information
            color_continuous_midpoint=data[color_column].median()  # Center color scale at median value
        )
        
        # Adjusts the figure layout for better display
        # Get the human-readable label for the color bar title
        column_label = variable_labels.get(color_column, color_column)
        
        fig.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},  # Removes unnecessary margins
            width=630,                                 # Sets a fixed width for the map
            height=600,                                # Sets a fixed height for the map
            coloraxis_colorbar=dict(
                title=column_label,                    # Use the human-readable label for the color scale
                title_side="right"                     # Position the title on the right side
            )
        )
        
        # Renders the interactive map in the app
        st.plotly_chart(fig, use_container_width=False)
    
    # RIGHT COLUMN - Contains data summary and exploration tools
    with col2:
        # Creates a heading for the data section
        st.subheader("Data Summary")
        
        # Creates a container for the statistics table that will update when the selected column changes
        stats_container = st.container()
        
        with stats_container:
            # Provides statistical summary of the selected variable with a human-readable label
            column_label = variable_labels.get(color_column, color_column)
            st.write(f"**{column_label} Statistics:**")
            stats = data[color_column].describe().reset_index()  # Generates descriptive statistics for the selected column
            stats.columns = ["Statistic", "Value"]         # Renames columns for clarity
            
            # Format the values based on data type (e.g., percentiles vs counts)
            if stats["Value"].dtype == "float64":
                stats["Value"] = stats["Value"].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
            
            st.dataframe(stats, use_container_width=True)  # Displays the statistics in a table
            
            # Add information about the selected variable's range and distribution
            min_val = data[color_column].min()
            max_val = data[color_column].max()
            st.info(f"Range: {min_val:.4f} to {max_val:.4f}")
            
            # Calculate and display regions with extreme values
            if "mun" in data.columns:  # Check if municipality column exists
                top_region = data.loc[data[color_column].idxmax(), "mun"]
                bottom_region = data.loc[data[color_column].idxmin(), "mun"]
                st.text(f"Highest value: {top_region}")
                st.text(f"Lowest value: {bottom_region}")
        
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