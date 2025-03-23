# IMPORT SECTION - Libraries needed for this application
import streamlit as st             # Core Streamlit library for creating web applications
import geopandas as gpd            # Extends pandas to work with geospatial data
import plotly.express as px        # High-level interface for creating interactive visualizations
import pandas as pd                # For data manipulation and analysis
import json                        # For GeoJSON processing and export
import csv                         # For reading the data dictionary
import os                          # For file path operations
import logging                     # For error logging
from typing import Dict, List, Optional, Union, Any  # Type hints for better code documentation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Custom exception classes for better error handling
class AppError(Exception):
    """Base exception class for application errors."""
    pass

class DataLoadError(AppError):
    """Exception raised for errors in data loading."""
    pass

class VisualizationError(AppError):
    """Exception raised for errors in visualization creation."""
    pass

# PAGE CONFIGURATION - Sets up how the application appears in the browser
st.set_page_config(
    page_title="Choropleth Map Visualization",  # Sets the browser tab title
    layout="wide",                              # Uses the full width of the browser window
    initial_sidebar_state="expanded"            # Show sidebar by default
)

# Initialize session state for persistent variables across reruns
if "map_height" not in st.session_state:
    st.session_state.map_height = 600
if "selected_regions" not in st.session_state:
    st.session_state.selected_regions = []
if "show_tutorial" not in st.session_state:
    st.session_state.show_tutorial = True

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
@st.cache_data(ttl=3600, max_entries=10)
def load_data():
    """
    Load and prepare geographic data with optimized caching.
    
    Returns:
        GeoDataFrame: The prepared geographic data
    
    Raises:
        DataLoadError: If data cannot be loaded or processed
    """
    try:
        # First try local file for faster loading
        data_path = "map_and_data.geojson"
        if os.path.exists(data_path):
            data = gpd.read_file(data_path)
            logging.info("Loaded data from local file")
        else:
            # Fall back to remote source
            logging.info("Local file not found, fetching from GitHub...")
            with st.spinner("Loading geographic data from remote source..."):
                data = gpd.read_file('https://github.com/quarcs-lab/project2021o-notebook/raw/main/map_and_data.geojson')
                # Save locally for future use
                data.to_file(data_path, driver="GeoJSON")
                logging.info("Saved data to local file for future use")
        
        # Converts the coordinate system to standard WGS84 (latitude/longitude) for web mapping
        data = data.to_crs(epsg=4326)
        
        # Creates a unique identifier column for each region to link data to map features
        data["id"] = data.index.astype(str)
        
        return data
    
    except Exception as e:
        logging.error(f"Failed to load data: {str(e)}", exc_info=True)
        raise DataLoadError(f"Could not load geographic data: {str(e)}")

@st.cache_data
def load_data_dictionary():
    """
    Load and parse the data dictionary CSV file.
    
    Returns:
        dict: A dictionary mapping variable names to their human-readable labels
    """
    try:
        # Create a dictionary mapping variable names to their human-readable labels
        variable_labels = {}
        
        # Check if the file exists locally first
        if os.path.exists('dataDefinitions.csv'):
            with open('dataDefinitions.csv', 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'Variable' in row and 'Label' in row:
                        variable_labels[row['Variable']] = row['Label']
            
            logging.info(f"Loaded {len(variable_labels)} variable definitions from data dictionary")
            return variable_labels
        else:
            logging.warning("Data dictionary file not found")
            return {}
            
    except Exception as e:
        logging.error(f"Error loading data dictionary: {str(e)}", exc_info=True)
        st.warning(f"Could not load data dictionary: {e}. Using original column names.")
        return {}

@st.experimental_memo
def get_filtered_data(data, filters=None):
    """
    Filter the dataset based on specified criteria.
    
    Args:
        data: The GeoDataFrame to filter
        filters: Dictionary of column:value pairs to filter by
        
    Returns:
        GeoDataFrame: Filtered dataset
    """
    if filters is None or not filters:
        return data
    
    # Apply filters and return subset
    filtered_data = data.copy()
    for column, value in filters.items():
        if value:  # Only apply non-empty filters
            filtered_data = filtered_data[filtered_data[column] == value]
    
    return filtered_data

# MAIN APPLICATION BLOCK - Wrapped in try-except for error handling
try:
    # Shows a loading indicator while fetching data
    data = load_data()
    variable_labels = load_data_dictionary()
    
    # SIDEBAR INFORMATION - Displays metadata about the dataset
    st.sidebar.subheader("Dataset Information")
    st.sidebar.info(f"Number of regions: {len(data)}")  # Shows how many geographic areas are in the dataset
    
    # Add a tutorial toggle in the sidebar
    show_tutorial = st.sidebar.checkbox("Show tutorial", value=st.session_state.show_tutorial)
    st.session_state.show_tutorial = show_tutorial
    
    if show_tutorial:
        st.sidebar.info("""
        **How to use this app:**
        1. Select a variable from the dropdown to visualize
        2. Adjust map style and opacity using the controls
        3. Explore statistics and charts in the data tabs
        4. Compare regions using the multi-select option
        5. Download data using the buttons at the bottom
        """)
    
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
    
    # Search box for finding variables
    search_term = st.sidebar.text_input("Search variables", "")
    
    # Filter column options based on search
    filtered_options = column_options
    if search_term:
        filtered_options = [
            option for option in column_options 
            if search_term.lower() in option["label"].lower() or search_term.lower() in option["value"].lower()
        ]
    
    if not filtered_options and search_term:
        st.sidebar.warning(f"No variables matching '{search_term}'")
        # Show all options when search yields no results
        filtered_options = column_options
    
    # First find the index of the IMDS option
    default_index = 0
    for i, option in enumerate(filtered_options):
        if option["value"] == "imds":
            default_index = i
            break
    
    # Use format_func to display the label while storing the actual column name
    color_column = st.sidebar.selectbox(
        "Select data to visualize",
        options=[option["value"] for option in filtered_options],
        index=min(default_index, len(filtered_options)-1) if filtered_options else 0,
        format_func=lambda x: variable_labels.get(x, x),
        help="Choose which data column to display on the map. Hover over each municipality to see its value."
    )
    
    # Dropdown menu for selecting the base map style
    map_styles = ["carto-positron", "open-street-map", "white-bg", "carto-darkmatter"]
    map_style = st.sidebar.selectbox(
        "Map style", 
        options=map_styles, 
        index=0,
        help="Select the background map design"
    )
    
    # Color scale selection
    color_scales = ["viridis", "plasma", "inferno", "magma", "cividis", 
                    "twilight", "RdBu_r", "Blues", "Greens", "Reds", "YlOrRd"]
    
    color_scale = st.sidebar.selectbox(
        "Color palette",
        options=color_scales,
        index=0,
        help="Choose the color scheme for the map"
    )
    
    # Custom midpoint selection
    use_median = st.sidebar.checkbox("Center color scale at median", value=True)
    if use_median:
        midpoint = data[color_column].median()
    else:
        midpoint = st.sidebar.slider(
            "Color scale midpoint",
            float(data[color_column].min()),
            float(data[color_column].max()),
            float(data[color_column].median())
        )
    
    # Slider to control the transparency of the choropleth layer
    opacity = st.sidebar.slider(
        "Map opacity", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.6, 
        step=0.1,
        help="Adjust the transparency of the colored regions"
    )
    
    # Slider to adjust the initial zoom level of the map
    zoom = st.sidebar.slider(
        "Zoom level", 
        min_value=3.0, 
        max_value=10.0, 
        value=4.5, 
        step=0.5,
        help="Adjust how close or far the map appears"
    )
    
    # Region comparison functionality
    st.sidebar.subheader("Compare Regions")
    selected_regions = st.sidebar.multiselect(
        "Select regions to compare",
        options=sorted(data["mun"].unique()),
        default=st.session_state.selected_regions,
        help="Choose multiple municipalities to compare their values"
    )
    st.session_state.selected_regions = selected_regions
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Map Visualization", "Data Analysis", "Documentation"])
    
    with tab1:
        # MAP VISUALIZATION TAB
        # PAGE LAYOUT - Creates a two-column structure for the main content
        col1, col2 = st.columns([2, 1])  # Left column is twice as wide as the right column
        
        # LEFT COLUMN - Contains the interactive map visualization
        with col1:
            # Converts the GeoDataFrame to a GeoJSON format that Plotly can interpret
            geojson_dict = data.__geo_interface__
            
            # Get the human-readable label for the selected column
            column_label = variable_labels.get(color_column, color_column)
            
            # Prepare hover data with labels
            hover_cols = [color_column] + [col for col in numeric_columns[:3] if col != color_column]
            
            # Create a labels dictionary for better hover information
            labels = {col: variable_labels.get(col, col) for col in hover_cols + ["mun"]}
            
            # Highlight selected regions if any
            if selected_regions:
                data['is_selected'] = data['mun'].isin(selected_regions)
                hover_cols.append('is_selected')
                labels['is_selected'] = 'Selected Region'
            
            # Creates the interactive choropleth map with Plotly Express
            fig = px.choropleth_mapbox(
                data_frame=data,                     # The source dataset with both geometry and attributes
                geojson=geojson_dict,                # The geographic boundaries in GeoJSON format
                locations="id",                      # Column that links data rows to geographic features
                color=color_column,                  # Selected column to determine the color of each region
                hover_name="mun",                    # Municipality name shown in hover tooltip
                hover_data=hover_cols,               # Dynamic hover data
                color_continuous_scale=color_scale,  # Color gradient palette for the choropleth
                mapbox_style=map_style,              # Base map style from user selection
                zoom=zoom,                           # Initial zoom level from user selection
                center={"lat": data.geometry.centroid.y.mean(), 
                       "lon": data.geometry.centroid.x.mean()},  # Centers map on the data
                opacity=opacity,                     # Transparency level from user selection
                labels=labels,                       # Use dictionary of labels for better hover information
                color_continuous_midpoint=midpoint   # Center color scale at selected point
            )
            
            # Adjusts the figure layout for better display
            fig.update_layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},  # Removes unnecessary margins
                height=st.session_state.map_height,        # Dynamic height based on user preference
                autosize=True,                             # Make responsive
                coloraxis_colorbar=dict(
                    title=column_label,                    # Use the human-readable label for the color scale
                    title_side="right"                     # Position the title on the right side
                )
            )
            
            # Renders the interactive map in the app
            st.plotly_chart(fig, use_container_width=True)
            
            # Map size controls
            map_size_col1, map_size_col2, map_size_col3 = st.columns(3)
            with map_size_col1:
                if st.button("Smaller Map"):
                    st.session_state.map_height = max(400, st.session_state.map_height - 50)
                    st.rerun()
            with map_size_col2:
                if st.button("Reset Size"):
                    st.session_state.map_height = 600
                    st.rerun()
            with map_size_col3:
                if st.button("Larger Map"):
                    st.session_state.map_height = min(900, st.session_state.map_height + 50)
                    st.rerun()
        
        # RIGHT COLUMN - Contains data summary and exploration tools
        with col2:
            # Creates a heading for the data section
            st.subheader("Data Summary")
            
            # Creates a container for the statistics table that will update when the selected column changes
            stats_container = st.container()
            
            with stats_container:
                # Provides statistical summary of the selected variable with a human-readable label
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
            
            # Region comparison section (if regions selected)
            if selected_regions:
                st.subheader("Region Comparison")
                comparison_data = data[data["mun"].isin(selected_regions)]
                
                # Create comparison chart
                comparison_fig = px.bar(
                    comparison_data,
                    x="mun",
                    y=color_column,
                    title=f"Comparison of {column_label}",
                    labels={"mun": "Municipality", color_column: column_label}
                )
                st.plotly_chart(comparison_fig, use_container_width=True)
    
    with tab2:
        # DATA ANALYSIS TAB
        st.subheader("Data Exploration")
        
        # Distribution visualization
        st.write("### Distribution Analysis")
        
        # Histogram for selected variable
        hist_fig = px.histogram(
            data, 
            x=color_column,
            title=f"Distribution of {column_label}",
            labels={color_column: column_label}
        )
        st.plotly_chart(hist_fig, use_container_width=True)
        
        # Box plot for regional comparison
        if st.checkbox("Show box plot"):
            box_fig = px.box(
                data,
                y=color_column,
                labels={color_column: column_label},
                title=f"Box Plot of {column_label}"
            )
            st.plotly_chart(box_fig, use_container_width=True)
        
        # Correlation analysis
        if st.checkbox("Show correlation with other variables"):
            # Let user select variables to correlate with
            correlation_vars = st.multiselect(
                "Select variables to correlate with",
                options=[opt["value"] for opt in column_options if opt["value"] != color_column],
                default=[numeric_columns[0]] if numeric_columns and numeric_columns[0] != color_column else [],
                format_func=lambda x: variable_labels.get(x, x)
            )
            
            if correlation_vars:
                # Calculate and display correlations
                corr_data = data[[color_column] + correlation_vars].corr()[color_column].drop(color_column).reset_index()
                corr_data.columns = ["Variable", "Correlation with " + column_label]
                
                # Format correlation values
                corr_data["Correlation with " + column_label] = corr_data["Correlation with " + column_label].apply(lambda x: f"{x:.4f}")
                
                # Add human-readable labels
                corr_data["Variable Label"] = corr_data["Variable"].apply(lambda x: variable_labels.get(x, x))
                
                # Show correlation table
                st.dataframe(corr_data[["Variable Label", "Correlation with " + column_label]], use_container_width=True)
                
                # Scatter plot for the highest correlation
                top_corr_var = corr_data.iloc[0]["Variable"]
                top_corr_label = variable_labels.get(top_corr_var, top_corr_var)
                
                scatter_fig = px.scatter(
                    data,
                    x=color_column,
                    y=top_corr_var,
                    hover_name="mun",
                    labels={color_column: column_label, top_corr_var: top_corr_label},
                    title=f"Scatter Plot: {column_label} vs {top_corr_label}"
                )
                st.plotly_chart(scatter_fig, use_container_width=True)
        
        # Raw data viewer
        if st.checkbox("Show raw data"):
            # Add search filter for the raw data
            filter_mun = st.text_input("Filter by municipality name")
            
            filtered_display_data = data
            if filter_mun:
                filtered_display_data = data[data["mun"].str.contains(filter_mun, case=False)]
            
            # Select columns to display
            cols_to_display = ["mun", color_column] + [col for col in numeric_columns[:5] if col != color_column]
            
            st.dataframe(filtered_display_data[cols_to_display], use_container_width=True)
            
            st.info(f"Showing {len(filtered_display_data)} of {len(data)} regions")
    
    with tab3:
        # DOCUMENTATION TAB
        st.subheader("Documentation")
        
        # Application description
        st.markdown("""
        ### About This Application
        
        This interactive choropleth map application allows you to explore various municipal development indicators. 
        The data is displayed geographically, allowing you to identify spatial patterns and regional differences.
        
        ### Data Sources
        
        The data comes from multiple sources compiled into a single geospatial dataset. It includes:
        
        - Socioeconomic indicators
        - Development indices
        - Geographic information
        
        ### Using the Application
        
        1. **Select variables**: Use the dropdown or search in the sidebar to choose which indicator to display.
        2. **Customize the map**: Adjust the color palette, map style, and opacity.
        3. **Explore the data**: View statistics, distributions, and correlations in the Data Analysis tab.
        4. **Compare regions**: Select multiple municipalities to directly compare their values.
        5. **Download data**: Use the buttons below to download the data in CSV or GeoJSON format.
        
        ### Variable Descriptions
        """)
        
        # Display variable descriptions if available
        if variable_labels:
            # Create a searchable table of variables
            var_search = st.text_input("Search for variables")
            
            var_data = []
            for var, label in variable_labels.items():
                if var in numeric_columns and (not var_search or var_search.lower() in var.lower() or var_search.lower() in label.lower()):
                    var_data.append({"Variable Code": var, "Description": label})
            
            if var_data:
                st.dataframe(pd.DataFrame(var_data), use_container_width=True)
            else:
                st.info("No matching variables found")
        else:
            st.warning("Variable descriptions are not available")
    
    # DATA EXPORT - Creates download buttons for both CSV and GeoJSON formats
    st.markdown("---")  # Horizontal divider
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        # CSV download option (without geometry)
        csv_data = data.drop(columns='geometry').to_csv(index=False)  # Converts data to CSV format
        st.download_button(
            label="Download data as CSV",     # Button text
            data=csv_data,                    # Data to be downloaded
            file_name="map_data.csv",         # Default filename for the download
            mime="text/csv",                  # MIME type for CSV files
            key="csv_download"                # Unique key for this button
        )
    
    with export_col2:
        # GeoJSON download option (includes geometry)
        geojson_data = json.dumps(data.__geo_interface__, indent=2)  # Converts to GeoJSON with formatting
        st.download_button(
            label="Download data as GeoJSON", # Button text
            data=geojson_data,                # GeoJSON data to be downloaded
            file_name="map_data.geojson",     # Default filename for the download
            mime="application/geo+json",      # MIME type for GeoJSON files
            key="geojson_download"            # Unique key for this button
        )

# ERROR HANDLING - Catches and displays any exceptions that occur
except DataLoadError as e:
    st.error(f"Failed to load data: {str(e)}")
    st.info("Try refreshing the page or checking your internet connection.")
except VisualizationError as e:
    st.error(f"Failed to create visualization: {str(e)}")
    st.info("Try selecting a different variable or map style.")
except Exception as e:
    # Shows user-friendly error messages
    st.error(f"An error occurred: {e}")                          # Displays the specific error
    st.error("Please check the data source or report this issue.")  # Provides general guidance
    # Log the error for debugging
    logging.error(f"Unexpected error: {str(e)}", exc_info=True)

# FOOTER - Adds attribution and source information at the bottom of the page
st.markdown("---")  # Horizontal divider
st.markdown("Data source: [GitHub Project Repository](https://github.com/quarcs-lab/project2021o-notebook)")
st.markdown(f"Application last updated: {pd.Timestamp.now().strftime('%Y-%m-%d')}")