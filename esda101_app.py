# IMPORT SECTION - Libraries needed for this application
import streamlit as st             # Core Streamlit library for creating web applications
import geopandas as gpd            # Extends pandas to work with geospatial data
import plotly.express as px        # High-level interface for creating interactive visualizations
import pandas as pd                # For data manipulation and analysis
import json                        # For GeoJSON processing and export
import csv                         # For reading the data dictionary
import os                          # For file path operations
import logging                     # For error logging
import requests                    # For making HTTP requests to external APIs
from datetime import datetime      # For timestamping
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

@st.cache_data
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
    map_styles = {
        "Carto Light": "carto-positron",
        "Carto Dark": "carto-darkmatter",
        "OpenStreetMap": "open-street-map",
        "White Background": "white-bg",
        "Stamen Terrain": "stamen-terrain",
        "Stamen Toner": "stamen-toner",
        "Stamen Watercolor": "stamen-watercolor"
    }
    
    # Map style selection with descriptive labels
    style_name = st.sidebar.selectbox(
        "Map style", 
        options=list(map_styles.keys()),
        index=0,
        help="Select the background map design"
    )
    
    # Get the actual style ID for Plotly
    map_style = map_styles[style_name]
    
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
                    
                    # Create a more visually appealing display for extremes
                    extreme_col1, extreme_col2 = st.columns(2)
                    with extreme_col1:
                        st.metric(
                            "Highest value", 
                            f"{data[color_column].max():.4f}", 
                            delta=None,
                            delta_color="normal"
                        )
                        st.text(f"Region: {top_region}")
                    
                    with extreme_col2:
                        st.metric(
                            "Lowest value", 
                            f"{data[color_column].min():.4f}", 
                            delta=None,
                            delta_color="normal"
                        )
                        st.text(f"Region: {bottom_region}")
            
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
        st.header("Data Analysis & Insights")
        
        # Create analysis subtabs for better organization
        analysis_tabs = st.tabs([
            "📊 Distribution", 
            "🔄 Correlation", 
            "📈 Descriptive Statistics", 
            "🧩 Pattern Analysis",
            "🔍 Data Explorer"
        ])
        
        with analysis_tabs[0]:
            # DISTRIBUTION ANALYSIS TAB
            st.subheader("Distribution Analysis")
            st.markdown("Analyze how values are distributed across municipalities")
            
            # Distribution visualization options
            dist_type = st.radio(
                "Select visualization type:",
                options=["Histogram", "Density Plot", "Box Plot", "Violin Plot"],
                horizontal=True
            )
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                # Distribution plot controls
                st.markdown("#### Plot Controls")
                
                if dist_type == "Histogram":
                    n_bins = st.slider("Number of bins", min_value=5, max_value=50, value=20)
                    hist_norm = st.checkbox("Normalize histogram", value=False)
                    cumulative = st.checkbox("Cumulative", value=False)
                    
                    # Create the histogram with user options
                    with col1:
                        if hist_norm:
                            hist_type = "probability density" if not cumulative else "probability"
                        else:
                            hist_type = "count" if not cumulative else "cumulative"
                            
                        hist_fig = px.histogram(
                            data, 
                            x=color_column,
                            nbins=n_bins,
                            histnorm=hist_type if hist_norm else None,
                            cumulative=cumulative,
                            title=f"Distribution of {column_label}",
                            labels={color_column: column_label},
                            marginal="box" if st.checkbox("Show marginal box plot", value=False) else None,
                            color_discrete_sequence=["#3366CC"]
                        )
                        hist_fig.update_layout(bargap=0.1)
                        st.plotly_chart(hist_fig, use_container_width=True)
                
                elif dist_type == "Density Plot":
                    kde_bw = st.slider("Bandwidth", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
                    
                    # Create the density plot
                    with col1:
                        # Prepare the data for KDE (using Plotly's histogram with kernel density estimation)
                        kde_fig = px.histogram(
                            data,
                            x=color_column,
                            histnorm="probability density",
                            title=f"Density Plot of {column_label}",
                            labels={color_column: column_label},
                            color_discrete_sequence=["#3366CC"]
                        )
                        
                        kde_fig.update_traces(
                            histfunc="kde",
                            selector=dict(type='histogram'),
                            kde_smoothing=kde_bw
                        )
                        
                        # Add vertical line for mean and median
                        mean_val = data[color_column].mean()
                        median_val = data[color_column].median()
                        
                        kde_fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                                        annotation_text=f"Mean: {mean_val:.2f}", 
                                        annotation_position="top right")
                        kde_fig.add_vline(x=median_val, line_dash="dash", line_color="green", 
                                        annotation_text=f"Median: {median_val:.2f}", 
                                        annotation_position="top left")
                        
                        st.plotly_chart(kde_fig, use_container_width=True)
                
                elif dist_type == "Box Plot":
                    # Optional grouping for the box plot
                    group_vars = [col for col in data.columns if col not in numeric_columns and col != 'geometry' and data[col].nunique() < 10]
                    group_var = st.selectbox(
                        "Group by (optional):",
                        options=["None"] + group_vars,
                        format_func=lambda x: "No Grouping" if x == "None" else x
                    )
                    
                    with col1:
                        if group_var == "None":
                            box_fig = px.box(
                                data,
                                y=color_column,
                                points="all" if st.checkbox("Show all points", value=False) else "outliers",
                                labels={color_column: column_label},
                                title=f"Box Plot of {column_label}",
                                color_discrete_sequence=["#3366CC"]
                            )
                        else:
                            box_fig = px.box(
                                data,
                                x=group_var,
                                y=color_column,
                                points="all" if st.checkbox("Show all points", value=False) else "outliers",
                                labels={color_column: column_label, group_var: group_var},
                                title=f"Box Plot of {column_label} by {group_var}",
                                color=group_var
                            )
                        
                        st.plotly_chart(box_fig, use_container_width=True)
                
                elif dist_type == "Violin Plot":
                    # Optional grouping for the violin plot
                    group_vars = [col for col in data.columns if col not in numeric_columns and col != 'geometry' and data[col].nunique() < 10]
                    group_var = st.selectbox(
                        "Group by (optional):",
                        options=["None"] + group_vars,
                        format_func=lambda x: "No Grouping" if x == "None" else x
                    )
                    
                    with col1:
                        if group_var == "None":
                            violin_fig = px.violin(
                                data,
                                y=color_column,
                                box=st.checkbox("Show box plot inside", value=True),
                                points="all" if st.checkbox("Show all points", value=False) else None,
                                labels={color_column: column_label},
                                title=f"Violin Plot of {column_label}",
                                color_discrete_sequence=["#3366CC"]
                            )
                        else:
                            violin_fig = px.violin(
                                data,
                                x=group_var,
                                y=color_column,
                                box=st.checkbox("Show box plot inside", value=True),
                                points="all" if st.checkbox("Show all points", value=False) else None,
                                labels={color_column: column_label, group_var: group_var},
                                title=f"Violin Plot of {column_label} by {group_var}",
                                color=group_var
                            )
                        
                        st.plotly_chart(violin_fig, use_container_width=True)
            
            # Add distribution statistics
            st.markdown("### Distribution Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{data[color_column].mean():.4f}")
            with col2:
                st.metric("Median", f"{data[color_column].median():.4f}")
            with col3:
                st.metric("Standard Deviation", f"{data[color_column].std():.4f}")
            with col4:
                st.metric("Coefficient of Variation", f"{data[color_column].std() / data[color_column].mean():.4f}")
            
            # Additional distribution stats in an expander
            with st.expander("More Distribution Statistics"):
                dist_stats_col1, dist_stats_col2 = st.columns(2)
                
                with dist_stats_col1:
                    st.markdown("#### Central Tendency & Dispersion")
                    st.write(f"**Range:** {data[color_column].min():.4f} to {data[color_column].max():.4f}")
                    st.write(f"**Interquartile Range (IQR):** {(data[color_column].quantile(0.75) - data[color_column].quantile(0.25)):.4f}")
                    st.write(f"**Variance:** {data[color_column].var():.4f}")
                    st.write(f"**Mode:** {data[color_column].mode().iloc[0]:.4f}")
                
                with dist_stats_col2:
                    st.markdown("#### Shape Statistics")
                    st.write(f"**Skewness:** {data[color_column].skew():.4f}")
                    st.write(f"**Kurtosis:** {data[color_column].kurtosis():.4f}")
                    
                    # Interpret skewness
                    skew_val = data[color_column].skew()
                    if abs(skew_val) < 0.5:
                        skew_interpret = "approximately symmetric"
                    elif skew_val < 0:
                        skew_interpret = "negatively skewed (longer tail on left)"
                    else:
                        skew_interpret = "positively skewed (longer tail on right)"
                    
                    st.write(f"**Distribution shape:** {skew_interpret}")
                    
                    # Calculate and show percentiles
                    st.markdown("#### Percentiles")
                    percentiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
                    percentile_values = [data[color_column].quantile(p) for p in percentiles]
                    percentile_df = pd.DataFrame({
                        'Percentile': [f"{int(p*100)}%" for p in percentiles],
                        'Value': [f"{v:.4f}" for v in percentile_values]
                    })
                    st.dataframe(percentile_df, use_container_width=True)
        
        with analysis_tabs[1]:
            # CORRELATION ANALYSIS TAB
            st.subheader("Correlation Analysis")
            st.markdown("Explore relationships between variables")
            
            corr_options = st.radio(
                "Select correlation analysis type:",
                options=["Single Variable Correlation", "Correlation Matrix", "Scatter Plot Matrix"],
                horizontal=True
            )
            
            if corr_options == "Single Variable Correlation":
                # Let user select variables to correlate with
                correlation_vars = st.multiselect(
                    "Select variables to correlate with",
                    options=[opt["value"] for opt in column_options if opt["value"] != color_column],
                    default=[numeric_columns[0]] if numeric_columns and numeric_columns[0] != color_column else [],
                    format_func=lambda x: variable_labels.get(x, x)
                )
                
                corr_method = st.radio(
                    "Correlation method:",
                    options=["Pearson", "Spearman"],
                    horizontal=True,
                    help="Pearson measures linear relationships, Spearman measures monotonic relationships"
                )
                
                if correlation_vars:
                    # Calculate and display correlations
                    corr_data = data[[color_column] + correlation_vars].corr(method=corr_method.lower())[color_column].drop(color_column).reset_index()
                    corr_data.columns = ["Variable", f"{corr_method} Correlation with {column_label}"]
                    
                    # Sort by absolute correlation value
                    corr_data = corr_data.reindex(corr_data[f"{corr_method} Correlation with {column_label}"].abs().sort_values(ascending=False).index)
                    
                    # Format correlation values
                    corr_data[f"{corr_method} Correlation with {column_label}"] = corr_data[f"{corr_method} Correlation with {column_label}"].apply(lambda x: f"{x:.4f}")
                    
                    # Add human-readable labels
                    corr_data["Variable Label"] = corr_data["Variable"].apply(lambda x: variable_labels.get(x, x))
                    
                    # Show correlation table with color bars
                    st.dataframe(
                        corr_data[["Variable Label", f"{corr_method} Correlation with {column_label}"]],
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Scatter plots for top correlations
                    st.markdown("### Top Correlation Scatter Plots")
                    top_n = min(3, len(correlation_vars))
                    top_corr_vars = corr_data["Variable"].head(top_n).tolist()
                    
                    scatter_cols = st.columns(top_n)
                    
                    for i, var in enumerate(top_corr_vars):
                        var_label = variable_labels.get(var, var)
                        
                        scatter_fig = px.scatter(
                            data,
                            x=color_column,
                            y=var,
                            hover_name="mun",
                            labels={color_column: column_label, var: var_label},
                            title=f"{column_label} vs {var_label}",
                            trendline="ols" if st.checkbox("Show trend line", value=True) else None,
                            color_discrete_sequence=["#3366CC"]
                        )
                        
                        # Add annotation with correlation value
                        corr_val = data[[color_column, var]].corr(method=corr_method.lower()).iloc[0, 1]
                        scatter_fig.add_annotation(
                            x=0.5, y=0.05,
                            text=f"{corr_method} correlation: {corr_val:.4f}",
                            showarrow=False,
                            xref="paper", yref="paper",
                            bgcolor="rgba(255, 255, 255, 0.8)"
                        )
                        
                        scatter_cols[i].plotly_chart(scatter_fig, use_container_width=True)
                    
                    # Optionally add regression analysis
                    if st.checkbox("Show regression analysis", value=False):
                        st.info("""
                        **Advanced Regression Analysis**
                        
                        Advanced regression analysis requires the statsmodels library, which is not currently available in this environment.
                        
                        A full regression analysis would provide:
                        - Model coefficients and their significance
                        - R-squared and adjusted R-squared values
                        - F-statistic and p-values
                        - Regression diagnostics like residual plots
                        
                        To perform this analysis, you could export the data and use the statsmodels package in a local Python environment.
                        """)
                        
                        # Show a simple linear relationship using numpy instead
                        if len(correlation_vars) > 0:
                            top_corr_var = corr_data.iloc[0]["Variable"]
                            top_corr_label = variable_labels.get(top_corr_var, top_corr_var)
                            
                            # Create scatter plot with trend line using px
                            reg_fig = px.scatter(
                                data,
                                x=color_column,
                                y=top_corr_var,
                                hover_name="mun",
                                labels={color_column: column_label, top_corr_var: top_corr_label},
                                title=f"Simple Linear Relationship: {column_label} vs {top_corr_label}",
                                trendline="ols"  # Built-in OLS trend line from Plotly
                            )
                            
                            st.plotly_chart(reg_fig, use_container_width=True)
                            
                            # Calculate and show basic regression statistics using numpy
                            import numpy as np
                            
                            X = data[color_column].values
                            y = data[top_corr_var].values
                            
                            # Drop rows with missing values
                            mask = ~(np.isnan(X) | np.isnan(y))
                            X = X[mask]
                            y = y[mask]
                            
                            # Add a constant column for intercept
                            X_with_const = np.column_stack((np.ones(len(X)), X))
                            
                            # Calculate coefficients using normal equation
                            try:
                                beta = np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T @ y
                                intercept, slope = beta
                                
                                # Calculate predictions
                                y_pred = intercept + slope * X
                                
                                # Calculate R-squared
                                mean_y = np.mean(y)
                                ss_total = np.sum((y - mean_y) ** 2)
                                ss_residual = np.sum((y - y_pred) ** 2)
                                r_squared = 1 - (ss_residual / ss_total)
                                
                                # Display results
                                st.markdown("#### Basic Regression Results")
                                st.markdown(f"**Equation**: {top_corr_label} = {intercept:.4f} + {slope:.4f} × {column_label}")
                                st.markdown(f"**R-squared**: {r_squared:.4f}")
                                
                                # Show stats in a better format
                                reg_stats_col1, reg_stats_col2 = st.columns(2)
                                with reg_stats_col1:
                                    st.metric("Intercept", f"{intercept:.4f}")
                                    st.metric("R-squared", f"{r_squared:.4f}")
                                with reg_stats_col2:
                                    st.metric("Slope", f"{slope:.4f}")
                                    st.metric("Correlation", f"{np.corrcoef(X, y)[0,1]:.4f}")
                                
                            except np.linalg.LinAlgError:
                                st.error("Unable to calculate regression due to linear algebra error (possibly collinearity or insufficient data).")
                
            elif corr_options == "Correlation Matrix":
                # Select variables for correlation matrix
                matrix_vars = st.multiselect(
                    "Select variables for correlation matrix",
                    options=[opt["value"] for opt in column_options],
                    default=[color_column] + [numeric_columns[i] for i in range(min(5, len(numeric_columns))) if numeric_columns[i] != color_column],
                    format_func=lambda x: variable_labels.get(x, x)
                )
                
                corr_method = st.radio(
                    "Correlation method:",
                    options=["Pearson", "Spearman"],
                    horizontal=True
                )
                
                if matrix_vars and len(matrix_vars) > 1:
                    # Calculate correlation matrix
                    corr_matrix = data[matrix_vars].corr(method=corr_method.lower())
                    
                    # Create heatmap
                    corr_z = corr_matrix.values
                    corr_x = [variable_labels.get(var, var) for var in matrix_vars]
                    corr_y = corr_x
                    
                    # Create heatmap
                    heatmap_fig = px.imshow(
                        corr_z,
                        x=corr_x,
                        y=corr_y,
                        color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1,
                        title=f"{corr_method} Correlation Matrix",
                        labels=dict(color="Correlation")
                    )
                    
                    # Show correlation values in the heatmap
                    heatmap_fig.update_traces(text=[[f"{val:.2f}" for val in row] for row in corr_z], texttemplate="%{text}")
                    
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                    
                    # Show correlation matrix as a table
                    if st.checkbox("Show correlation matrix as table", value=False):
                        # Format the correlation matrix for display
                        corr_matrix_display = corr_matrix.round(4)
                        
                        # Use human readable variable labels
                        corr_matrix_display.index = [variable_labels.get(var, var) for var in corr_matrix_display.index]
                        corr_matrix_display.columns = [variable_labels.get(var, var) for var in corr_matrix_display.columns]
                        
                        st.dataframe(corr_matrix_display, use_container_width=True)
            
            elif corr_options == "Scatter Plot Matrix":
                # Select variables for scatter plot matrix
                scatter_vars = st.multiselect(
                    "Select variables for scatter plot matrix",
                    options=[opt["value"] for opt in column_options],
                    default=[color_column] + [numeric_columns[i] for i in range(min(3, len(numeric_columns))) if numeric_columns[i] != color_column],
                    format_func=lambda x: variable_labels.get(x, x),
                    help="Select 2-6 variables for best visualization"
                )
                
                if scatter_vars and 2 <= len(scatter_vars) <= 6:
                    # Create scatter plot matrix
                    splom_fig = px.scatter_matrix(
                        data,
                        dimensions=scatter_vars,
                        labels={var: variable_labels.get(var, var) for var in scatter_vars},
                        title="Scatter Plot Matrix",
                        opacity=0.5
                    )
                    
                    # Adjust layout for better visualization
                    splom_fig.update_layout(
                        width=800,
                        height=800
                    )
                    
                    st.plotly_chart(splom_fig, use_container_width=True)
                elif len(scatter_vars) > 6:
                    st.warning("Please select 6 or fewer variables for the scatter plot matrix to ensure readability.")
        
        with analysis_tabs[2]:
            # DESCRIPTIVE STATISTICS TAB
            st.subheader("Descriptive Statistics")
            st.markdown("Detailed statistical summary of the data")
            
            # Let user select variables for descriptive statistics
            desc_vars = st.multiselect(
                "Select variables for descriptive statistics",
                options=[opt["value"] for opt in column_options],
                default=[color_column],
                format_func=lambda x: variable_labels.get(x, x)
            )
            
            if desc_vars:
                # Calculate descriptive statistics
                desc_stats = data[desc_vars].describe().T
                
                # Add additional statistics
                desc_stats['skew'] = data[desc_vars].skew()
                desc_stats['kurtosis'] = data[desc_vars].kurtosis()
                desc_stats['median'] = data[desc_vars].median()
                desc_stats['CV'] = desc_stats['std'] / desc_stats['mean']  # Coefficient of variation
                
                # Reorder columns
                cols_order = ['count', 'mean', 'median', 'std', 'CV', 'min', '25%', '50%', '75%', 'max', 'skew', 'kurtosis']
                desc_stats = desc_stats[cols_order]
                
                # Format the statistics
                desc_stats = desc_stats.round(4)
                
                # Rename index with human-readable variable labels
                desc_stats.index = [variable_labels.get(var, var) for var in desc_stats.index]
                
                # Display the descriptive statistics
                st.dataframe(desc_stats, use_container_width=True)
                
                # Add download button for descriptive statistics
                csv_stats = desc_stats.reset_index().to_csv(index=False)
                st.download_button(
                    label="Download descriptive statistics as CSV",
                    data=csv_stats,
                    file_name="descriptive_statistics.csv",
                    mime="text/csv"
                )
                
                # Statistical summary visualizations
                st.markdown("### Visual Statistical Summary")
                
                # Choose visualization type
                summary_viz = st.radio(
                    "Select visualization type:",
                    options=["Bar Chart", "Radar Chart", "Parallel Coordinates"],
                    horizontal=True
                )
                
                if summary_viz == "Bar Chart":
                    # Create a bar chart of selected statistics across variables
                    stat_options = ['mean', 'median', 'std', 'min', 'max']
                    selected_stats = st.multiselect(
                        "Select statistics to visualize",
                        options=stat_options,
                        default=['mean', 'median']
                    )
                    
                    if selected_stats:
                        # Prepare data for visualization
                        viz_data = []
                        for var in desc_vars:
                            var_label = variable_labels.get(var, var)
                            for stat in selected_stats:
                                viz_data.append({
                                    'Variable': var_label,
                                    'Statistic': stat.capitalize(),
                                    'Value': desc_stats.loc[var_label, stat]
                                })
                        
                        viz_df = pd.DataFrame(viz_data)
                        
                        bar_fig = px.bar(
                            viz_df,
                            x='Variable',
                            y='Value',
                            color='Statistic',
                            barmode='group',
                            title="Statistical Summary by Variable"
                        )
                        
                        st.plotly_chart(bar_fig, use_container_width=True)
                
                elif summary_viz == "Radar Chart":
                    # Create a radar chart of standardized variables
                    if 2 <= len(desc_vars) <= 10:
                        # Calculate z-scores for radar chart
                        radar_data = {}
                        for var in desc_vars:
                            var_label = variable_labels.get(var, var)
                            # Normalize values between 0 and 1 for radar chart
                            min_val = data[var].min()
                            max_val = data[var].max()
                            radar_data[var_label] = [(val - min_val) / (max_val - min_val) if max_val > min_val else 0.5 for val in data[var]]
                        
                        radar_df = pd.DataFrame(radar_data)
                        
                        # Calculate average (normalized) value for each variable
                        radar_means = radar_df.mean().reset_index()
                        radar_means.columns = ['Variable', 'Value']
                        
                        # Create radar chart
                        radar_fig = px.line_polar(
                            radar_means,
                            r='Value',
                            theta='Variable',
                            line_close=True,
                            title="Normalized Average Values Radar Chart"
                        )
                        
                        radar_fig.update_traces(fill='toself')
                        
                        st.plotly_chart(radar_fig, use_container_width=True)
                        st.markdown("*Note: Values are normalized between 0 and 1 for each variable*")
                    else:
                        st.warning("Please select between 2 and 10 variables for the radar chart.")
                
                elif summary_viz == "Parallel Coordinates":
                    # Create parallel coordinates plot for multivariate analysis
                    if 2 <= len(desc_vars) <= 10:
                        # Optional color variable
                        color_var = st.selectbox(
                            "Color by (optional):",
                            options=["None"] + [opt["value"] for opt in column_options],
                            format_func=lambda x: "No coloring" if x == "None" else variable_labels.get(x, x)
                        )
                        
                        # Create parallel coordinates plot
                        if color_var == "None":
                            parallel_fig = px.parallel_coordinates(
                                data,
                                dimensions=desc_vars,
                                labels={var: variable_labels.get(var, var) for var in desc_vars},
                                title="Parallel Coordinates Plot"
                            )
                        else:
                            parallel_fig = px.parallel_coordinates(
                                data,
                                dimensions=desc_vars,
                                color=color_var,
                                labels={var: variable_labels.get(var, var) for var in desc_vars + [color_var]},
                                title=f"Parallel Coordinates Plot (colored by {variable_labels.get(color_var, color_var)})"
                            )
                        
                        st.plotly_chart(parallel_fig, use_container_width=True)
                    else:
                        st.warning("Please select between 2 and 10 variables for the parallel coordinates plot.")
        
        with analysis_tabs[3]:
            # PATTERN ANALYSIS TAB
            st.subheader("Pattern Analysis")
            st.markdown("Identify patterns, clusters, and outliers in the data")
            
            pattern_options = st.radio(
                "Select analysis type:",
                options=["Outlier Detection", "Clustering Analysis", "Spatial Patterns"],
                horizontal=True
            )
            
            if pattern_options == "Outlier Detection":
                # Outlier detection section
                st.markdown("### Outlier Detection")
                
                # Let user select variables for outlier detection
                outlier_vars = st.multiselect(
                    "Select variables for outlier detection",
                    options=[opt["value"] for opt in column_options],
                    default=[color_column],
                    format_func=lambda x: variable_labels.get(x, x)
                )
                
                # Outlier detection method
                outlier_method = st.radio(
                    "Detection method:",
                    options=["Z-Score", "IQR (Interquartile Range)", "Isolation Forest"],
                    horizontal=True
                )
                
                if outlier_vars:
                    if outlier_method == "Z-Score":
                        threshold = st.slider("Z-Score threshold", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
                        
                        # Calculate z-scores
                        z_scores = pd.DataFrame()
                        for var in outlier_vars:
                            var_mean = data[var].mean()
                            var_std = data[var].std()
                            z_scores[var] = (data[var] - var_mean) / var_std
                        
                        # Identify outliers
                        outliers = pd.DataFrame(index=data.index)
                        for var in outlier_vars:
                            outliers[var] = abs(z_scores[var]) > threshold
                        
                        # Count outliers per variable
                        outlier_counts = outliers.sum().reset_index()
                        outlier_counts.columns = ['Variable', 'Number of Outliers']
                        
                        # Add human-readable variable labels
                        outlier_counts['Variable Label'] = outlier_counts['Variable'].apply(lambda x: variable_labels.get(x, x))
                        
                        # Calculate percentage of outliers
                        outlier_counts['Percentage of Data'] = (outlier_counts['Number of Outliers'] / len(data) * 100).round(2)
                        outlier_counts['Percentage of Data'] = outlier_counts['Percentage of Data'].apply(lambda x: f"{x}%")
                        
                        # Show outlier summary
                        st.markdown(f"Using Z-Score method with threshold {threshold}")
                        st.dataframe(outlier_counts[['Variable Label', 'Number of Outliers', 'Percentage of Data']], use_container_width=True)
                        
                        # Outlier visualization
                        if outlier_vars:
                            # Create box plots with outliers highlighted
                            for var in outlier_vars:
                                var_label = variable_labels.get(var, var)
                                
                                # Create two datasets - outliers and non-outliers
                                outlier_mask = abs(z_scores[var]) > threshold
                                outlier_data = data[outlier_mask]
                                non_outlier_data = data[~outlier_mask]
                                
                                fig = px.box(
                                    data,
                                    y=var,
                                    labels={var: var_label},
                                    title=f"Box Plot with Outliers Highlighted: {var_label}",
                                    points=False
                                )
                                
                                # Add non-outliers as jittered points
                                fig.add_trace(
                                    px.strip(
                                        non_outlier_data,
                                        y=var,
                                        color_discrete_sequence=["blue"]
                                    ).data[0]
                                )
                                
                                # Add outliers as highlighted points
                                if not outlier_data.empty:
                                    fig.add_trace(
                                        px.scatter(
                                            outlier_data,
                                            y=var,
                                            text="mun",
                                            color_discrete_sequence=["red"]
                                        ).data[0]
                                    )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # List of top outliers
                            if st.checkbox("Show municipalities with most outliers", value=True):
                                # Count outliers per municipality
                                outlier_rows = outliers.sum(axis=1).reset_index()
                                outlier_rows.columns = ['index', 'Number of Variables with Outliers']
                                
                                # Get municipality names
                                outlier_rows['Municipality'] = outlier_rows['index'].apply(lambda idx: data.loc[idx, 'mun'])
                                
                                # Sort and show top outliers
                                top_outlier_rows = outlier_rows[outlier_rows['Number of Variables with Outliers'] > 0].sort_values(
                                    'Number of Variables with Outliers', ascending=False
                                ).head(10)
                                
                                if not top_outlier_rows.empty:
                                    st.markdown("### Top Municipalities with Outliers")
                                    st.dataframe(top_outlier_rows[['Municipality', 'Number of Variables with Outliers']], use_container_width=True)
                                else:
                                    st.info("No outliers detected with current threshold.")
                    
                    elif outlier_method == "IQR (Interquartile Range)":
                        iqr_factor = st.slider("IQR factor", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
                        
                        # Calculate IQR bounds
                        outliers = pd.DataFrame(index=data.index)
                        for var in outlier_vars:
                            q1 = data[var].quantile(0.25)
                            q3 = data[var].quantile(0.75)
                            iqr = q3 - q1
                            lower_bound = q1 - (iqr_factor * iqr)
                            upper_bound = q3 + (iqr_factor * iqr)
                            
                            outliers[var] = (data[var] < lower_bound) | (data[var] > upper_bound)
                        
                        # Count outliers per variable
                        outlier_counts = outliers.sum().reset_index()
                        outlier_counts.columns = ['Variable', 'Number of Outliers']
                        
                        # Add human-readable variable labels
                        outlier_counts['Variable Label'] = outlier_counts['Variable'].apply(lambda x: variable_labels.get(x, x))
                        
                        # Calculate percentage of outliers
                        outlier_counts['Percentage of Data'] = (outlier_counts['Number of Outliers'] / len(data) * 100).round(2)
                        outlier_counts['Percentage of Data'] = outlier_counts['Percentage of Data'].apply(lambda x: f"{x}%")
                        
                        # Show outlier summary
                        st.markdown(f"Using IQR method with factor {iqr_factor}")
                        st.dataframe(outlier_counts[['Variable Label', 'Number of Outliers', 'Percentage of Data']], use_container_width=True)
                        
                        # Outlier visualization
                        if outlier_vars:
                            # Create box plots with outliers highlighted
                            for var in outlier_vars:
                                var_label = variable_labels.get(var, var)
                                
                                # Calculate bounds for this variable
                                q1 = data[var].quantile(0.25)
                                q3 = data[var].quantile(0.75)
                                iqr = q3 - q1
                                lower_bound = q1 - (iqr_factor * iqr)
                                upper_bound = q3 + (iqr_factor * iqr)
                                
                                # Create two datasets - outliers and non-outliers
                                outlier_mask = (data[var] < lower_bound) | (data[var] > upper_bound)
                                outlier_data = data[outlier_mask]
                                non_outlier_data = data[~outlier_mask]
                                
                                fig = px.box(
                                    data,
                                    y=var,
                                    labels={var: var_label},
                                    title=f"Box Plot with Outliers Highlighted: {var_label}",
                                    points=False
                                )
                                
                                # Add non-outliers as jittered points
                                fig.add_trace(
                                    px.strip(
                                        non_outlier_data,
                                        y=var,
                                        color_discrete_sequence=["blue"]
                                    ).data[0]
                                )
                                
                                # Add outliers as highlighted points
                                if not outlier_data.empty:
                                    fig.add_trace(
                                        px.scatter(
                                            outlier_data,
                                            y=var,
                                            text="mun",
                                            color_discrete_sequence=["red"]
                                        ).data[0]
                                    )
                                
                                # Add lines for bounds
                                fig.add_hline(y=lower_bound, line_dash="dash", line_color="red", 
                                              annotation_text=f"Lower bound: {lower_bound:.2f}", 
                                              annotation_position="bottom right")
                                fig.add_hline(y=upper_bound, line_dash="dash", line_color="red", 
                                              annotation_text=f"Upper bound: {upper_bound:.2f}", 
                                              annotation_position="top right")
                                
                                st.plotly_chart(fig, use_container_width=True)
                    
                    elif outlier_method == "Isolation Forest":
                        contamination = st.slider("Expected proportion of outliers", min_value=0.01, max_value=0.3, value=0.05, step=0.01)
                        
                        try:
                            from sklearn.ensemble import IsolationForest
                            
                            # Prepare data for Isolation Forest
                            X = data[outlier_vars].copy()
                            
                            # Train Isolation Forest model
                            iso_forest = IsolationForest(contamination=contamination, random_state=42)
                            outlier_pred = iso_forest.fit_predict(X)
                            
                            # Convert predictions to outlier mask (-1 for outliers, 1 for inliers)
                            outlier_mask = outlier_pred == -1
                            
                            # Get outlier municipalities
                            outlier_municipalities = data.loc[outlier_mask, 'mun'].tolist()
                            
                            # Show outlier summary
                            st.markdown(f"Using Isolation Forest method with contamination {contamination}")
                            st.metric("Number of Outliers Detected", len(outlier_municipalities))
                            st.metric("Percentage of Data", f"{(len(outlier_municipalities) / len(data) * 100):.2f}%")
                            
                            # Visualize outliers
                            if len(outlier_vars) >= 2:
                                # Create scatter plot for the first two variables
                                var1 = outlier_vars[0]
                                var2 = outlier_vars[1]
                                
                                var1_label = variable_labels.get(var1, var1)
                                var2_label = variable_labels.get(var2, var2)
                                
                                # Create datasets
                                outlier_data = data[outlier_mask]
                                non_outlier_data = data[~outlier_mask]
                                
                                # Create base scatter plot
                                fig = px.scatter(
                                    non_outlier_data,
                                    x=var1,
                                    y=var2,
                                    labels={var1: var1_label, var2: var2_label},
                                    title=f"Isolation Forest Outliers: {var1_label} vs {var2_label}",
                                    color_discrete_sequence=["blue"]
                                )
                                
                                # Add outliers
                                if not outlier_data.empty:
                                    outlier_scatter = px.scatter(
                                        outlier_data,
                                        x=var1,
                                        y=var2,
                                        hover_name="mun",
                                        color_discrete_sequence=["red"]
                                    )
                                    
                                    for trace in outlier_scatter.data:
                                        fig.add_trace(trace)
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Display list of outlier municipalities
                                if outlier_municipalities:
                                    st.markdown("### Municipalities Identified as Outliers")
                                    
                                    # Split into columns for better display
                                    num_cols = 3
                                    mun_cols = st.columns(num_cols)
                                    
                                    for i, mun in enumerate(sorted(outlier_municipalities)):
                                        mun_cols[i % num_cols].write(f"- {mun}")
                                else:
                                    st.info("No outliers detected with current settings.")
                                
                            elif len(outlier_vars) == 1:
                                # Create simple histogram for one variable
                                var = outlier_vars[0]
                                var_label = variable_labels.get(var, var)
                                
                                # Create datasets
                                outlier_data = data[outlier_mask]
                                non_outlier_data = data[~outlier_mask]
                                
                                fig = px.histogram(
                                    non_outlier_data,
                                    x=var,
                                    labels={var: var_label},
                                    title=f"Isolation Forest Outliers: {var_label}",
                                    color_discrete_sequence=["blue"]
                                )
                                
                                # Add outliers as scatter points
                                if not outlier_data.empty:
                                    outlier_scatter = px.scatter(
                                        outlier_data,
                                        x=var,
                                        y=[0] * len(outlier_data),  # Place at bottom of histogram
                                        hover_name="mun",
                                        color_discrete_sequence=["red"],
                                        opacity=0.7,
                                        size=[10] * len(outlier_data)
                                    )
                                    
                                    for trace in outlier_scatter.data:
                                        fig.add_trace(trace)
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                        except ImportError:
                            st.error("scikit-learn is required for Isolation Forest analysis but is not available.")
            
            elif pattern_options == "Clustering Analysis":
                # Clustering analysis section
                st.markdown("### Clustering Analysis")
                
                # Let user select variables for clustering
                cluster_vars = st.multiselect(
                    "Select variables for clustering",
                    options=[opt["value"] for opt in column_options],
                    default=[color_column] + [numeric_columns[i] for i in range(min(2, len(numeric_columns))) if numeric_columns[i] != color_column],
                    format_func=lambda x: variable_labels.get(x, x)
                )
                
                if cluster_vars and len(cluster_vars) >= 2:
                    # Clustering method
                    cluster_method = st.radio(
                        "Clustering method:",
                        options=["K-Means", "Hierarchical"],
                        horizontal=True
                    )
                    
                    # Number of clusters
                    n_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=3)
                    
                    try:
                        from sklearn.preprocessing import StandardScaler
                        from sklearn.cluster import KMeans, AgglomerativeClustering
                        
                        # Prepare data for clustering
                        X = data[cluster_vars].copy()
                        
                        # Standardize the data
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        # Perform clustering
                        if cluster_method == "K-Means":
                            clustering = KMeans(n_clusters=n_clusters, random_state=42)
                        else:  # Hierarchical
                            clustering = AgglomerativeClustering(n_clusters=n_clusters)
                        
                        # Fit the model and predict clusters
                        cluster_labels = clustering.fit_predict(X_scaled)
                        
                        # Add cluster labels to the data
                        data_with_clusters = data.copy()
                        data_with_clusters['Cluster'] = cluster_labels
                        data_with_clusters['Cluster'] = data_with_clusters['Cluster'].astype(str)
                        
                        # Show cluster summary
                        st.markdown(f"Using {cluster_method} with {n_clusters} clusters")
                        
                        # Count municipalities per cluster
                        cluster_counts = data_with_clusters['Cluster'].value_counts().reset_index()
                        cluster_counts.columns = ['Cluster', 'Number of Municipalities']
                        
                        # Calculate percentage
                        cluster_counts['Percentage'] = (cluster_counts['Number of Municipalities'] / len(data) * 100).round(2)
                        cluster_counts['Percentage'] = cluster_counts['Percentage'].apply(lambda x: f"{x}%")
                        
                        # Show cluster counts
                        st.dataframe(cluster_counts, use_container_width=True)
                        
                        # Cluster visualization
                        if len(cluster_vars) >= 2:
                            # Select first two variables for visualization
                            viz_vars = st.multiselect(
                                "Select variables for visualization (2 required)",
                                options=cluster_vars,
                                default=cluster_vars[:2],
                                format_func=lambda x: variable_labels.get(x, x)
                            )
                            
                            if len(viz_vars) == 2:
                                var1, var2 = viz_vars
                                var1_label = variable_labels.get(var1, var1)
                                var2_label = variable_labels.get(var2, var2)
                                
                                # Create scatter plot
                                cluster_fig = px.scatter(
                                    data_with_clusters,
                                    x=var1,
                                    y=var2,
                                    color='Cluster',
                                    hover_name="mun",
                                    labels={var1: var1_label, var2: var2_label},
                                    title=f"Cluster Results: {var1_label} vs {var2_label}"
                                )
                                
                                st.plotly_chart(cluster_fig, use_container_width=True)
                                
                                # Feature distribution by cluster
                                st.markdown("### Feature Distribution by Cluster")
                                
                                # Select a variable to analyze by cluster
                                analyze_var = st.selectbox(
                                    "Select variable to analyze by cluster",
                                    options=cluster_vars,
                                    format_func=lambda x: variable_labels.get(x, x)
                                )
                                
                                analyze_var_label = variable_labels.get(analyze_var, analyze_var)
                                
                                # Create box plot by cluster
                                cluster_box_fig = px.box(
                                    data_with_clusters,
                                    x='Cluster',
                                    y=analyze_var,
                                    color='Cluster',
                                    labels={'Cluster': 'Cluster', analyze_var: analyze_var_label},
                                    title=f"{analyze_var_label} by Cluster"
                                )
                                
                                st.plotly_chart(cluster_box_fig, use_container_width=True)
                                
                                # Cluster characteristics
                                st.markdown("### Cluster Characteristics")
                                
                                # Calculate cluster centers (mean of each feature per cluster)
                                cluster_centers = data_with_clusters.groupby('Cluster')[cluster_vars].mean()
                                
                                # Standardize centers for better comparison
                                centers_std = pd.DataFrame(index=cluster_centers.index)
                                for var in cluster_vars:
                                    global_mean = data[var].mean()
                                    global_std = data[var].std()
                                    centers_std[var] = (cluster_centers[var] - global_mean) / global_std
                                
                                # Rename columns to human-readable labels
                                centers_std.columns = [variable_labels.get(var, var) for var in cluster_vars]
                                
                                # Create heatmap of standardized cluster centers
                                centers_fig = px.imshow(
                                    centers_std.values,
                                    x=centers_std.columns,
                                    y=centers_std.index,
                                    color_continuous_scale="RdBu_r",
                                    title="Standardized Cluster Centers Heatmap",
                                    labels=dict(x="Variable", y="Cluster", color="Std. Value"),
                                    aspect="auto"
                                )
                                
                                # Show value in heatmap cells
                                centers_fig.update_traces(text=[[f"{val:.2f}" for val in row] for row in centers_std.values], texttemplate="%{text}")
                                
                                st.plotly_chart(centers_fig, use_container_width=True)
                                
                                # Create radar chart for each cluster
                                if len(cluster_vars) <= 8:  # Limit to avoid overcrowded radar charts
                                    st.markdown("### Cluster Profiles (Radar Charts)")
                                    
                                    # Normalize data between 0 and 1 for radar charts
                                    radar_data = pd.DataFrame(index=cluster_centers.index)
                                    for var in cluster_vars:
                                        min_val = data[var].min()
                                        max_val = data[var].max()
                                        if max_val > min_val:
                                            radar_data[var] = (cluster_centers[var] - min_val) / (max_val - min_val)
                                        else:
                                            radar_data[var] = 0.5  # Default if no range
                                    
                                    # Rename columns to human-readable labels
                                    radar_data.columns = [variable_labels.get(var, var) for var in cluster_vars]
                                    
                                    # Create a radar chart for each cluster
                                    radar_list = []
                                    for cluster in radar_data.index:
                                        cluster_radar = pd.DataFrame({
                                            'Variable': radar_data.columns,
                                            'Value': radar_data.loc[cluster],
                                            'Cluster': cluster
                                        })
                                        radar_list.append(cluster_radar)
                                    
                                    radar_df = pd.concat(radar_list)
                                    
                                    radar_fig = px.line_polar(
                                        radar_df,
                                        r='Value',
                                        theta='Variable',
                                        color='Cluster',
                                        line_close=True,
                                        title="Cluster Profiles"
                                    )
                                    
                                    radar_fig.update_traces(fill='toself')
                                    
                                    st.plotly_chart(radar_fig, use_container_width=True)
                                    st.markdown("*Note: Values are normalized between 0 and 1 for each variable*")
                                
                                # List of municipalities in each cluster
                                if st.checkbox("Show municipalities by cluster", value=False):
                                    # Create expanders for each cluster
                                    for cluster in sorted(data_with_clusters['Cluster'].unique()):
                                        with st.expander(f"Cluster {cluster} Municipalities"):
                                            cluster_munis = data_with_clusters[data_with_clusters['Cluster'] == cluster]['mun'].sort_values().tolist()
                                            
                                            # Display in columns for better layout
                                            num_cols = 3
                                            mun_cols = st.columns(num_cols)
                                            
                                            for i, mun in enumerate(cluster_munis):
                                                mun_cols[i % num_cols].write(f"- {mun}")
                            
                            else:
                                st.warning("Please select exactly 2 variables for visualization.")
                        
                        # Add download button for data with clusters
                        csv_clusters = data_with_clusters[['mun', 'Cluster'] + cluster_vars].to_csv(index=False)
                        st.download_button(
                            label="Download data with cluster assignments",
                            data=csv_clusters,
                            file_name="data_with_clusters.csv",
                            mime="text/csv"
                        )
                        
                    except ImportError:
                        st.error("scikit-learn is required for clustering analysis but is not available.")
                
                else:
                    st.warning("Please select at least 2 variables for clustering analysis.")
            
            elif pattern_options == "Spatial Patterns":
                # Spatial patterns analysis
                st.markdown("### Spatial Pattern Analysis")
                
                # Auto-correlation options
                st.markdown("#### Spatial Distribution")
                
                if hasattr(data, 'geometry'):
                    # Create a simple choropleth map with the color variable
                    spatial_fig = px.choropleth_mapbox(
                        data,
                        geojson=data.__geo_interface__,
                        locations="id",
                        color=color_column,
                        color_continuous_scale="viridis",
                        mapbox_style="carto-positron",
                        zoom=4,
                        center={"lat": data.geometry.centroid.y.mean(), "lon": data.geometry.centroid.x.mean()},
                        opacity=0.7,
                        labels={color_column: variable_labels.get(color_column, color_column)},
                        title=f"Spatial Distribution of {variable_labels.get(color_column, color_column)}"
                    )
                    
                    st.plotly_chart(spatial_fig, use_container_width=True)
                    
                    # Explanation of spatial patterns
                    st.markdown("""
                    #### Interpreting Spatial Patterns:
                    
                    1. **Clustering**: Are similar values located close together? This suggests spatial autocorrelation.
                    2. **Dispersion**: Are similar values scattered randomly? This suggests lack of spatial dependence.
                    3. **Gradients**: Is there a directional trend (e.g., North-South, urban-rural)?
                    4. **Hot/Cold Spots**: Are there areas with unusually high or low values?
                    """)
                    
                    # Spatial outliers section
                    st.markdown("#### Spatial Outliers")
                    st.markdown("Regions with values significantly different from their neighbors")
                    
                    # Calculate local spatial statistics
                    st.info("""
                    To perform more advanced spatial statistics (like Moran's I, LISA, or Getis-Ord G*),
                    you would need to define spatial weights based on adjacency or distance between municipalities.
                    
                    This would require additional spatial analysis libraries like PySAL, which are not currently integrated.
                    """)
                    
                    # Simple spatial outlier identification
                    st.markdown("#### Identify Potential Spatial Outliers")
                    
                    # This is a simple approach to identify potential spatial outliers
                    if st.button("Calculate potential spatial outliers"):
                        st.warning("This is a simplified approach that uses distance-based calculations")
                        
                        try:
                            # Calculate centroids
                            centroids = data.geometry.centroid
                            
                            # Create a distance matrix (simplified approach)
                            from scipy.spatial.distance import pdist, squareform
                            
                            # Extract centroid coordinates
                            coords = np.column_stack((centroids.x, centroids.y))
                            
                            # Calculate distance matrix
                            dist_matrix = squareform(pdist(coords))
                            
                            # Get the k nearest neighbors for each point
                            k = min(5, len(data) - 1)
                            
                            # For each municipality, compare its value to its neighbors
                            spatial_outliers = []
                            
                            for i in range(len(data)):
                                # Get indices of k nearest neighbors
                                neighbor_indices = np.argsort(dist_matrix[i])[1:k+1]  # Skip the first (itself)
                                
                                # Get the values of the neighbors
                                neighbor_values = data.iloc[neighbor_indices][color_column].values
                                
                                # Calculate z-score compared to neighbors
                                neighbor_mean = np.mean(neighbor_values)
                                neighbor_std = np.std(neighbor_values)
                                
                                if neighbor_std > 0:
                                    z_score = (data.iloc[i][color_column] - neighbor_mean) / neighbor_std
                                    
                                    # If z-score is high, it's a potential spatial outlier
                                    if abs(z_score) > 2:
                                        spatial_outliers.append({
                                            'Municipality': data.iloc[i]['mun'],
                                            'Value': data.iloc[i][color_column],
                                            'Neighbor Avg': neighbor_mean,
                                            'Z-Score': z_score
                                        })
                            
                            if spatial_outliers:
                                # Create dataframe of spatial outliers
                                spatial_outliers_df = pd.DataFrame(spatial_outliers)
                                
                                # Sort by absolute z-score
                                spatial_outliers_df = spatial_outliers_df.sort_values('Z-Score', key=abs, ascending=False)
                                
                                # Format values
                                spatial_outliers_df['Value'] = spatial_outliers_df['Value'].round(4)
                                spatial_outliers_df['Neighbor Avg'] = spatial_outliers_df['Neighbor Avg'].round(4)
                                spatial_outliers_df['Z-Score'] = spatial_outliers_df['Z-Score'].round(2)
                                
                                # Display the results
                                st.dataframe(spatial_outliers_df, use_container_width=True)
                                
                                # Add spatial outliers to the map
                                outlier_municipalities = spatial_outliers_df['Municipality'].tolist()
                                data['is_spatial_outlier'] = data['mun'].isin(outlier_municipalities)
                                
                                # Create a new map highlighting spatial outliers
                                outlier_fig = px.choropleth_mapbox(
                                    data,
                                    geojson=data.__geo_interface__,
                                    locations="id",
                                    color="is_spatial_outlier",
                                    color_discrete_map={True: "red", False: "lightgrey"},
                                    mapbox_style="carto-positron",
                                    zoom=4,
                                    center={"lat": data.geometry.centroid.y.mean(), "lon": data.geometry.centroid.x.mean()},
                                    opacity=0.7,
                                    hover_name="mun",
                                    title="Potential Spatial Outliers"
                                )
                                
                                st.plotly_chart(outlier_fig, use_container_width=True)
                                
                            else:
                                st.info("No significant spatial outliers detected.")
                            
                        except Exception as e:
                            st.error(f"Error in spatial outlier calculation: {str(e)}")
                
                else:
                    st.warning("Geometry data is required for spatial pattern analysis.")
        
        with analysis_tabs[4]:
            # DATA EXPLORER TAB
            st.subheader("Data Explorer")
            st.markdown("Browse, filter, and export the dataset")
            
            # Advanced filtering section
            st.markdown("### Advanced Filtering")
            
            # Create filter controls
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                # Text filter
                filter_mun = st.text_input("Filter by municipality name")
                
                # Numeric range filter
                numeric_filter_var = st.selectbox(
                    "Filter by numeric variable",
                    options=["None"] + numeric_columns,
                    format_func=lambda x: "No filter" if x == "None" else variable_labels.get(x, x)
                )
                
                if numeric_filter_var != "None":
                    min_val = float(data[numeric_filter_var].min())
                    max_val = float(data[numeric_filter_var].max())
                    
                    num_range = st.slider(
                        f"Range for {variable_labels.get(numeric_filter_var, numeric_filter_var)}",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val)
                    )
            
            with filter_col2:
                # Categorical filter
                cat_columns = [col for col in data.columns if col not in numeric_columns and col != 'geometry' and data[col].nunique() < 20]
                
                cat_filter_var = st.selectbox(
                    "Filter by categorical variable",
                    options=["None"] + cat_columns,
                    format_func=lambda x: "No filter" if x == "None" else x
                )
                
                if cat_filter_var != "None":
                    cat_values = st.multiselect(
                        f"Select values for {cat_filter_var}",
                        options=sorted(data[cat_filter_var].unique()),
                        default=list(data[cat_filter_var].unique())
                    )
            
            # Apply filters
            filtered_data = data.copy()
            
            # Municipality name filter
            if filter_mun:
                filtered_data = filtered_data[filtered_data["mun"].str.contains(filter_mun, case=False)]
            
            # Numeric range filter
            if numeric_filter_var != "None":
                filtered_data = filtered_data[(filtered_data[numeric_filter_var] >= num_range[0]) & 
                                           (filtered_data[numeric_filter_var] <= num_range[1])]
            
            # Categorical filter
            if cat_filter_var != "None" and cat_values:
                filtered_data = filtered_data[filtered_data[cat_filter_var].isin(cat_values)]
            
            # Show filter summary
            st.markdown(f"### Filtered Data ({len(filtered_data)} of {len(data)} regions)")
            
            # Column selection
            st.markdown("### Select Columns to Display")
            
            # Default columns to display
            default_display_cols = ["mun", color_column] + [col for col in numeric_columns[:5] if col != color_column]
            
            # Let user select columns
            display_cols = st.multiselect(
                "Choose columns to display",
                options=[col for col in data.columns if col != 'geometry'],
                default=default_display_cols,
                format_func=lambda x: variable_labels.get(x, x) if x in variable_labels else x
            )
            
            if not display_cols:
                display_cols = default_display_cols
            
            # Display the filtered and column-selected data
            st.dataframe(filtered_data[display_cols], use_container_width=True)
            
            # Export options
            st.markdown("### Export Data")
            
            export_options = st.radio(
                "Export format:",
                options=["CSV", "Excel", "GeoJSON"],
                horizontal=True
            )
            
            if export_options == "CSV":
                # Export to CSV
                csv_data = filtered_data[display_cols].to_csv(index=False)
                st.download_button(
                    label="Download filtered data as CSV",
                    data=csv_data,
                    file_name="filtered_data.csv",
                    mime="text/csv"
                )
            
            elif export_options == "Excel":
                # Export to Excel (mock-up since we can't create actual Excel files in Streamlit)
                st.info("Excel export option is currently not supported in this environment.")
                st.download_button(
                    label="Download filtered data as CSV (Excel format not available)",
                    data=filtered_data[display_cols].to_csv(index=False),
                    file_name="filtered_data.csv",
                    mime="text/csv"
                )
            
            elif export_options == "GeoJSON":
                # Export to GeoJSON (if geometry is available)
                if hasattr(filtered_data, 'geometry'):
                    # Prepare GeoJSON with selected attributes
                    export_data = filtered_data[display_cols + ['geometry']]
                    geojson_data = json.dumps(export_data.__geo_interface__, indent=2)
                    
                    st.download_button(
                        label="Download filtered data as GeoJSON",
                        data=geojson_data,
                        file_name="filtered_data.geojson",
                        mime="application/geo+json"
                    )
                else:
                    st.warning("Geometry data is required for GeoJSON export.")
                    st.download_button(
                        label="Download filtered data as CSV (GeoJSON not available)",
                        data=filtered_data[display_cols].to_csv(index=False),
                        file_name="filtered_data.csv",
                        mime="text/csv"
                    )
    
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

# Add documentation about map styles
# FOOTER - Adds attribution and source information at the bottom of the page
st.markdown("---")  # Horizontal divider
st.markdown("Data source: [GitHub Project Repository](https://github.com/quarcs-lab/project2021o-notebook)")
st.markdown(f"Application last updated: {pd.Timestamp.now().strftime('%Y-%m-%d')}")