# IMPORT SECTION - Libraries needed for this application
import streamlit as st
import geopandas as gpd
import plotly.express as px
import pandas as pd
import json
import os
import logging
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Custom exception classes for better error handling
class AppError(Exception):
    """Base exception class for application errors."""
    pass

class DataLoadError(AppError):
    """Exception raised for errors in data loading."""
    pass

# PAGE CONFIGURATION
st.set_page_config(page_title="Choropleth Map Visualization", layout="wide", initial_sidebar_state="expanded")

# Initialize session state
if "map_height" not in st.session_state:
    st.session_state.map_height = 600
if "selected_regions" not in st.session_state:
    st.session_state.selected_regions = []
if "show_tutorial" not in st.session_state:
    st.session_state.show_tutorial = True

# APPLICATION TITLE AND DESCRIPTION
st.title("Interactive Choropleth Map")
st.markdown("""
This application displays a choropleth map showing the Index of Sustainable Development at the municipal level. 
The application includes a data dictionary with 139 variables.
""")

# SIDEBAR SETUP
st.sidebar.header("Map Controls")

# DATA LOADING FUNCTIONS
@st.cache_data(ttl=3600, max_entries=10)
def load_data():
    """Loads and prepares geographic data."""
    try:
        data_path = "map_and_data.geojson"
        if os.path.exists(data_path):
            data = gpd.read_file(data_path)
            logging.info("Loaded data from local file")
        else:
            with st.spinner("Loading geographic data from remote source..."):
                data = gpd.read_file('https://github.com/quarcs-lab/project2021o-notebook/raw/main/map_and_data.geojson')
                data.to_file(data_path, driver="GeoJSON")
                logging.info("Saved data to local file")
        data = data.to_crs(epsg=4326)
        data["id"] = data.index.astype(str)
        return data
    except Exception as e:
        logging.error(f"Failed to load data: {e}", exc_info=True)
        raise DataLoadError(f"Could not load geographic data: {e}")

@st.cache_data
def load_data_dictionary() -> Dict[str, str]:
    """Loads and parses the data dictionary."""
    try:
        variable_labels = {}
        if os.path.exists('dataDefinitions.csv'):
            with open('dataDefinitions.csv', 'r', encoding='utf-8') as f:
                reader = pd.read_csv(f)  # Use pandas for simpler CSV reading
                if 'Variable' in reader.columns and 'Label' in reader.columns:
                    variable_labels = dict(zip(reader['Variable'], reader['Label']))
            logging.info(f"Loaded {len(variable_labels)} variable definitions")
            return variable_labels
        else:
            logging.warning("Data dictionary file not found")
            return {}
    except Exception as e:
        logging.error(f"Error loading data dictionary: {e}", exc_info=True)
        st.warning(f"Could not load data dictionary: {e}. Using original column names.")
        return {}

# MAIN APPLICATION BLOCK
try:
    data = load_data()
    variable_labels = load_data_dictionary()

    st.sidebar.subheader("Dataset Information")
    st.sidebar.info(f"Number of regions: {len(data)}")

    show_tutorial = st.sidebar.checkbox("Show tutorial", value=st.session_state.show_tutorial)
    st.session_state.show_tutorial = show_tutorial

    if show_tutorial:
        st.sidebar.info("""**How to use:** Select a variable, adjust map style, explore statistics, compare regions, download data.""")

    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    column_options = [{"label": variable_labels.get(col, col), "value": col} for col in numeric_columns]
    column_options.sort(key=lambda x: x["label"])

    # USER CONTROLS
    search_term = st.sidebar.text_input("Search variables", "")
    filtered_options = [opt for opt in column_options if search_term.lower() in opt["label"].lower() or search_term.lower() in opt["value"].lower()] if search_term else column_options

    if not filtered_options and search_term:
        st.sidebar.warning(f"No variables matching '{search_term}'")
        filtered_options = column_options  # Show all if no match

    default_index = next((i for i, option in enumerate(filtered_options) if option["value"] == "imds"), 0)

    color_column = st.sidebar.selectbox(
        "Select data to visualize",
        options=[option["value"] for option in filtered_options],
        index=default_index,
        format_func=lambda x: variable_labels.get(x, x),
        help="Choose data to display. Hover over each municipality to see its value."
    )

    map_styles = {"Carto Light": "carto-positron", "Carto Dark": "carto-darkmatter", "OpenStreetMap": "open-street-map", "White Background": "white-bg"}
    style_name = st.sidebar.selectbox("Map style", options=list(map_styles.keys()), index=0, help="Select background map design")
    map_style = map_styles[style_name]

    color_scales = ["viridis", "plasma", "inferno", "magma", "cividis", "twilight", "RdBu_r", "Blues", "Greens", "Reds", "YlOrRd"]
    color_scale = st.sidebar.selectbox("Color palette", options=color_scales, index=0, help="Choose the color scheme")

    use_median = st.sidebar.checkbox("Center color scale at median", value=True)
    midpoint = data[color_column].median() if use_median else st.sidebar.slider("Color scale midpoint", float(data[color_column].min()), float(data[color_column].max()), float(data[color_column].median()))

    opacity = st.sidebar.slider("Map opacity", min_value=0.0, max_value=1.0, value=0.6, step=0.1, help="Adjust transparency")
    zoom = st.sidebar.slider("Zoom level", min_value=3.0, max_value=10.0, value=4.5, step=0.5, help="Adjust zoom")

    st.sidebar.subheader("Compare Regions")
    selected_regions = st.sidebar.multiselect(
        "Select regions to compare",
        options=sorted(data["mun"].unique()),
        default=st.session_state.selected_regions,
        help="Choose municipalities to compare"
    )
    st.session_state.selected_regions = selected_regions

    tab1, tab2, tab3 = st.tabs(["Map Visualization", "Data Analysis", "Documentation"])

    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            geojson_dict = data.__geo_interface__
            column_label = variable_labels.get(color_column, color_column)
            hover_cols = [color_column] + [col for col in numeric_columns[:3] if col != color_column]
            labels = {col: variable_labels.get(col, col) for col in hover_cols + ["mun"]}

            if selected_regions:
                data['is_selected'] = data['mun'].isin(selected_regions)
                hover_cols.append('is_selected')
                labels['is_selected'] = 'Selected Region'

            fig = px.choropleth_mapbox(
                data_frame=data,
                geojson=geojson_dict,
                locations="id",
                color=color_column,
                hover_name="mun",
                hover_data=hover_cols,
                color_continuous_scale=color_scale,
                mapbox_style=map_style,
                zoom=zoom,
                center={"lat": data.geometry.centroid.y.mean(), "lon": data.geometry.centroid.x.mean()},
                opacity=opacity,
                labels=labels,
                color_continuous_midpoint=midpoint
            )

            fig.update_layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                height=st.session_state.map_height,
                autosize=True,
                coloraxis_colorbar=dict(title=column_label, title_side="right")
            )
            st.plotly_chart(fig, use_container_width=True)

            map_size_col1, map_size_col2, map_size_col3 = st.columns(3)
            with map_size_col1:
                if st.button("Smaller"): st.session_state.map_height = max(400, st.session_state.map_height - 50); st.rerun()
            with map_size_col2:
                if st.button("Reset Size"): st.session_state.map_height = 600; st.rerun()
            with map_size_col3:
                if st.button("Larger"): st.session_state.map_height = min(900, st.session_state.map_height + 50); st.rerun()

        with col2:
            st.subheader("Data Summary")
            with st.container():
                st.write(f"**{column_label} Statistics:**")
                stats = data[color_column].describe().reset_index()
                stats.columns = ["Statistic", "Value"]
                stats["Value"] = stats["Value"].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
                st.dataframe(stats, use_container_width=True)

                min_val, max_val = data[color_column].min(), data[color_column].max()
                st.info(f"Range: {min_val:.4f} to {max_val:.4f}")

                if "mun" in data.columns:
                    top_region = data.loc[data[color_column].idxmax(), "mun"]
                    bottom_region = data.loc[data[color_column].idxmin(), "mun"]
                    extreme_col1, extreme_col2 = st.columns(2)
                    with extreme_col1:
                        st.metric("Highest value", f"{max_val:.4f}", delta=None); st.text(f"Region: {top_region}")
                    with extreme_col2:
                        st.metric("Lowest value", f"{min_val:.4f}", delta=None); st.text(f"Region: {bottom_region}")

            if selected_regions:
                st.subheader("Region Comparison")
                comparison_data = data[data["mun"].isin(selected_regions)]
                comparison_fig = px.bar(comparison_data, x="mun", y=color_column, title=f"Comparison of {column_label}", labels={"mun": "Municipality", color_column: column_label})
                st.plotly_chart(comparison_fig, use_container_width=True)

    with tab2:
        st.subheader("Data Exploration")
        st.write("### Distribution Analysis")

        hist_fig = px.histogram(data, x=color_column, title=f"Distribution of {column_label}", labels={color_column: column_label})
        st.plotly_chart(hist_fig, use_container_width=True)

        if st.checkbox("Show box plot"):
            box_fig = px.box(data, y=color_column, labels={color_column: column_label}, title=f"Box Plot of {column_label}")
            st.plotly_chart(box_fig, use_container_width=True)

        if st.checkbox("Show correlation with other variables"):
            correlation_vars = st.multiselect(
                "Select variables to correlate",
                options=[opt["value"] for opt in column_options if opt["value"] != color_column],
                default=[numeric_columns[0]] if numeric_columns and numeric_columns[0] != color_column else [],
                format_func=lambda x: variable_labels.get(x, x)
            )
            if correlation_vars:
                corr_data = data[[color_column] + correlation_vars].corr()[color_column].drop(color_column).reset_index()
                corr_data.columns = ["Variable", "Correlation with " + column_label]
                corr_data["Correlation with " + column_label] = corr_data["Correlation with " + column_label].apply(lambda x: f"{x:.4f}")
                corr_data["Variable Label"] = corr_data["Variable"].apply(lambda x: variable_labels.get(x, x))
                st.dataframe(corr_data[["Variable Label", "Correlation with " + column_label]], use_container_width=True)

                top_corr_var = corr_data.iloc[0]["Variable"]
                top_corr_label = variable_labels.get(top_corr_var, top_corr_var)
                scatter_fig = px.scatter(data, x=color_column, y=top_corr_var, hover_name="mun", labels={color_column: column_label, top_corr_var: top_corr_label}, title=f"Scatter Plot: {column_label} vs {top_corr_label}")
                st.plotly_chart(scatter_fig, use_container_width=True)

        if st.checkbox("Show raw data"):
            filter_mun = st.text_input("Filter by municipality name")
            filtered_display_data = data[data["mun"].str.contains(filter_mun, case=False)] if filter_mun else data
            cols_to_display = ["mun", color_column] + [col for col in numeric_columns[:5] if col != color_column]
            st.dataframe(filtered_display_data[cols_to_display], use_container_width=True)
            st.info(f"Showing {len(filtered_display_data)} of {len(data)} regions")

    with tab3:
        st.subheader("Documentation")
        st.markdown("""### About This Application\n\nThis interactive choropleth map application allows you to explore various municipal development indicators.  The data is displayed geographically.
        \n### Data Sources\n\nThe data comes from multiple sources compiled into a single geospatial dataset. It includes:\n\n- Socioeconomic indicators\n- Development indices\n- Geographic information
        \n### Using the Application\n\n1. **Select variables**: Use the dropdown or search. 2. **Customize the map**. 3. **Explore the data**: View statistics. 4. **Compare regions**. 5. **Download data**.
        \n### Variable Descriptions""")

        if variable_labels:
            var_search = st.text_input("Search for variables")
            var_data = [{"Variable Code": var, "Description": label} for var, label in variable_labels.items() if var in numeric_columns and (not var_search or var_search.lower() in var.lower() or var_search.lower() in label.lower())]
            if var_data:
                st.dataframe(pd.DataFrame(var_data), use_container_width=True)
            else:
                st.info("No matching variables found")
        else:
            st.warning("Variable descriptions are not available")

    st.markdown("---")
    export_col1, export_col2 = st.columns(2)

    with export_col1:
        csv_data = data.drop(columns='geometry').to_csv(index=False)
        st.download_button(label="Download data as CSV", data=csv_data, file_name="map_data.csv", mime="text/csv", key="csv_download")

    with export_col2:
        geojson_data = json.dumps(data.__geo_interface__, indent=2)
        st.download_button(label="Download data as GeoJSON", data=geojson_data, file_name="map_data.geojson", mime="application/geo+json", key="geojson_download")

except DataLoadError as e:
    st.error(f"Failed to load data: {e}")
    st.info("Try refreshing the page or checking your internet connection.")
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.error("Please check the data source or report this issue.")
    logging.error(f"Unexpected error: {e}", exc_info=True)

st.markdown("---")