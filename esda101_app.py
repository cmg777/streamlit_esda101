# 🔹 IMPORT SECTION - Libraries needed for this application
# This section imports all necessary libraries and modules.
# 🔸 Streamlit for web app creation.
# 🔸 Geopandas for handling geospatial data.
# 🔸 Plotly Express for interactive plotting.
# 🔸 Pandas for data manipulation.
# 🔸 JSON, OS, and logging for data handling and debugging.
# 🔸 Typing for type hints.
import streamlit as st
import geopandas as gpd
import plotly.express as px
import pandas as pd
import json
import os
import logging
from typing import Dict

# 🔹 LOGGING CONFIGURATION
# This section sets up logging to track application events.
# 🔸 Logs include timestamps, log levels, and messages.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 🔹 CUSTOM EXCEPTION CLASSES
# These classes provide specific exceptions for better error handling.
# 🔸 AppError: Base exception for the app.
# 🔸 DataLoadError: Raised when there are issues loading data.
class AppError(Exception):
    """Base exception class for application errors."""
    pass

class DataLoadError(AppError):
    """Exception raised for errors in data loading."""
    pass

# 🔹 PAGE CONFIGURATION
# This configures the Streamlit page settings.
# 🔸 Sets page title, layout, and initial sidebar state.
st.set_page_config(page_title="Exploratory Data Analysis of Regional Data", layout="wide", initial_sidebar_state="expanded")

# 🔹 INITIALIZE SESSION STATE
# This section initializes session variables to manage app state.
# 🔸 map_height: Controls the height of the displayed map.
# 🔸 selected_regions: Stores user-selected regions for comparison.
# 🔸 show_tutorial: Determines if tutorial info should be shown.
if "map_height" not in st.session_state:
    st.session_state.map_height = 600
if "selected_regions" not in st.session_state:
    st.session_state.selected_regions = []
if "show_tutorial" not in st.session_state:
    st.session_state.show_tutorial = True

# 🔹 APPLICATION TITLE AND DESCRIPTION
# Sets the main title and provides a markdown description of the app.
st.title("Regional Development Indicators of Bolivia: A Dashboard for Exploratory Analysis")
st.markdown("""
This application displays a choropleth map for several indicators of regional development.
""")

# 🔹 SIDEBAR SETUP
# Configures the sidebar with controls and headers.
st.sidebar.header("Map Controls")

# 🔹 DATA LOADING FUNCTIONS
# These functions handle the loading and caching of geographic and dictionary data.

# 📌 load_data:
#    - Loads geographic data from a local file if available.
#    - Otherwise, retrieves from a remote source, saves locally, and converts CRS.
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

# 📌 load_data_dictionary:
#    - Loads a CSV file with variable definitions.
#    - Converts it to a dictionary mapping variable names to labels.
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

# 🔹 SIDEBAR ADDITIONAL LINK
# Adds a link to open the computational notebook.
st.sidebar.markdown("""---""")
st.sidebar.markdown(
    '<a href="https://colab.research.google.com/drive/1JHf8wPxSxBdKKhXaKQZUzhEpVznKGiep?usp=sharing" target="_blank">'
    '<strong>Open Computational Notebook</strong></a>',
    unsafe_allow_html=True
)

# 🔹 MAIN APPLICATION BLOCK
# This section encapsulates the primary logic of the application.
try:
    # 🔸 Load geographic data and variable definitions.
    data = load_data()
    variable_labels = load_data_dictionary()

    # 🔸 Display dataset information in the sidebar.
    st.sidebar.subheader("Dataset Information")
    st.sidebar.info(f"Number of regions: {len(data)}")

    # 🔸 Tutorial toggle control.
    show_tutorial = st.sidebar.checkbox("Show tutorial", value=st.session_state.show_tutorial)
    st.session_state.show_tutorial = show_tutorial

    if show_tutorial:
        st.sidebar.info("""**How to use:** Select a variable, adjust map style, explore statistics, compare regions, download data.""")

    # 🔸 Identify numeric columns and create options with variable labels.
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    column_options = [{"label": variable_labels.get(col, col), "value": col} for col in numeric_columns]
    column_options.sort(key=lambda x: x["label"])

    # 🔸 USER CONTROLS: Search and select variables.
    search_term = st.sidebar.text_input("Search variables", "")
    filtered_options = [opt for opt in column_options if search_term.lower() in opt["label"].lower() or search_term.lower() in opt["value"].lower()] if search_term else column_options

    if not filtered_options and search_term:
        st.sidebar.warning(f"No variables matching '{search_term}'")
        filtered_options = column_options  # Show all if no match

    # 🔸 Set default variable selection (e.g., "imds").
    default_index = next((i for i, option in enumerate(filtered_options) if option["value"] == "imds"), 0)

    color_column = st.sidebar.selectbox(
        "Select data to visualize",
        options=[option["value"] for option in filtered_options],
        index=default_index,
        format_func=lambda x: variable_labels.get(x, x),
        help="Choose data to display. Hover over each municipality to see its value."
    )

    # 🔸 Map style selection.
    map_styles = {"Carto Dark": "carto-darkmatter", "Carto Light": "carto-positron",  "OpenStreetMap": "open-street-map", "White Background": "white-bg"}
    style_name = st.sidebar.selectbox("Map style", options=list(map_styles.keys()), index=0, help="Select background map design")
    map_style = map_styles[style_name]

    # 🔸 Color palette selection.
    color_scales = ["RdBu_r", "viridis", "plasma", "inferno", "magma", "cividis", "twilight", "Blues", "Greens", "Reds", "YlOrRd"]
    color_scale = st.sidebar.selectbox("Color palette", options=color_scales, index=0, help="Choose the color scheme")

    # 🔸 Configure color scale center using median or a custom slider.
    use_median = st.sidebar.checkbox("Center color scale at median", value=True)
    midpoint = data[color_column].median() if use_median else st.sidebar.slider("Color scale midpoint", float(data[color_column].min()), float(data[color_column].max()), float(data[color_column].median()))

    # 🔸 Additional map controls for opacity and zoom.
    opacity = st.sidebar.slider("Map opacity", min_value=0.0, max_value=1.0, value=0.9, step=0.1, help="Adjust transparency")
    zoom = st.sidebar.slider("Zoom level", min_value=3.0, max_value=10.0, value=4.5, step=0.5, help="Adjust zoom")

    # 🔸 Sidebar control for comparing regions.
    st.sidebar.subheader("Compare Regions")
    selected_regions = st.sidebar.multiselect(
        "Select regions to compare",
        options=sorted(data["mun"].unique()),
        default=st.session_state.selected_regions,
        help="Choose municipalities to compare"
    )
    st.session_state.selected_regions = selected_regions

    # 🔹 Define application tabs for organizing content.
    tab1, tab2, tab3 = st.tabs(["Map Visualization", "Exploratory Data Analysis (EDA)", "Documentation"])

    # 🔹 TAB 1: Map Visualization
    with tab1:
        # Split the layout into two columns: map display and data summary.
        col1, col2 = st.columns([2, 1])

        with col1:
            # Prepare geojson data for mapping.
            geojson_dict = data.__geo_interface__
            column_label = variable_labels.get(color_column, color_column)
            hover_cols = [color_column] + [col for col in numeric_columns[:3] if col != color_column]
            labels = {col: variable_labels.get(col, col) for col in hover_cols + ["mun"]}

            # Mark selected regions if any.
            if selected_regions:
                data['is_selected'] = data['mun'].isin(selected_regions)
                hover_cols.append('is_selected')
                labels['is_selected'] = 'Selected Region'

            # Create the choropleth map using Plotly.
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

            # Update layout parameters for the map.
            fig.update_layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                height=st.session_state.map_height,
                autosize=True,
                coloraxis_colorbar=dict(title=column_label, title_side="right")
            )
            st.plotly_chart(fig, use_container_width=True)

            # Provide buttons to adjust map height.
            map_size_col1, map_size_col2, map_size_col3 = st.columns(3)
            with map_size_col1:
                if st.button("Smaller"):
                    st.session_state.map_height = max(400, st.session_state.map_height - 50)
                    st.rerun()
            with map_size_col2:
                if st.button("Reset Size"):
                    st.session_state.map_height = 600
                    st.rerun()
            with map_size_col3:
                if st.button("Larger"):
                    st.session_state.map_height = min(900, st.session_state.map_height + 50)
                    st.rerun()

        with col2:
            # 🔸 Display data summary and statistics.
            st.subheader("Data Summary")
            with st.container():
                st.write(f"**{column_label} Statistics:**")
                stats = data[color_column].describe().reset_index()
                stats.columns = ["Statistic", "Value"]
                stats["Value"] = stats["Value"].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
                st.dataframe(stats, use_container_width=True)

                # Display the range of values.
                min_val, max_val = data[color_column].min(), data[color_column].max()
                st.info(f"Range: {min_val:.2f} to {max_val:.2f}")

                # Identify regions with highest and lowest values.
                if "mun" in data.columns:
                    top_region = data.loc[data[color_column].idxmax(), "mun"]
                    bottom_region = data.loc[data[color_column].idxmin(), "mun"]
                    extreme_col1, extreme_col2 = st.columns(2)
                    with extreme_col1:
                        st.metric("Highest value", f"{max_val:.2f}", delta=None)
                        st.text(f"Region: {top_region}")
                    with extreme_col2:
                        st.metric("Lowest value", f"{min_val:.2f}", delta=None)
                        st.text(f"Region: {bottom_region}")

            # 🔸 Region comparison: Bar chart for selected regions.
            if selected_regions:
                st.subheader("Region Comparison")
                comparison_data = data[data["mun"].isin(selected_regions)]
                comparison_fig = px.bar(comparison_data, x="mun", y=color_column, title=f"Comparison of {column_label}", labels={"mun": "Municipality", color_column: column_label})
                st.plotly_chart(comparison_fig, use_container_width=True)

    # 🔹 TAB 2: Data Analysis
    with tab2:
        st.subheader("Data Exploration")
        st.write("### Distribution Analysis")

        # 🔸 Histogram to show data distribution.
        hist_fig = px.histogram(data, x=color_column, title=f"Distribution of {column_label}", labels={color_column: column_label})
        st.plotly_chart(hist_fig, use_container_width=True)

        # 🔸 Box plot
        box_fig = px.box(data, y=color_column, labels={color_column: column_label}, title=f"Box Plot of {column_label}")
        st.plotly_chart(box_fig, use_container_width=False)

        # 🔸 Correlation analysis 
        # Here, the y-axis variable is assumed to be "imds" (selected via color_column) 
        # and the default x-axis variable is "ln_t400NTLpc2012" if available.
        default_corr = ["ln_t400NTLpc2012"] if "ln_t400NTLpc2012" in [opt["value"] for opt in column_options if opt["value"] != color_column] else []

        correlation_vars = st.multiselect(
            "Select variables to correlate",
            options=[opt["value"] for opt in column_options if opt["value"] != color_column],
            default=default_corr,
            format_func=lambda x: variable_labels.get(x, x)
        )

        if correlation_vars:
            # 📌 Compute correlations between the primary variable (assumed "imds") and the selected variables.
            corr_series = data[[color_column] + correlation_vars].corr()[color_column].drop(color_column)
            corr_df = corr_series.reset_index().rename(columns={"index": "Variable", color_column: "Correlation"})
            
            # 📌 Calculate absolute correlations and sort by strength.
            corr_df["Absolute Correlation"] = corr_df["Correlation"].abs()
            corr_df = corr_df.sort_values("Absolute Correlation", ascending=False)
            
            # 📌 Format the correlation values for display.
            corr_label = f"Correlation with {column_label}"
            corr_df[corr_label] = corr_df["Correlation"].apply(lambda x: f"{x:.4f}")
            corr_df["Variable Label"] = corr_df["Variable"].apply(lambda x: variable_labels.get(x, x))
            
            # 📌 Display the correlation table.
            st.dataframe(corr_df[["Variable Label", corr_label]], use_container_width=True)
            
            # 📌 Determine the x-axis variable: prefer "ln_t400NTLpc2012" if selected.
            if "ln_t400NTLpc2012" in correlation_vars:
                x_var = "ln_t400NTLpc2012"
            else:
                x_var = corr_df.iloc[0]["Variable"]
            x_label = variable_labels.get(x_var, x_var)
            
            # 📌 Create a scatter plot with a regression line (OLS) using Plotly.
            scatter_fig = px.scatter(
                data,
                x=x_var,
                y=color_column,
                hover_name="mun",
                labels={x_var: x_label, color_column: column_label},
                title=f"Scatter Plot: {column_label} vs {x_label}"
            )
            st.plotly_chart(scatter_fig, use_container_width=True)


        # 🔸 Option to display raw data with filtering.
        if st.checkbox("Show raw data"):
            filter_mun = st.text_input("Filter by municipality name")
            filtered_display_data = data[data["mun"].str.contains(filter_mun, case=False)] if filter_mun else data
            cols_to_display = ["mun", color_column] + [col for col in numeric_columns[:5] if col != color_column]
            st.dataframe(filtered_display_data[cols_to_display], use_container_width=True)
            st.info(f"Showing {len(filtered_display_data)} of {len(data)} regions")

    # 🔹 TAB 3: Documentation
    with tab3:
        st.subheader("🧭 Documentation")
        st.markdown("""


This interactive application enables users to explore municipal development indicators across Bolivia. It offers:

- 🗺️ Data visualization on a map  
- 📈 Distribution and comparative analysis tools  
- 💾 Downloadable datasets  
- 🧮 Access to a cloud-based computational notebook on [Google Colab](https://colab.research.google.com/drive/1JHf8wPxSxBdKKhXaKQZUzhEpVznKGiep?usp=sharing)  

> ⚠️ This application is open source and a work in progress. Source code available at: [github.com/cmg777/streamlit_esda101](https://github.com/cmg777/streamlit_esda101)

---

## 🚀 Using the Application

1. 📝 **Select variables** via dropdown or search  
2. 🎨 **Customize the map** by Sustainable Development Goal or indicator  
3. 📊 **Explore the data**: view distributions and statistics  
4. 🔍 **Compare municipalities** across indicators  
5. 💾 **Download** data for offline use  

---

## 📚 Sources and Credits

- Primary data source: [Municipal Atlas of the SDGs in Bolivia 2020](https://sdsnbolivia.org/Atlas/)  
- Web app and computational notebook by [Carlos Mendez](https://carlos-mendez.org)

**Citation**:  
Mendez, C. (2025, March 24). *Regional Development Indicators of Bolivia: A Dashboard for Exploratory Analysis* (Version 0.0.2) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.15074864  

---

## 🌐 Context and Motivation

Adopted in 2015, the **2030 Agenda for Sustainable Development** established 17 Sustainable Development Goals. While global metrics offer useful benchmarks, they often overlook subnational disparities—particularly in heterogeneous countries like Bolivia.

- 🇧🇴 Bolivia ranks **79/166** on the 2020 SDG Index (score: 69.3)  
- 🏘️ The *Atlas* reveals **intra-national disparities** comparable to **global inter-country variation**  

---

## 📊 Composite Index: Índice Municipal de Desarrollo Sostenible (IMDS)

The **Municipal Sustainable Development Index (IMDS)** summarizes municipal performance using 62 indicators across 15 Sustainable Development Goals. Systmatic and reliable information on goals 12 and 14 are not available at municipal level.

### 🎯 Methodological Criteria

- ✅ Relevance to local Sustainable Development Goal targets  
- 📥 Data availability from official or trusted sources  
- 🌐 Full municipal coverage (339 municipalities)  
- 🕒 Data mostly from 2012–2019  
- 🧮 Low redundancy between indicators  

---

## 🗃️ Indicators by Sustainable Development Goal

### 🧱 Goal 1: No Poverty

- Energy poverty rate (2012, INE)  
- Multidimensional Poverty Index (2013, UDAPE)  
- Unmet Basic Needs (2012, INE)  
- Access to basic services: water, sanitation, electricity (2012, INE)

### 🌾 Goal 2: Zero Hunger

- Chronic malnutrition in children under five (2016, Ministry of Health)  
- Obesity prevalence in women (2016, Ministry of Health)  
- Average agricultural unit size (2013, Agricultural Census)  
- Tractor density per 1,000 farms (2013, Agricultural Census)

### 🏥 Goal 3: Good Health and Well-being

- Infant and under-five mortality rates (2016, Ministry of Health)  
- Institutional birth coverage (2016, Ministry of Health)  
- Incidence of Chagas, HIV, malaria, tuberculosis, dengue (2016, Ministry of Health)  
- Adolescent fertility rate (2016, Ministry of Health)

### 📚 Goal 4: Quality Education

- Secondary school dropout rates, by gender (2016, Ministry of Education)  
- Adult literacy rate (2012, INE)  
- Share of population with higher education (2012, INE)  
- Share of qualified teachers, initial and secondary levels (2016, Ministry of Education)

### ⚖️ Goal 5: Gender Equality

- Gender parity in education, labor participation, and poverty (2012–2016, INE and UDAPE)  
- *Note: Data on gender-based violence not available at municipal level*

### 💧 Goal 6: Clean Water and Sanitation

- Access to potable water (2012, INE)  
- Access to sanitation services (2012, INE)  
- Proportion of treated wastewater (2015, Ministry of Environment)

### ⚡ Goal 7: Affordable and Clean Energy

- Electricity coverage (2012, INE)  
- Per capita electricity consumption (2015, Ministry of Energy)  
- Use of clean cooking energy (2015, Ministry of Hydrocarbons)  
- CO₂ emissions per capita, energy-related (2015, international satellite data)

### 💼 Goal 8: Decent Work and Economic Growth

- Share of non-functioning electricity meters (proxy for informality/unemployment) (2015, Ministry of Energy)  
- Labor force participation rate (2012, INE)  
- Youth not in education, employment, or training (NEET rate) (2015, Ministry of Labor)

### 🏗️ Goal 9: Industry, Innovation, and Infrastructure

- Internet access in households (2012, INE)  
- Mobile signal coverage (2015, telecommunications data)  
- Availability of urban infrastructure (2015, Ministry of Public Works)

### ⚖️ Goal 10: Reduced Inequality

- Proxy measures: municipal differences in poverty and participation rates (2012–2016, INE and UDAPE)

### 🏘️ Goal 11: Sustainable Cities and Communities

- Urban housing adequacy (2012, INE)  
- Access to collective transportation (2015, Ministry of Transport)

### 🌍 Goal 13: Climate Action

- Natural disaster resilience index (2015, Ministry of Environment)  
- CO₂ emissions and forest degradation (2015, satellite data)

### 🌳 Goal 15: Life on Land

- Deforestation rates (2015, satellite data)  
- Biodiversity loss indicators (2015, Ministry of Environment)

### 🕊️ Goal 16: Peace, Justice, and Strong Institutions

- Birth registration coverage (2012, INE)  
- Crime and homicide rates (2015, Ministry of Government)  
- Corruption perceptions (2015, civil society organizations)

### 🤝 Goal 17: Partnerships for the Goals

- Municipal fiscal capacity (2015, Ministry of Economy)  
- Public investment per capita (2015, Ministry of Economy)

---

## 🧩 Technical Highlights

- Index scores range from **<40 to >80** (0–100 scale)  
- Traffic-light visualization: green (goal met), red (critical)  
- Composite indices for benchmarking municipalities  
- Interactive map enables issue-specific and regional analysis  

---

## 💡 Applications

- 🏛️ **Governance**: Local planning, policy design, budgeting  
- 🎓 **Research**: Open-access data for academic analysis (Excel, Stata, SDSS formats)  
- 🧭 **Civil society**: Local development monitoring and accountability  
- 💼 **Private sector**: Investment targeting based on local conditions  

---

## ⚠️ Limitations and Future Work

- No disaggregated data for Indigenous Territories (TIOC)  
- Many indicators based on 2012 Census; updates pending  
- Limited information for Goals 12 and 14 at municipal level  
- No indicators for educational quality (due to lack of standardized testing)  
- Gender violence data unavailable at municipal scale  

---

## 🔗 Access 

- **Original website**: [atlas.sdsnbolivia.org](http://atlas.sdsnbolivia.org)  
- **Original Publication**: [sdsnbolivia.org/Atlas](http://www.sdsnbolivia.org/Atlas)  
- **Source Code of the Web App**: [github.com/cmg777/streamlit_esda101](https://github.com/cmg777/streamlit_esda101)  
- **Computational Notebook**: [Google Colab](https://colab.research.google.com/drive/1JHf8wPxSxBdKKhXaKQZUzhEpVznKGiep?usp=sharing)
          

### Variable Descriptions""")

        # 🔸 Display variable descriptions if available.
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
        st.markdown(
            '<a href="https://colab.research.google.com/drive/1JHf8wPxSxBdKKhXaKQZUzhEpVznKGiep?usp=sharing" target="_blank">'
            '<strong>Open Computational Notebook</strong></a>',
            unsafe_allow_html=True
        )

    # 🔹 DATA EXPORT SECTION
    st.markdown("---")
    export_col1, export_col2 = st.columns(2)

    with export_col1:
        # 🔸 Provide CSV download option (geometry column is dropped).
        csv_data = data.drop(columns='geometry').to_csv(index=False)
        st.download_button(label="Download data as CSV", data=csv_data, file_name="map_data.csv", mime="text/csv", key="csv_download")

    with export_col2:
        # 🔸 Provide GeoJSON download option.
        geojson_data = json.dumps(data.__geo_interface__, indent=2)
        st.download_button(label="Download data as GeoJSON", data=geojson_data, file_name="map_data.geojson", mime="application/geo+json", key="geojson_download")

# 🔹 ERROR HANDLING
# Catch specific and general exceptions during the application run.
except DataLoadError as e:
    st.error(f"Failed to load data: {e}")
    st.info("Try refreshing the page or checking your internet connection.")
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.error("Please check the data source or report this issue.")
    logging.error(f"Unexpected error: {e}", exc_info=True)

# 🔹 FINAL SEPARATOR
st.markdown("---")
