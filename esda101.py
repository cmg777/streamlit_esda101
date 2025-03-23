
# %% Importing necessary libraries for data analysis and visualization
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # Importing matplotlib image for image plotting
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# %% Importing libraries for spatial data and visualization
import geopandas as gpd
import folium
from folium import Figure
import contextily as cx
import libpysal
from libpysal  import weights
from libpysal.weights import Queen

# %% Exploratory Spatial Data Analysis (ESDA) tools
import mapclassify as mc
import esda
from esda.moran import Moran, Moran_Local

# %% Spatial plotting tools
import splot
from splot.esda import moran_scatterplot, plot_moran, lisa_cluster, plot_local_autocorrelation
from splot.libpysal import plot_spatial_weights
from splot.mapping import vba_choropleth

# %% Statistical modeling
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %% Suppressing warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# %% Import data
data = gpd.read_file('https://github.com/quarcs-lab/project2021o-notebook/raw/main/map_and_data.geojson')
data.head(3)

gdf = data[['mun', 'rank_imds', 'imds', 'geometry']]
gdf

#%% Visualize spatial data using the explore() method of a GeoDataFrame
gdf.explore(
    # Specify the column to visualize on the map
    column='imds',
    # Specify the attributes to display in the tooltip when hovering over map features
    tooltip=['mun', 'imds', 'rank_imds'],
    # Choose the classification scheme for data visualization
    scheme='fisherjenks',
    # Specify the number of classes for classification
    k=3,
    # Choose the colormap for data visualization
    cmap='coolwarm',
    # Specify whether to display a legend
    legend=True,
    # Choose the basemap tiles provider
    tiles='CartoDB positron',
    # Customize the style of the basemap tiles
    style_kwds=dict(color="gray", weight=0.5),
    # Customize the appearance of the legend
    legend_kwds=dict(colorbar=False)
)