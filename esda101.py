  import geopandas as gpd
  import plotly.express as px

  # Load data
  data = gpd.read_file('https://github.com/quarcs-lab/project2021o-notebook/raw/main/map_and_data.geojson')
  data = data.to_crs(epsg=4326)

  # Ensure a unique ID column
  data["id"] = data.index.astype(str)

  # Convert GeoDataFrame to GeoJSON dictionary
  geojson_dict = data.__geo_interface__

  # Plot with proper geojson and explicit color scale "viridis"
  fig = px.choropleth_mapbox(
      data_frame=data,
      geojson=geojson_dict,
      locations="id",
      color="imds",
      hover_name = 'mun',
      hover_data=['imds', 'rank_imds'],
      color_continuous_scale="viridis",
      mapbox_style="carto-positron",
      zoom=4.5,
      center={"lat": data.geometry.centroid.y.mean(), "lon": data.geometry.centroid.x.mean()},
      opacity=0.6
  )

  # Set figure dimensions to 600x600 pixels (6 in x 6 in at 100 DPI)
  fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, width=630, height=600)
  fig.show()
