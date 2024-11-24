from dash import Dash, dcc, html, Input, Output, callback
import plotly.express as px
import pandas as pd
import numpy as np
import json
import os
from astropy.io import fits

# Load the FITS file and extract the pixel mask
fits_file = '../resource/data0.fits'
fits_file = os.path.join(os.path.dirname(__file__), fits_file) #calculate absolute path for fits file on the fly
hdul = fits.open(fits_file)
image_data = hdul[0].data  # Image is in the primary HDU
hdul.close()

# Create a binary mask (adjust the threshold as needed)
threshold = np.mean(image_data) + 2 * np.std(image_data)  # Example threshold
pixel_mask = (image_data > threshold).astype(int)

# Convert pixel mask to a Panda DF
rows, cols = np.where(pixel_mask == 1)  # Gets the white pixels' coords
df = pd.DataFrame({
    "x": cols,
    "y": rows,  
    "category": "Pixel Mask"  # Category for toggling
})

"""
    Need the same process as above but for the corresponding predicitons.

    - Change coordinates to match fits?
    - Overlay the image of the space under the graph?
    - Fix the x, y scale to change the look?

"""

# Init the dash app
app = Dash(__name__)

# Scatter plot
fig = px.scatter(df, x="x", y="y", color="category", title="Pixel Mask Scatter Plot")
fig.update_layout(clickmode='event+select')
fig.update_traces(marker_size=5)

# Layout
app.layout = html.Div([
    dcc.Graph(
        id='pixel-mask-scatter',
        figure=fig
    ),
    html.Div(className='row', children=[
        html.Div([
            dcc.Markdown("""
                **Hover Data**

                Mouse over points in the graph.
            """),
            html.Pre(id='hover-data', style={'border': 'thin lightgrey solid', 'overflowX': 'scroll'})
        ], className='three columns'),

        html.Div([
            dcc.Markdown("""
                **Click Data**

                Click on points in the graph.
            """),
            html.Pre(id='click-data', style={'border': 'thin lightgrey solid', 'overflowX': 'scroll'}),
        ], className='three columns'),

        html.Div([
            dcc.Markdown("""
                **Selection Data**

                Use the lasso or rectangle tool to select points in the graph.
            """),
            html.Pre(id='selected-data', style={'border': 'thin lightgrey solid', 'overflowX': 'scroll'}),
        ], className='three columns'),
    ])
])

# Callbacks - display interactions
@callback(
    Output('hover-data', 'children'),
    Input('pixel-mask-scatter', 'hoverData'))
def display_hover_data(hoverData):
    return json.dumps(hoverData, indent=2)


@callback(
    Output('click-data', 'children'),
    Input('pixel-mask-scatter', 'clickData'))
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)


@callback(
    Output('selected-data', 'children'),
    Input('pixel-mask-scatter', 'selectedData'))
def display_selected_data(selectedData):
    return json.dumps(selectedData, indent=2)


if __name__ == '__main__':
    app.run(debug=True)
