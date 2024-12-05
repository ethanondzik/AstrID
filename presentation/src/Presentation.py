import os
import re
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback, no_update
import json
import numpy as np
import pandas as pd
from astropy.io import fits


#----------------FITS functions-----------------------

def load_fits_image(fits_file):

    hdul = fits.open(fits_file)
    image_data = hdul[0].data  # Image is in the primary HDU
    hdul.close()
    return image_data

def load_fits_prediction(fits_file):

    hdul = fits.open(fits_file)
    prediction_data = hdul[0].data  # Image is in the primary HDU
    hdul.close()
    return prediction_data

def normalize_image(image_data):

    normalized_image = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 255
    return normalized_image.astype(np.uint8)

def create_pixel_mask(image_data, threshold_factor=4): #4 makes it less crowded

    threshold = np.mean(image_data) + threshold_factor * np.std(image_data)
    pixel_mask = (image_data > threshold).astype(int)
    return pixel_mask

def pixel_mask_to_dataframe(pixel_mask):

    rows, cols = np.where(pixel_mask == 1)  # Get the white pixel coords
    return pd.DataFrame({"x": cols, "y": rows, "Category": "Pixel Mask"})


#----------------image functions-----------------------

def create_image_plot(normalized_image):

    fig = px.imshow(normalized_image, color_continuous_scale="gray")
    fig.update_layout(dragmode="drawrect", title="FITS Image")
    return fig

def add_pixel_mask_to_plot(fig, df):

    fig.add_trace(go.Scatter(
        x=df["x"], y=df["y"],
        mode="markers",
        marker=dict(
            size=4,
            symbol="star",
            color="DarkSlateBlue",
            opacity=0.5
        ),
        name="Pixel Mask"
    ))

#----------------Dash app functions-----------------------

def create_dash_app(fig, available_fits_files):

    app = Dash(__name__)

    app.layout = html.Div(
        [
            html.H3("Choose a FITS file"),
            dcc.Dropdown(
                id="fits-file-dropdown",
                options=[{"label": file, "value": file} for file in available_fits_files],
                value=available_fits_files[0],  # Defaults to the first file
                style={"width": "50%"}
            ),
            html.H3("Drag and draw rectangle annotations"),
            dcc.Graph(id="graph-picture", figure=fig),
            dcc.Markdown("Characteristics of shapes"),
            html.Pre(id="annotations-data"),
        ]
    )

    # The callback is what updates the image and pixel mask when a new fits file is selected
    @callback(
        Output("graph-picture", "figure"),
        Input("fits-file-dropdown", "value"),
        prevent_initial_call=True,
    )
    def update_image(selected_file):
        
        # Load/process the selected FITS file
        fits_file = os.path.join('data', 'fits', selected_file)
        image_data = load_fits_image(fits_file)
        normalized_image = normalize_image(image_data)
        pixel_mask = create_pixel_mask(image_data)
        df = pixel_mask_to_dataframe(pixel_mask)

        # Create/update the plot
        fig = create_image_plot(normalized_image)
        add_pixel_mask_to_plot(fig, df)
        return fig

    @callback(
        Output("annotations-data", "children"),
        Input("graph-picture", "relayoutData"),
        prevent_initial_call=True,
    )
    def on_new_annotation(relayout_data): #inherited this from dash app API, not sure if keeping it

        if "shapes" in relayout_data:
            return json.dumps(relayout_data["shapes"], indent=2)
        else:
            return no_update

    return app

#----------------Main-----------------------

def main():

    #---------------------- Load the data file-------------------------------------
    
    # List FITS files in the 'data/fits' directory
    fits_dir = os.path.join('data', 'fits')
    available_fits_files = [file for file in os.listdir(fits_dir) if file.endswith(".fits") and file.startswith("data")]

    # Sort files numerically based on the number in the filename
    available_fits_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

    #---------------------- Load the correspoinding validate file-------------------------------------
    validate_dir = os.path.join('data', 'fits')
    available_validate_files = [file for file in os.listdir(validate_dir) if file.endswith(".fits") and file.startswith("validate")]


    # Process the data FITS file
    fits_file = os.path.join(fits_dir, available_fits_files[0])
    image_data = load_fits_image(fits_file)
    normalized_image = normalize_image(image_data)
    data_pixel_mask = create_pixel_mask(image_data)
    df = pixel_mask_to_dataframe(data_pixel_mask)

    # Process the validate fits file
    validate_file =  os.path.join(validate_dir, available_fits_files[0])
    validate_data = load_fits_prediction(validate_file)
    # validate_pixel_mask = 
    #------------ NEED TO FIGURE OUT HOW THE PREDICTION IS STORED---------

    # Image plot and add pixel mask
    fig = create_image_plot(normalized_image)
    add_pixel_mask_to_plot(fig, df)

    # Run the Dash app
    app = create_dash_app(fig, available_fits_files)
    app.run(debug=True)

if __name__ == "__main__":
    main()
