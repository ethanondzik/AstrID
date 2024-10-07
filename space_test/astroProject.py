from astroquery.skyview import SkyView
from astroquery.vizier import Vizier
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import cv2
import numpy as np
from astropy.io import fits
import os
import pandas as pd

def fetch_image_and_stars(coords=None, object_name=None):
    if coords is None:
        # Define the coordinates for the Eagle Nebula Pillars of Creation
        coords = SkyCoord('18h18m48s -13d49m00s', frame='icrs')
        # Track the name of the object
        object_name = 'Eagle Nebula Pillars of Creation'

    # Fetch an image from SkyView
    image_list = SkyView.get_images(position=coords, survey=['DSS'], radius=0.1 * u.deg)
    image = image_list[0][0].data

    # Create a folder for the object with the name of the object with underscores instead of spaces
    object_name = object_name.replace(' ', '_')
    os.makedirs(f'images/{object_name}', exist_ok=True)

    # Save the image as a FITS file
    fits_file_path = f'images/{object_name}/{object_name}.fits'
    hdu = fits.PrimaryHDU(image)
    hdul = fits.HDUList([hdu])
    hdul.writeto(fits_file_path, overwrite=True)

    # Download the image in all available color maps to the object folder
    for cmap in plt.colormaps():
        plt.imsave(f'images/{object_name}/{object_name}_{cmap}.png', image, cmap=cmap)

    print(f"Images and FITS file saved in 'images/{object_name}'")

    # Fetch star data from Vizier
    v = Vizier(columns=['*'])
    result = v.query_region(coords, radius=0.1 * u.deg, catalog='I/239/hip_main')
    stars = result[0]

    # Save star data to a CSV file
    stars_file_path = f'images/{object_name}/{object_name}_stars.csv'
    stars.write(stars_file_path, format='csv', overwrite=True)

    # Display the image
    plt.imshow(image, cmap='gray')
    plt.title('Sky Image')
    plt.xlabel('RA')
    plt.ylabel('Dec')
    plt.show()

    # Print star data
    print(stars)

    # Preprocess the image and star data for analysis
    preprocess_data(image, stars)

def preprocess_data(image, stars):
    # Example preprocessing steps
    # Normalize the image
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Extract star positions and magnitudes
    star_positions = np.array([stars['RAJ2000'], stars['DEJ2000']]).T
    star_magnitudes = stars['Vmag']

    # Save preprocessed data for further analysis
    np.save('preprocessed_image.npy', normalized_image)
    star_data = pd.DataFrame({'RA': stars['RAJ2000'], 'Dec': stars['DEJ2000'], 'Magnitude': stars['Vmag']})
    star_data.to_csv('preprocessed_star_data.csv', index=False)

    print("Preprocessed data saved.")


if __name__ == '__main__':
    # Example usage with custom coordinates
    custom_coords = SkyCoord('05h35m17.3s -05d23m28s', frame='icrs')  # Coordinates for the Orion Nebula
    fetch_image_and_stars(coords=custom_coords, object_name='Orion Nebula')

    # Example usage with default coordinates
    fetch_image_and_stars() 

    # # Define the coordinates for the Eagle Nebula Pillars of Creation
    # coords = SkyCoord('18h18m48s -13d49m00s', frame='icrs')

    # # Save the image using OpenCV
    # cv2.imwrite('sky_image.png', image)