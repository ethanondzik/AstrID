from urllib.error import HTTPError
from astroquery.skyview import SkyView
from astroquery.mast import Observations
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
from astropy.visualization import astropy_mpl_style
from astropy.table import Table
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import re
import pandas as pd





"""
Possible function imports from this file:
# Import custom functions to extract our Image arrays and Pixel Mask arrays from our created fits files dataset
from dataGathering import extractImageArray, extractPixelMaskArray, extract_star_catalog
from dataGathering import getStarData, getCoordRangeFromPixels, getStarsInImage, getPixelCoordsFromStar, getImagePlot, getPixelMaskPlot
from dataGathering import displayRawImage, displayRawPixelMask, displayImagePlot, displayPixelMaskPlot, displayPixelMaskOverlayPlot

# Import custom functions to import the dataset
from dataGathering import importDataset

"""




plt.style.use(astropy_mpl_style)


def save_plot_as_image(ax, filename, pixels=512):
    """
    Save the plot as an image file with specified dimensions.

    Parameters:
    ----------
    ax : matplotlib.axes.Axes
        The Axes object containing the plot.
    filename : str
        The filename for the saved image.
    pixels : int
        The width and height of the image in pixels.

    """
    plt.close('all')
    # Temporarily switch to the Agg backend
    original_backend = plt.get_backend()
    plt.switch_backend('Agg')

    # Save the plot as an image file
    image_filename = filename.replace('.fits', '.png')
    fig = ax.figure
    fig.set_size_inches(pixels / fig.dpi, pixels / fig.dpi)
    plt.savefig(image_filename, format='png', bbox_inches='tight', pad_inches=0)

    plt.close('all')
    # Switch back to the original backend
    plt.switch_backend(original_backend)

    return image_filename


def convert_image_to_fits(image_filename, fits_filename, hdu_name):
    """
    Convert the saved image to FITS format and append it to the FITS file.

    Parameters:
    ----------
    image_filename : str
        The filename of the saved image.
    fits_filename : str
        The filename of the FITS file.
    hdu_name : str
        The name of the HDU to be added to the FITS file.
    """
    # Read the saved image file
    image_data = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
    
    # Resize the image to match the FITS dimensions if necessary
    image_data = cv2.resize(image_data, (512, 512), interpolation=cv2.INTER_AREA)
    
    # Create a new ImageHDU for the image data
    image_hdu = fits.ImageHDU(image_data, name=hdu_name)
    
    # Append the ImageHDU to the existing FITS file
    with fits.open(fits_filename, mode='update') as hdul:
        hdul.append(image_hdu)
        hdul.flush()


def get_random_coordinates(avoid_galactic_plane=True):
    if avoid_galactic_plane:
        while True:
            ra = random.uniform(0, 360)
            # dec = random.uniform(-90, 90)
            # Limit dec upper and lower bound to avoid the "galactic plane"
            dec = random.uniform(-60, 60)
            coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
            galactic_coords = coords.galactic
            if abs(galactic_coords.b.deg) > 10:  # Avoiding ±10 degrees around the galactic plane
                break
    else:
        ra = random.uniform(0, 360)
        dec = random.uniform(-90, 90)
    
    return ra, dec



def clean_dec_value(dec_value):
    """
    Clean up the declination value by removing extraneous spaces.

    Parameters:
    ----------
    dec_value : str
        The declination value to be cleaned.

    Returns:
    -------
    str
        The cleaned declination value.
    """

    # Regular expression to keep only valid characters
    valid_chars = re.compile(r'[^0-9+\-dms.]')
    return valid_chars.sub('', dec_value)


def getStarData(catalog_type='II/246', iterations=1, filename='data'):

    # Create a new directory to store the
    if not os.path.exists('data'):
        os.makedirs('data')


    for i in range(iterations):

        filename_str = filename + str(i)
        file_path = 'data/fits/' + filename_str + '.fits'
        attempts = 0

        while attempts < 100:
            try:
                # ra = random.uniform(0, 360)
                # dec = random.uniform(-90, 90)
                ra, dec = get_random_coordinates()
                coords = SkyCoord(ra, dec, unit='deg', frame='icrs')

                # coords = SkyCoord(ra=172.63903944*u.deg, dec=48.98346557*u.deg, frame='icrs')


                print('SkyView')        #DEBUG


                # Fetch image data from SkyView
                image_list = SkyView.get_images(position=coords, survey=['DSS'], radius=0.25 * u.deg, pixels=512)

                # Extract the image data from the list
                image_hdu = image_list[0][0]
                image = image_list[0][0].data

                # Extract WCS information from image
                wcs = WCS(image_hdu.header)


                print('Vizier')        #DEBUG


                # Fetch star data from Vizier using the 2MASS catalog
                v = Vizier(columns=['*'])
                v.ROW_LIMIT = -1
                catalog_list = v.query_region(coords, radius=0.35 * u.deg, catalog=catalog_type)
                catalog = catalog_list[0]


                print('Save')        #DEBUG


                # Save the image as a FITS file
                image_hdu = fits.PrimaryHDU(image, header=image_hdu.header)
                hdul = fits.HDUList([image_hdu])
                hdul.writeto(file_path, overwrite=True)



                print('Save Catalog')        #DEBUG

                # Save the star catalog
                with fits.open(file_path, mode='update') as hdul:
                    # Sanitize the header if necessary
                    sanitized_catalog = Table(catalog, meta=sanitize_header(catalog.meta))
                    
                    # Create a binary table HDU for the star catalog
                    star_hdu = fits.BinTableHDU(sanitized_catalog, name='STAR_CATALOG')
                    
                    # Append the star catalog HDU to the FITS file
                    hdul.append(star_hdu)
                    hdul.flush()


                coord_range = getCoordRangeFromPixels(wcs)

                # Copy the catalog and convert the table to a pandas DataFrame for easier manipulation
                catalog_df = catalog.copy().to_pandas()


                stars_in_image = getStarsInImage(wcs, catalog_df, coord_range)
                # print("Stars in image: ", stars_in_image)
                print("Number of cataloged stars in image: ", len(stars_in_image))

                
                # Get the pixel coordinates of the first star in the image
                pixel_coords = getPixelCoordsFromStar(stars_in_image[1], wcs)


                # return
                break

            except HTTPError as e:
                if e.code == 404:
                    print(f"HTTP Error 404: Not Found. Generating new coordinates and retrying...)")
                    attempts += 1
                else:
                    raise e  # Re-raise the exception if it's not a 404 error
            except Exception as e:
                print(f"An error occurred: {e}. Generating new coordinates and retrying...)")
                attempts += 1
        # raise RuntimeError(f"Failed to fetch and save data after {attempts} attempts.")




        x_dim = wcs.pixel_shape[0] # May need to swap x and y dim! (but I think it's right...)
        y_dim = wcs.pixel_shape[1]

        # Pixel-mask of stars
        pixel_mask = np.zeros((x_dim, y_dim))


        print('Drawing')        #DEBUG


        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection=wcs)
        for star in stars_in_image: 

            pixel_coords = getPixelCoordsFromStar(star, wcs)
            # pixel_mask[int(np.round(pixel_coords[0]))][int(np.round(pixel_coords[1]))] = 1

            # Ensure the pixel coordinates are within bounds
            x, y = int(np.round(pixel_coords[0])), int(np.round(pixel_coords[1]))
            if 0 <= x < x_dim and 0 <= y < y_dim:
                pixel_mask[x][y] = 1

            # print('PIXEL COORDS: ', pixel_coords)

            Drawing_colored_circle = plt.Circle(( pixel_coords[0] , pixel_coords[1] ), 0.1, fill=False, edgecolor='Blue')
            ax.add_artist( Drawing_colored_circle )
            ax.set_title(f'{filename}.fits')
            ax.set_xlabel('RA')
            ax.set_ylabel('Dec')
            ax.grid(color='white', ls='dotted')

        # Save the plot as an image file
        image_filename = save_plot_as_image(ax, file_path)
        
        # Convert the saved image to FITS format and append it to the FITS file
        convert_image_to_fits(image_filename, file_path, 'star_overlay')
        


        # Save the pixel mask to the FITS file
        with fits.open(file_path, mode='update') as hdul:
            hdu = fits.ImageHDU(pixel_mask, name='pixel_mask')
            hdul.append(hdu)
            hdul.flush()



# Ensure the header keywords conform to FITS standards
def sanitize_header(header):
    sanitized_header = {}
    for key, value in header.items():
        if len(key) > 8:
            key = key[:8]  # Truncate to 8 characters
        sanitized_header[key] = value
    return sanitized_header


# Function that takes a wcs object and returns an array of the range of ICRS coordinates in the image
def getCoordRangeFromPixels(wcs):

    x_dim = wcs.pixel_shape[0] # May need to swap x and y dim! (but I think it's right...)
    y_dim = wcs.pixel_shape[1]

    coord_range = {}

    coord_range['lower_left'] = wcs.all_pix2world([0], [0], 1)
    coord_range['lower_right'] = wcs.all_pix2world([x_dim], [0], 1)
    coord_range['upper_left'] = wcs.all_pix2world([0], [y_dim], 1)
    coord_range['upper_right'] = wcs.all_pix2world([x_dim], [y_dim], 1)
    
    return coord_range



# Get all the stars in the image
def getStarsInImage(wcs, catalog_df, coord_range):

    # NOTE: X Max and min are reversed for some reason.. orientation of image in coord system...?


    x_max = coord_range['lower_left'][0]
    x_min = coord_range['lower_right'][0]

    y_min = coord_range['lower_left'][1]
    y_max = coord_range['upper_left'][1]

    stars_in_image = []

    print("Number of stars in catalog query: ", len(catalog_df))
    
    for star in catalog_df.iterrows(): 

        # rej = star[1][0]
        # dej = star[1][1]    
        
        # NOTE : Above was causing warning:
        # FutureWarning: Series.getitem treating keys as positions is deprecated. In a future version, 
        # integer keys will always be treated as labels (consistent with DataFrame behavior). 
        # To access a value by position, use ser.iloc[pos] rej = star[1][0] 
        
        rej = star[1].iloc[0]
        dej = star[1].iloc[1]

        if rej < x_max and rej > x_min: 

            # print('Star is in x-coords')

            if dej < y_max and dej > y_min: 

                # Then star is within bounds of image! Add it to a list of stars in the image
                # print('Star is in y-coords')

                stars_in_image.append(star)


    return stars_in_image



# Get a star from the catalog and convert is coords to pixel coords
def getPixelCoordsFromStar(star, wcs):

    star_coords = star[1]['_2MASS']

    def parseStarCoords(coords):

        if '-' in coords:

            rej, dej = coords.split('-')
            rej = rej[0:2] + 'h' + rej[2:4] + 'm' + rej[4:6] + '.' + rej[6:] + 's'
            dej = '-' + dej[0:2] + 'd' + dej[2:4] + 'm' + dej[4:6] + '.' + dej[6:] + 's'

        elif '+' in coords:

            rej, dej = coords.split('+')
            rej = rej[0:2] + 'h' + rej[2:4] + 'm' + rej[4:6] + '.' + rej[6:] + 's'
            dej = '+' + dej[0:2] + 'd' + dej[2:4] + 'm' + dej[4:6] + '.' + dej[6:] + 's'

        # print('COORDS:', rej + ' ' + dej)

        dej = clean_dec_value(dej)  # Clean the declination value

        return rej + dej
    


    # coords = parseStarCoords(star_coords)

    # c = SkyCoord(coords, frame=ICRS)



    # NOTE: The above code was not working when an incorrect value (an 'A' or a 'B') came through from a star_coords:
    # ValueError: Cannot parse first argument data "- 46d40m13.7As" for attribute dec

    # I added a function to clean the declination value inside the parseStarCoords function and a try block to catch the ValueError
    coords = parseStarCoords(star_coords)

    try:
        c = SkyCoord(coords, frame=ICRS)
    except ValueError as e:
        print(f"Error parsing coordinates: {coords}")
        raise e

    pixel_coords = wcs.world_to_pixel(c)
    # print('Pixel Coords:', pixel_coords)
    return pixel_coords


def extract_star_catalog(file_path):
    """
    Extract the star catalog from the FITS file.

    Parameters:
    ----------
    file_path : str
        The path to the FITS file.

    Returns:
    -------
    catalog : Table
        The star catalog as an astropy Table.
    """
    with fits.open(file_path) as hdul:
        # Locate the STAR_CATALOG HDU
        star_hdu = hdul['STAR_CATALOG']
        
        # Read the star catalog into an astropy Table
        catalog = Table(star_hdu.data)
    
    return catalog



def displayRawImage(file_path):
    """
    Display the raw image data from a FITS file.

    Parameters:
    ----------
    filename : str
        The path to the FITS file.
    """
    with fits.open(file_path) as hdul:
        image_data = hdul[0].data

    plt.figure(figsize=(10, 10))
    plt.imshow(image_data, cmap='gray', origin='lower')
    plt.title('Raw Image Data')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.grid(False)
    plt.show()

def displayRawPixelMask(file_path):
    """
    Display the raw image data from a FITS file.

    Parameters:
    ----------
    filename : str
        The path to the FITS file.
    """
    with fits.open(file_path) as hdul:
            image_hdu = hdul[0]
            wcs = WCS(image_hdu.header)

            pixel_mask = hdul['pixel_mask'].data

    plt.figure(figsize=(10, 10))
    plt.imshow(pixel_mask, cmap='gray', origin='lower')
    plt.title('Raw Image Data')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.grid(False)
    plt.show()



# Display the image with coords overlaid on top
def displayImagePlot(file_path):

    with fits.open(file_path) as hdul:
        image_hdu = hdul[0]
        wcs = WCS(image_hdu.header)

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection=wcs)

        ax.imshow(image_hdu.data, cmap='gray', origin='lower')
        ax.set_title(file_path)
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        ax.grid(color='white', ls='dotted')

        plt.show()




# Get the image
def getImagePlot(file_path):

    with fits.open(file_path) as hdul:
        image_hdu = hdul[0]
        wcs = WCS(image_hdu.header)

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection=wcs)

        ax.imshow(image_hdu.data, cmap='gray', origin='lower')
        ax.set_title(file_path)
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        ax.grid(color='white', ls='dotted')

        return fig, ax
    

# Get the image
def extractImageArray(file_path):

    with fits.open(file_path) as hdul:
        image_hdu = hdul[0]

        image = image_hdu.data

        return image
    

# Display the pixel mask
def displayPixelMaskPlot(file_path):

    with fits.open(file_path) as hdul:
        image_hdu = hdul[0]
        wcs = WCS(image_hdu.header)

        pixel_mask = hdul['pixel_mask'].data

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection=wcs)

        ax.imshow(pixel_mask, cmap='gray', origin='lower')
        ax.set_title(file_path)
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        ax.grid(color='white', ls='dotted')

        plt.show()


# Get the pixel mask
def getPixelMaskPlot(file_path):
    

    with fits.open(file_path) as hdul:
        image_hdu = hdul[0]
        wcs = WCS(image_hdu.header)

        pixel_mask = hdul['pixel_mask'].data

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection=wcs)

        ax.imshow(pixel_mask, cmap='gray', origin='lower')
        ax.set_title(file_path)
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        ax.grid(color='white', ls='dotted')

        return fig, ax
    

# Get the pixel mask
def extractPixelMaskArray(file_path):
    

    with fits.open(file_path) as hdul:
        image_hdu = hdul[0]
        wcs = WCS(image_hdu.header)

        pixel_mask = hdul['pixel_mask'].data

        return pixel_mask
    

# Display the image with the star overlay
def displayPixelMaskOverlayPlot(file_path, catalog='II/246'):
    """
    Display the image with the star overlay.

    Parameters:
    ----------
    file_path : str
        The path to the FITS file.
    catalog : Table
        The star catalog.
    """

    with fits.open(file_path) as hdul:
        image_hdu = hdul[0]
        wcs = WCS(image_hdu.header)

        coord_range = getCoordRangeFromPixels(wcs)

        catalog = extract_star_catalog(file_path)

        # Convert the table to a pandas DataFrame for easier manipulation
        catalog_df = catalog.to_pandas()

        stars_in_image = getStarsInImage(wcs, catalog_df, coord_range)
        print("Number of cataloged stars in image: ", len(stars_in_image))

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection=wcs)

        x_dim = wcs.pixel_shape[0]
        y_dim = wcs.pixel_shape[1]

        # Pixel-mask of stars
        pixel_mask = np.zeros((x_dim, y_dim))

        print('Drawing')  # DEBUG

        for star in stars_in_image:
            pixel_coords = getPixelCoordsFromStar(star, wcs)
            # pixel_mask[int(np.round(pixel_coords[0]))][int(np.round(pixel_coords[1]))] = 1
            # Ensure the pixel coordinates are within bounds
            x, y = int(np.round(pixel_coords[0])), int(np.round(pixel_coords[1]))
            if 0 <= x < x_dim and 0 <= y < y_dim:
                pixel_mask[x][y] = 1

            Drawing_colored_circle = plt.Circle((pixel_coords[0], pixel_coords[1]), 2, fill=False, edgecolor='Blue')
            ax.add_artist(Drawing_colored_circle)

        ax.set_title(f'{file_path}')
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        ax.grid(color='white', ls='dotted')

        ax.imshow(image_hdu.data, cmap='gray', origin='lower')
        plt.show()

    # Example usage
    # displayPixelMaskImage('data/star0.fits', catalog)



# Get the image with the star overlay
def getPixelMaskOverlayPlot(file_path, catalog='II/246'):
    """
    Display the image with the star overlay.

    Parameters:
    ----------
    filename : str
        The path to the FITS file.
    catalog : Table
        The star catalog.
    """

    with fits.open(file_path) as hdul:
        image_hdu = hdul[0]
        wcs = WCS(image_hdu.header)

        coord_range = getCoordRangeFromPixels(wcs)

        catalog = extract_star_catalog(file_path)

        # Convert the table to a pandas DataFrame for easier manipulation
        catalog_df = catalog.to_pandas()

        stars_in_image = getStarsInImage(wcs, catalog_df, coord_range)
        print("Number of cataloged stars in image: ", len(stars_in_image))

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection=wcs)

        x_dim = wcs.pixel_shape[0]
        y_dim = wcs.pixel_shape[1]

        # Pixel-mask of stars
        pixel_mask = np.zeros((x_dim, y_dim))

        print('Drawing')  # DEBUG

        for star in stars_in_image:
            pixel_coords = getPixelCoordsFromStar(star, wcs)
            # pixel_mask[int(np.round(pixel_coords[0]))][int(np.round(pixel_coords[1]))] = 1
            # Ensure the pixel coordinates are within bounds
            x, y = int(np.round(pixel_coords[0])), int(np.round(pixel_coords[1]))
            if 0 <= x < x_dim and 0 <= y < y_dim:
                pixel_mask[x][y] = 1

            Drawing_colored_circle = plt.Circle((pixel_coords[0], pixel_coords[1]), 2, fill=False, edgecolor='Blue')
            ax.add_artist(Drawing_colored_circle)

        ax.set_title(f'{file_path}')
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        ax.grid(color='white', ls='dotted')

        ax.imshow(image_hdu.data, cmap='gray', origin='lower')

        return fig, ax, stars_in_image, wcs

    # Example usage
    # fig, ax = getPixelMaskImage('star0.fits', catalog)
    # plt.show()




def saveFitsImages(filename, file_path, catalog_type='II/246'):

    plt.style.use(astropy_mpl_style)

    if file_path is None:
        file_path = 'data/fits/' + filename
    # else:
        # file_path = file_path + filename


    with fits.open(file_path + filename) as hdul:
        image_hdu = hdul[0]
        wcs = WCS(hdul[0].header)

        fig, axs = plt.subplots(1, 3, figsize=(21, 7), subplot_kw={'projection': wcs})

        # Get the first image (Original Image)
        axs[0].imshow(image_hdu.data, cmap='gray', origin='lower')
        axs[0].set_title('Original Image')
        axs[0].set_xlabel('RA')
        axs[0].set_ylabel('Dec')
        axs[0].grid(color='white', ls='dotted')


        # Get the second image (Pixel Mask)
        if 'pixel_mask' in hdul:
            pixel_mask_hdu = hdul['pixel_mask']
            axs[1].imshow(pixel_mask_hdu.data, cmap='gray', origin='lower')
        else:
            axs[1].text(0.5, 0.5, 'No Pixel Mask', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
        axs[1].set_title('Pixel Mask')
        axs[1].set_xlabel('RA')
        axs[1].set_ylabel('Dec')
        axs[1].grid(color='white', ls='dotted')


        # Get the third image (Pixel Mask Overlay)
        coord_range = getCoordRangeFromPixels(wcs)
        catalog = extract_star_catalog(file_path + filename)
        catalog_df = catalog.to_pandas()
        stars_in_image = getStarsInImage(wcs, catalog_df, coord_range)
        print("Number of cataloged stars in image: ", len(stars_in_image))

        axs[2].imshow(image_hdu.data, cmap='gray', origin='lower')
        for star in stars_in_image:
            pixel_coords = getPixelCoordsFromStar(star, wcs)
            x, y = int(np.round(pixel_coords[0])), int(np.round(pixel_coords[1]))
            if 0 <= x < wcs.pixel_shape[0] and 0 <= y < wcs.pixel_shape[1]:
                Drawing_colored_circle = plt.Circle((pixel_coords[0], pixel_coords[1]), 2, fill=False, edgecolor='Blue')
                axs[2].add_artist(Drawing_colored_circle)
        axs[2].set_title('Pixel Mask Overlay')
        axs[2].set_xlabel('RA')
        axs[2].set_ylabel('Dec')
        axs[2].grid(color='white', ls='dotted')

        # Save the combined image
        plt.tight_layout()
        image_filename = filename.replace('.fits', '.png')
        image_path = file_path + image_filename
        plt.savefig(image_path)
        plt.show()

    # Example usage
    # saveFitsImages('data1.fits')


def importDataset(dataset_path = 'data/fits/', dataset_name = 'data'):
    """
    Import the dataset from the specified folder and extract the image and mask arrays.

    Parameters:
    ----------
    dataset_path : str
        The path to the folder containing the dataset.
    dataset_name : str
        The name of the dataset.('data' for testing, or 'validate' for validation/predictions)

    Returns:
    -------
    images : list
        A list of the image arrays.
    masks : list
        A list of the mask arrays.
    star_data : list
        A list of the star data.
    fits_files : list
        A list of the FITS files in the dataset folder.
    """

    # Create images and masks arrays lists
    images = []
    masks = []

    # Create a list of all the wcs data in the dataset folder
    wcs_data = []

    # Create df to store the star data inside each fits file
    stars_in_image = []

    # Create a list of all the fits files in the dataset folder
    fits_files = os.listdir(dataset_path)

    # For all the fits files in the dataset folder specified in file_path, extract the image and mask arrays to the respective lists
    file_path = dataset_path
    # for file in os.listdir(file_path):
    for file in os.listdir(file_path):
        if file.endswith('.png'):
            os.remove(file_path + file)
        if file.startswith(dataset_name) and file.endswith('.fits'):
            images.append(extractImageArray(file_path + file))
            masks.append(extractPixelMaskArray(file_path + file))
            wcs = wcs_data.append(WCS(fits.open(file_path + file)[0].header))
            stars_in_image.append(getStarsInImage(wcs, extract_star_catalog(file_path + file).to_pandas(), getCoordRangeFromPixels(WCS(fits.open(file_path + file)[0].header))))


            print(file + ' added to dataset')

    return images, masks, stars_in_image, wcs_data, fits_files