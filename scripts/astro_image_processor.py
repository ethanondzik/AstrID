import sys
import os
import random
from urllib.error import HTTPError
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astroquery.skyview import SkyView
from astroquery.vizier import Vizier
import astropy.units as u
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
import cv2

class AstroImageProcessor:
    """
    A class to process and visualize astronomical images from various surveys.

    This class provides methods to fetch images from specified surveys, save and load images to/from FITS files, 
    process images using OpenCV, and display both unprocessed and processed images.

    Attributes:
    ----------
    coordinates : SkyCoord
        The coordinates of the target region in the sky.
    surveys : list of str
        The list of survey names to fetch images from.
    radius : Quantity
        The radius around the coordinates to fetch images, in degrees.
    images : dict
        A dictionary to store the fetched images.
    headers : dict
        A dictionary to store the headers of the fetched images.
    fits : HDUList
        The FITS file object.
    star_data : Table
        The star data table.

    Methods:
    -------
    fetch_images():
        Fetches images from the specified surveys and stores them in the images dictionary.
    save_to_fits(filename):
        Saves the fetched images to a FITS file.
    load_from_fits(filename):
        Loads images from a FITS file into the images dictionary.
    display_fits_info(filename):
        Displays information about the FITS file.
    display_images():
        Displays the fetched images in a 2x2 grid.
    display_unprocessed_and_processed_images():
        Processes the images with OpenCV and displays both unprocessed and processed images.
    display_header_info():
        Displays the header information of the fetched images.
    process_with_opencv():
        Processes the images using OpenCV's edge detection and returns the processed images.
    """
    DATA_DIR = '../data/raw/'

    def __init__(self, coordinates=None, surveys=None, radius=0.15, fits_filename='output.fits'):
        if coordinates is None:
            coordinates = self.generate_random_coordinates()
        self.coordinates = coordinates
        self.surveys = surveys if surveys else ['DSS']
        self.radius = radius * u.deg if isinstance(radius, (int, float)) else radius
        self.images = {}
        self.headers = {}
        self.star_data = None

        # Fetch images, retrieve star data, and save to FITS file
        self.fetch_and_save_data_pipeline()

    def get_available_surveys(self) -> list:
        """
        Returns a list of available surveys from SkyView.

        Returns:
        -------
        available_surveys : list
            A list of available surveys from SkyView.
        """
        available_surveys = SkyView.list_surveys()
        return available_surveys

    def generate_random_coordinates(self):
        """
        Generate random coordinates within the range of available surveys from SkyView.

        Returns:
        -------
        coordinates : SkyCoord
            The generated random coordinates.
        """
        ra = random.uniform(0, 360)
        dec = random.uniform(-90, 90)
        coordinates = SkyCoord(ra, dec, unit='deg', frame='icrs')
        return coordinates

    def fetch_images(self):
        """
        Fetches images from the specified surveys and stores them in the images dictionary.
        """
        for survey in self.surveys:
            while True:
                try:
                    image_list = SkyView.get_images(position=self.coordinates, survey=[survey], radius=self.radius)
                    self.images[survey] = image_list[0][0].data
                    self.headers[survey] = image_list[0][0].header
                    break  # Exit the loop if the image is fetched successfully
                except HTTPError as e:
                    if e.code == 404:
                        print(f"HTTP Error 404: Not Found for survey {survey} at coordinates {self.coordinates}. Generating new coordinates...")
                        self.coordinates = self.generate_random_coordinates()
                    else:
                        raise e  # Re-raise the exception if it's not a 404 error


    def retrieve_images_from_vizier(self, catalog_name, column_name, image_survey):
        """
        Retrieves images from the Vizier catalog based on the coordinates and stores them in the images dictionary.

        Parameters:
        ----------
        catalog_name : str
            The name of the Vizier catalog to retrieve data from.
        column_name : str
            The column name containing the image name in the Vizier catalog.
        image_survey : str
            The name of the survey to fetch images from.
        """
        vizier = Vizier(columns=[column_name])
        result = vizier.query_region(self.coordinates, radius=self.radius, catalog=catalog_name)
        image_names = result[0][column_name]
        for image_name in image_names:
            image_list = SkyView.get_images(position=self.coordinates, survey=[image_survey], radius=self.radius)
            self.images[image_name] = image_list[0][0].data
            self.headers[image_name] = image_list[0][0].header

    def retrieve_star_data(self, catalog='I/239/hip_main'):
        """
        Retrieve star data for the given coordinates using Vizier.
    
        Returns:
        -------
        star_data : Table
            A table containing the star data.
        """
        vizier = Vizier(columns=['*', '+_r'])
        
        try:
            result = vizier.query_region(self.coordinates, radius=self.radius, catalog=catalog)
            if len(result) > 0:
                self.star_data = result[0]
                return self.star_data
            else:
                print(f"No star data found for coordinates {self.coordinates}.")
                return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    
    def get_coordinates(self):
        """
        Get the current coordinates.

        Returns:
        -------
        coordinates : SkyCoord
            The current coordinates.
        """
        return self.coordinates

    def fetch_and_save_data_pipeline(self, filename, max_attempts=10):
        """
        Combines the functionality of fetching images, retrieving star data, and saving to FITS file.
        Retries the process with new coordinates if an HTTP 404 error occurs.

        Parameters:
        ----------
        filename : str
            The name of the FITS file to save the data to.
        max_attempts : int
            The maximum number of attempts to retry the process with new coordinates.
        """
        attempts = 0
        while attempts < max_attempts:
            try:
                # Generate random coordinates
                self.coordinates = self.generate_random_coordinates()
                
                # Fetch images
                self.fetch_images()
                
                # Retrieve star data
                star_data = self.retrieve_star_data()
                
                # Save images and star data to FITS file
                self.save_to_fits(filename)
                self.append_star_data_to_fits(filename)
                
                # Exit the loop if successful
                return
            except HTTPError as e:
                if e.code == 404:
                    print(f"HTTP Error 404: Not Found. Generating new coordinates and retrying... (Attempt {attempts + 1}/{max_attempts})")
                    attempts += 1
                else:
                    raise e  # Re-raise the exception if it's not a 404 error
            except Exception as e:
                print(f"An error occurred: {e}. Generating new coordinates and retrying... (Attempt {attempts + 1}/{max_attempts})")
                attempts += 1
        
        raise RuntimeError(f"Failed to fetch and save data after {max_attempts} attempts.")


    def append_star_data_to_fits(self, filename):
        """
        Append or replace star data in the FITS file as a new extension.

        Parameters:
        ----------
        filename : str
            The name of the FITS file to append the star data to.
        """
        if not isinstance(filename, str):
            raise TypeError(f"filename should be a string representing the filename, but got {type(filename).__name__}.")
        
        # Check if the file exists
        if not os.path.exists(filename):
            # Create a new FITS file with a primary HDU
            primary_hdu = fits.PrimaryHDU()
            hdul = fits.HDUList([primary_hdu])
            hdul.writeto(filename)
        
        with fits.open(filename, mode='update') as hdul:
            # Check if 'STAR_DATA' extension already exists
            if 'STAR_DATA' in hdul:
                # Find the index of the existing 'STAR_DATA' extension
                star_data_index = [i for i, hdu in enumerate(hdul) if hdu.name == 'STAR_DATA'][0]
                # Replace the existing 'STAR_DATA' extension
                hdul[star_data_index] = fits.BinTableHDU(self.star_data, name='STAR_DATA')
            else:
                # Append the new 'STAR_DATA' extension
                star_hdu = fits.BinTableHDU(self.star_data, name='STAR_DATA')
                hdul.append(star_hdu)
            hdul.flush()
    
    def get_star_data_table_from_fits(self, filename):
        """
        Get star data from the FITS file.
    
        Parameters:
        ----------
        filename : str
            The name of the FITS file to get the star data from.
        """
        with fits.open(filename) as hdul:
            self.star_data = hdul['STAR_DATA'].data

        # Convert the star data to an astropy Table
        star_data_table = Table(self.star_data)

        return star_data_table
    
    def print_star_data_table_from_fits(self, filename):
        """
        Print the star data from the FITS file in a formatted table.
        """
        # Get star data from the FITS file.
        with fits.open(filename) as hdul:
            self.star_data = hdul['STAR_DATA'].data

        # Convert the star data to an astropy Table
        star_data_table = Table(self.star_data)

        # Print the table
        print(star_data_table)

    def get_star_data_from_fits(self, filename):
        """
        Get star data from the FITS file.

        Parameters:
        ----------
        filename : str
            The name of the FITS file to get the star data from.
        """
        with fits.open(filename) as hdul:
            if 'STAR_DATA' in hdul:
                self.star_data = hdul['STAR_DATA'].data
            else:
                raise KeyError("Extension 'STAR_DATA' not found.")


    def save_to_fits(self, filename):
        """
        Save the fetched images to a FITS file.
        """
        hdus = [fits.PrimaryHDU()]
        for survey, image in self.images.items():
            hdu = fits.ImageHDU(image, name=survey)
            hdus.append(hdu)
        
        hdul = fits.HDUList(hdus)
        hdul.writeto(filename, overwrite=True)

    def load_fits_as_object(self):
        """
        Load the FITS file as an object.
        """
        self.fits = fits.open('../data/raw/' + filename)

    def display_fits_info(self, filename=None):
        """
        Displays information about the FITS file.

        Parameters:
        ----------
        filename : str
            The name of the FITS file to display information about.
        """
        with fits.open(filename) as hdul:
            hdul.info()

    def display_images(self):
        """
        Displays the fetched images in a grid layout based on the number of images.
        """
        num_images = len(self.images)
        
        if num_images == 1:
            fig, axs = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={'projection': WCS(self.headers[self.surveys[0]])})
            axs = [axs]  # Make axs iterable
        elif num_images == 2:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': WCS(self.headers[self.surveys[0]])})
        elif num_images == 3:
            fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': WCS(self.headers[self.surveys[0]])})
        else:
            fig, axs = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={'projection': WCS(self.headers[self.surveys[0]])})
        
        axs = axs.flatten() if num_images > 1 else axs

        for i, survey in enumerate(self.surveys):
            if survey in self.images:
                ax = axs[i]
                ax.imshow(self.images[survey], cmap='gray', origin='lower')
                ax.set_title(f'{survey} Image')
                ax.coords.grid(color='white', ls='dotted')
                ax.set_xlabel('RA', fontsize=8)
                ax.set_ylabel('Dec', fontsize=8)
        
        plt.tight_layout()
        plt.show()

    def display_unprocessed_and_processed_images(self, images=None, processed_images=None):
        """
        Processes the images with OpenCV and displays both unprocessed and processed images.
        """
        if images is None:
            images = self.images
        if processed_images is None:
            processed_images = self.process_with_opencv()

        # Display unprocessed and processed images for the first two surveys
        fig1, axs1 = plt.subplots(2, 2, figsize=(14, 14), subplot_kw={'projection': WCS(self.headers[self.surveys[0]])})
        for i, survey in enumerate(self.surveys[:2]):
            # Display unprocessed image
            ax = axs1[0, i]
            ax.imshow(self.images[survey], cmap='gray', origin='lower')
            ax.set_title(f'{survey} Unprocessed Image')
            ax.set_xlabel('RA', fontsize=8)
            ax.set_ylabel('Dec', fontsize=8)
            ax.grid(True, linestyle=':', color='white')  # Enable white dotted grid lines

            # Display processed image
            ax = axs1[1, i]
            ax.imshow(processed_images[survey], cmap='gray', origin='lower')
            ax.set_title(f'{survey} Processed Image')
            ax.set_xlabel('RA', fontsize=8)
            ax.set_ylabel('Dec', fontsize=8)
            ax.grid(True, linestyle=':', color='white')  # Enable white dotted grid lines

        plt.tight_layout()
        plt.show()

        # Display unprocessed and processed images for the next two surveys
        fig2, axs2 = plt.subplots(2, 2, figsize=(14, 14), subplot_kw={'projection': WCS(self.headers[self.surveys[2]])})
        for i, survey in enumerate(self.surveys[2:]):
            # Display unprocessed image
            ax = axs2[0, i]
            ax.imshow(self.images[survey], cmap='gray', origin='lower')
            ax.set_title(f'{survey} Unprocessed Image')
            ax.set_xlabel('RA', fontsize=8)
            ax.set_ylabel('Dec', fontsize=8)
            ax.grid(True, linestyle=':', color='white')  # Enable white dotted grid lines

            # Display processed image
            ax = axs2[1, i]
            ax.imshow(processed_images[survey], cmap='gray', origin='lower')
            ax.set_title(f'{survey} Processed Image')
            ax.set_xlabel('RA', fontsize=8)
            ax.set_ylabel('Dec', fontsize=8)
            ax.grid(True, linestyle=':', color='white')  # Enable white dotted grid lines

        plt.tight_layout()
        plt.show()

    def display_header_info(self):
        """
        Displays the header information of the fetched images.
        """
        for survey, header in self.headers.items():
            print(f"Header information for {survey}:")
            for card in header.cards:
                print(f"{card.keyword} = {card.value} / {card.comment}")
            print("\n")

    def process_with_opencv(self, blur_ksize=(5, 5), block_size=11, C=2, canny_thresh1=100, canny_thresh2=200, iterations=1) -> dict:
        """
        Processes the images using OpenCV to improve the recognition of stars on a black background of the sky.

        Parameters:
        ----------
        blur_ksize : tuple of int
            Kernel size for Gaussian blur. Must be positive and odd.
        block_size : int
            Size of the pixel neighborhood used to calculate the threshold value. Must be odd.
        C : int
            Constant subtracted from the mean or weighted mean in adaptive thresholding.
        canny_thresh1 : int
            First threshold for the hysteresis procedure in Canny edge detection.
        canny_thresh2 : int
            Second threshold for the hysteresis procedure in Canny edge detection.

        Returns:
        -------
        processed_images : dict
            A dictionary containing the processed images.
        """
        processed_images = {}
        for survey, image in self.images.items():
            # Ensure the image is properly scaled before converting to uint8
            image_scaled = (image - np.min(image)) / (np.max(image) - np.min(image))
            image_uint8 = (image_scaled * 255).astype(np.uint8)
            
            # Apply Gaussian Blur to reduce noise
            blurred_image = cv2.GaussianBlur(image_uint8, blur_ksize, 0)
            
            # Apply adaptive thresholding
            adaptive_thresh = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY, block_size, C)
            
            # Enhance contrast using histogram equalization
            equalized_image = cv2.equalizeHist(adaptive_thresh)
            
            # Apply morphological operations to enhance star features
            kernel = np.ones((3, 3), np.uint8)
            dilated_image = cv2.dilate(equalized_image, kernel, iterations=iterations)
            
            # Perform edge detection
            processed_image = cv2.Canny(dilated_image, canny_thresh1, canny_thresh2)
            
            # Update header information
            self.headers[survey]['PROCESS'] = 'Enhanced Edge Detection'
            
            # Store processed image
            processed_images[survey] = processed_image
        

            # # Visualize intermediate steps
            # fig, axs = plt.subplots(1, 6, figsize=(24, 4))
            # # Show the original floating-point image
            # axs[0].imshow(image, cmap='gray')
            # axs[0].set_title('Original Float Image')
            # # Show the image after conversion to uint8
            # axs[1].imshow(image_uint8, cmap='gray')
            # axs[1].set_title('Converted to uint8')
            # # Show the blurred image
            # axs[2].imshow(blurred_image, cmap='gray')
            # axs[2].set_title('Blurred Image')
            # # Show the adaptive threshold image
            # axs[3].imshow(adaptive_thresh, cmap='gray')
            # axs[3].set_title('Adaptive Threshold')
            # # Show the equalized image
            # axs[4].imshow(equalized_image, cmap='gray')
            # axs[4].set_title('Equalized Image')
            # # Show the final processed image
            # axs[5].imshow(processed_image, cmap='gray')
            # axs[5].set_title('Processed Image')


            plt.show()
        
        return processed_images


# Example usage
if __name__ == "__main__":
    # Generate initial random coordinates
    coordinates = SkyCoord("18h18m48s -13d49m00s", frame='icrs')
    surveys = ['DSS', 'DSS1 Blue', 'DSS2 Red', 'WISE 3.4']
    processor = AstroImageProcessor(coordinates, surveys, radius=0.15)

    # Fetch images with error handling and retry logic
    processor.fetch_images()
    processor.save_to_fits('random_coordinates_images.fits')
    processor.add_star_data_to_fits('random_coordinates_images.fits')
    processor.display_images()

    # Load from FITS and process with OpenCV
    processor.load_from_fits('random_coordinates_images.fits')
    processor.display_unprocessed_and_processed_images()








            for star in stars_in_image: 

            pixel_coords = getPixelCoordsFromStar(star, wcs)
            pixel_mask[int(np.floor(pixel_coords[0]))][int(np.floor(pixel_coords[1]))] = 1

            # print('PIXEL COORDS: ', pixel_coords)

            Drawing_colored_circle = plt.Circle(( pixel_coords[0] , pixel_coords[1] ), 0.1, fill=False, edgecolor='Blue')
            ax.add_artist( Drawing_colored_circle )
            ax.set_title(f'{filename}.fits')
            ax.set_xlabel('RA')
            ax.set_ylabel('Dec')
            ax.grid(color='white', ls='dotted')

        ax.imshow(image_hdu.data, cmap='gray', origin='lower')
        plt.show()