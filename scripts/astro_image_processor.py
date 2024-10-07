import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
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

    def __init__(self, coordinates, surveys, radius=0.15):
        self.coordinates = coordinates
        self.surveys = surveys
        self.radius = radius * u.deg if isinstance(radius, (int, float)) else radius
        self.images = {}
        self.headers = {}

    def fetch_images(self):
        """
        Fetches images from the specified surveys and stores them in the images dictionary.
        """
        for survey in self.surveys:
            image_list = SkyView.get_images(position=self.coordinates, survey=[survey], radius=self.radius)
            self.images[survey] = image_list[0][0].data
            self.headers[survey] = image_list[0][0].header

    def save_to_fits(self, filename):
        """
        Saves the fetched images to a FITS file.

        Parameters:
        ----------
        filename : str
            The name of the FITS file to save the images to.
        """
        primary_hdu = fits.PrimaryHDU()
        hdus = [primary_hdu]
        for survey in self.surveys:
            hdu = fits.ImageHDU(self.images[survey], header=self.headers[survey], name=survey)
            hdus.append(hdu)
        hdul = fits.HDUList(hdus)
        hdul.writeto(filename, overwrite=True)

    def load_from_fits(self, filename):
        """
        Loads images from a FITS file into the images dictionary.

        Parameters:
        ----------
        filename : str
            The name of the FITS file to load the images from.
        """
        hdul = fits.open(filename)
        for hdu in hdul[1:]:
            self.images[hdu.name] = hdu.data
            self.headers[hdu.name] = hdu.header
        hdul.close()

    def display_fits_info(self, filename):
        """
        Displays information about the FITS file.

        Parameters:
        ----------
        filename : str
            The name of the FITS file to display information about.
        """
        hdul = fits.open(filename)
        hdul.info()
        hdul.close()

    def display_images(self):
        """
        Displays the fetched images in a 2x2 grid.
        """
        fig, axs = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={'projection': WCS(self.headers[self.surveys[0]])})
        for i, survey in enumerate(self.surveys):
            ax = axs[i // 2, i % 2]
            ax.imshow(self.images[survey], cmap='gray', origin='lower')
            ax.set_title(f'{survey} Image')
            ax.coords.grid(color='white', ls='dotted')
            ax.set_xlabel('RA', fontsize=8)
            ax.set_ylabel('Dec', fontsize=8)
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
    coordinates = SkyCoord("18h18m48s -13d49m00s", frame='icrs')
    surveys = ['DSS', 'DSS1 Blue', 'DSS2 Red', 'WISE 3.4']
    processor = AstroImageProcessor(coordinates, surveys)
    processor.fetch_images()
    processor.save_to_fits('eagle_nebula_images.fits')
    processor.display_images()

    # Load from FITS and process with OpenCV
    processor.load_from_fits('eagle_nebula_images.fits')
    processor.display_unprocessed_and_processed_images()