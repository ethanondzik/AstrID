import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
import cv2

class AstroImageProcessor:
    def __init__(self, coordinates, surveys):
        self.coordinates = coordinates
        self.surveys = surveys
        self.images = {}
        self.headers = {}

    def fetch_images(self):
        for survey in self.surveys:
            image_list = SkyView.get_images(position=self.coordinates, survey=[survey])
            self.images[survey] = image_list[0][0].data
            self.headers[survey] = image_list[0][0].header

    def save_to_fits(self, filename):
        primary_hdu = fits.PrimaryHDU()
        hdus = [primary_hdu]
        for survey in self.surveys:
            hdu = fits.ImageHDU(self.images[survey], header=self.headers[survey], name=survey)
            hdus.append(hdu)
        hdul = fits.HDUList(hdus)
        hdul.writeto(filename, overwrite=True)

    def load_from_fits(self, filename):
        hdul = fits.open(filename)
        for hdu in hdul[1:]:
            self.images[hdu.name] = hdu.data
            self.headers[hdu.name] = hdu.header
        hdul.close()

    def display_images(self):
        fig, axs = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={'projection': WCS(self.headers[self.surveys[0]])})
        for i, survey in enumerate(self.surveys):
            ax = axs[i // 2, i % 2]
            ax.imshow(self.images[survey], cmap='gray', origin='lower')
            ax.set_title(f'{survey} Image')
            ax.coords.grid(color='white', ls='dotted')
            ax.set_xlabel('RA', fontsize=8)
            ax.set_ylabel('Dec', fontsize=8)
        plt.show()

    def display_header_info(self):
        for survey in self.surveys:
            print(f"Header information for {survey}:")
            print(self.headers[survey])
            print("\n")

    def process_with_opencv(self):
        # Example OpenCV processing: edge detection
        processed_images = {}
        for survey, image in self.images.items():
            processed_image = cv2.Canny((image * 255).astype(np.uint8), 100, 200)
            processed_images[survey] = processed_image
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
    processed_images = processor.process_with_opencv()

    # Display processed images
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    for i, survey in enumerate(surveys):
        ax = axs[i // 2, i % 2]
        ax.imshow(processed_images[survey], cmap='gray', origin='lower')
        ax.set_title(f'{survey} Processed Image')
        ax.set_xlabel('RA', fontsize=8)
        ax.set_ylabel('Dec', fontsize=8)
    plt.show()