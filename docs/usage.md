# Usage Guide

This guide provides instructions on how to use the AstrID project to identify and classify astronomical objects.

## Fetching High-Resolution Images

To fetch high-resolution images from SkyView, use the following code snippet:

```python
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord
import astropy.units as u

# Coordinates for the SkyView request
coords = SkyCoord('18h18m48s -13d49m00s', frame='icrs')

# Fetch a higher resolution image from SkyView
image_list = SkyView.get_images(position=coords, survey=['DSS'], radius=0.1 * u.deg, pixels=[2000, 2000])
image_hdu = image_list[0][0]
image = image_hdu.data
```

## Displaying Images with WCS Projection

To display the fetched images with WCS projection and overlay a WCS grid, use the following code snippet:

```python
import matplotlib.pyplot as plt
from astropy.wcs import WCS

# Extract WCS information
wcs = WCS(image_hdu.header)

# Display the image with WCS projection
fig = plt.figure(figsize=(10, 10), dpi=150)
ax = fig.add_subplot(111, projection=wcs)
ax.imshow(image, cmap='gray', origin='lower')
ax.set_title('High Resolution Sky Image')
ax.set_xlabel('RA')
ax.set_ylabel('Dec')

# Overlay the WCS grid
ax.grid(color='white', ls='dotted')

# Show the plot
plt.show()
```

## Querying Vizier Catalogs

To query the Hipparcos or 2MASS catalogs using Vizier, use the following code snippets:

### Hipparcos Catalog

```python
from astroquery.vizier import Vizier

# Query the Hipparcos catalog
vizier = Vizier(catalog='I/239/hip_main')
result = vizier.query_region("18h18m48s -13d49m00s", radius="0d10m0s")

# Process the result
table = result[0]
print(table)
```

### 2MASS Catalog

```python
from astroquery.vizier import Vizier

# Query the 2MASS catalog
vizier = Vizier(catalog='II/246')
result = vizier.query_region("18h18m48s -13d49m00s", radius="0d10m0s")

# Process the result
table = result[0]
print(table)
```

## Running Jupyter Notebooks

To run the Jupyter notebooks, follow these steps:

1. Activate the virtual environment:
   ```sh
   source .venv/bin/activate
   ```

2. Start Jupyter Notebook:
   ```sh
   jupyter notebook
   ```

3. Open the desired notebook file (`*.ipynb`) and select the appropriate kernel.

Refer to the `design_decisions.md` file for more information on the design choices made during the development of this project.