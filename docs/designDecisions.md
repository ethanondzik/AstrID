# Design Decisions

This document outlines the key design decisions made during the development of the AstrID project.

## Project Name

- **Decision**: The project is named **AstrID**.
- **Reasoning**: The name cleverly combines "astrid" with "ID" to emphasize the identification aspect of the project. It also has a double meaning, as it can be interpreted as "Astro Identification" and a play on the word "asteroid."

## Catalog Choices

### Hipparcos Catalog (`I/239/hip_main`)

- **Reasoning**: The Hipparcos catalog provides high-precision astrometric data for over 100,000 stars, including positions, parallaxes, proper motions, and magnitudes. This makes it ideal for projects requiring precise astrometric data and stellar positions.

### 2MASS Catalog (`II/246`)

- **Reasoning**: The 2MASS catalog contains near-infrared data for millions of objects, including J, H, and K band magnitudes. It is suitable for projects focusing on infrared observations and studies of stellar populations, star formation, and galactic structure.

## Image Resolution

- **Decision**: Increase the resolution of images obtained through SkyView by specifying a higher pixel size.
- **Implementation**: Use the `pixels` parameter in the `SkyView.get_images` function to set the image size to 1024x1024 pixels, increasing the resolution.

## Plotting Improvements

- **Decision**: Improve the image representation by increasing the figure size and resolution, and overlaying the WCS grid for better identification of stars.
- **Implementation**:
  ```python
  fig = plt.figure(figsize=(10, 10), dpi=150)
  ax = fig.add_subplot(111, projection=wcs)
  ax.imshow(image, cmap='gray', origin='lower')
  ax.set_title('High Resolution Sky Image')
  ax.set_xlabel('RA')
  ax.set_ylabel('Dec')
  ax.grid(color='white', ls='dotted')
  ```

## Machine Learning Approach

- **Decision**: Use machine learning to classify different types of stars and other celestial objects.
- **Reasoning**: Machine learning techniques can help automate the classification process and improve accuracy.
- **Implementation**: Start with simple classification algorithms (e.g., decision tree, k-nearest neighbors) and use libraries like OpenCV for image processing and scikit-learn for machine learning.

## Stretch Goal: Black Hole Identification

- **Decision**: Extend the project to identify potential black hole candidates using advanced machine learning techniques.
- **Reasoning**: Identifying black holes is a challenging but valuable goal that can significantly enhance the project's impact.
- **Implementation**: Research advanced techniques and additional data sources for black hole identification, integrate new features and data into the existing model, and train and evaluate the extended model.