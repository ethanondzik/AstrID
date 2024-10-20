# Requirements

## Needs to solve an existing data science/machine learning problem

### Which you must describe and justify
AstrID aims to identify and classify astronomical objects, such as stars and galaxies, using data from various space catalogs. The project leverages machine learning to enhance our understanding of celestial phenomena and potentially identify black hole candidates.

### Why is it interesting to pursue?
Understanding and classifying astronomical objects is crucial for advancing our knowledge of the universe. By automating the classification process, we can analyze vast amounts of data more efficiently, leading to new discoveries and insights. Identifying black hole candidates is particularly exciting as it can contribute to our understanding of these mysterious objects.

### What are the most salient challenges?
- **Data Quality**: Ensuring the data from various catalogs is accurate and consistent.
- **Data Integration**: Combining data from different sources and formats.
- **Model Accuracy**: Developing a model that can accurately classify different types of astronomical objects.
- **Computational Resources**: Handling the large datasets and complex computations required for training the model.

### Don't mention time or organization

## Dataset

### Is a dataset readily available?
Yes, datasets from space catalogs like Hipparcos and 2MASS are readily available.
We created a dataset using Astropy -> SkyView, saving the resulting fits files with additional elements and specific formatting desired for our machine learning.

#### Does it fit the needs of the project?
Yes, these datasets contain high-resolution images and detailed information about various astronomical objects, which are essential for training and evaluating the model.

#### How will you obtain it?
The data can be obtained through APIs and data repositories provided by the respective space catalogs.

#### Does it require preprocessing? Any clean up/missing fields?
Yes, the data requires preprocessing to handle missing fields, normalize values, and ensure consistency across different datasets.

### Will you create your own?
No, we will use existing datasets from space catalogs.

#### What are the downsides of this approach?
- **Data Limitations**: The existing datasets may have limitations in terms of coverage and resolution.
- **Data Quality**: There may be inconsistencies or errors in the data that need to be addressed.

### An ideal dataset rarely exists
#### Reflect on the pros/cons of the ideal case and the data you are using
**Pros**:
- **High-Quality Data**: The datasets from space catalogs are generally of high quality and contain detailed information.
- **Wide Coverage**: These datasets cover a wide range of astronomical objects, providing a comprehensive dataset for training the model.

**Cons**:
- **Inconsistencies**: There may be inconsistencies in the data that need to be addressed during preprocessing.
- **Missing Data**: Some fields may be missing or incomplete, requiring additional preprocessing.

## Don't focus on the implementation itself yet

### What conclusions would you like to arrive to?
- **Accurate Classification**: Develop a model that can accurately classify different types of astronomical objects.
- **Black Hole Identification**: Identify potential black hole candidates.
- **Insights**: Gain new insights into the characteristics and distribution of various astronomical objects.

### What types of ML algorithms would you use?
- **Convolutional Neural Networks (CNNs)**: For image classification and feature extraction.
- **U-Net Model**: For segmentation and classification of astronomical objects.
- **Decision Trees/K-Nearest Neighbors**: For initial classification and feature extraction.

#### I am expecting that this will change during the course of the project
Yes, the choice of algorithms may evolve as we experiment with different approaches and refine the model.

## Python-based
Yes, the project is implemented in Python.

## Hosted on the CSCI servers
The project will be hosted on the CSCI servers for accessibility and collaboration.