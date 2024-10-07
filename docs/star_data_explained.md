# SkyView
## Available Survey Types and Their Usage

In the context of the **AstrID** project, various survey types can be utilized to gather data for identifying and classifying astronomical objects. Below is a list of available survey types and their potential usage:

#### Allbands:GOODS/HDF/CDF
- **GOODS: HST ACS B, V, I, Z**: High-resolution optical images from the Hubble Space Telescope. Useful for detailed studies of star formation and galaxy morphology.
- **GOODS: HST NICMOS**: Near-infrared images from the Hubble Space Telescope. Ideal for observing star formation regions obscured by dust.
- **GOODS: Spitzer IRAC 3.6, 4.5, 5.8, 8.0**: Infrared images from the Spitzer Space Telescope. Useful for studying the thermal emission from stars and galaxies.
- **GOODS: Herschel 100, 160, 250, 350, 500**: Far-infrared images from the Herschel Space Observatory. Useful for studying cold dust and star formation in galaxies.

#### GammaRay
- **Fermi, EGRET, COMPTEL**: Gamma-ray surveys. Useful for studying high-energy processes such as those occurring in black holes and neutron stars.

#### HardX-ray
- **INTEGRAL, RXTE**: Hard X-ray surveys. Useful for studying high-energy phenomena and compact objects like black holes and neutron stars.

#### IR:2MASS
- **2MASS-J, H, K**: Near-infrared surveys. Useful for studying stellar populations, star formation, and galactic structure.

#### IR:AKARI
- **AKARI N60, WIDE-S, WIDE-L, N160**: Mid- to far-infrared surveys. Useful for studying star formation and the interstellar medium.

#### IR:IRAS
- **IRIS, SFD Dust Map**: Infrared surveys. Useful for studying the distribution of dust and star formation regions.

#### IR:Planck
- **Planck**: Surveys at various frequencies. Useful for studying the cosmic microwave background and large-scale structure of the universe.

#### IR:UKIDSS
- **UKIDSS-Y, J, H, K**: Near-infrared surveys. Useful for deep surveys of the sky, studying stellar populations and galactic structure.

#### IR:WISE
- **WISE 3.4, 4.6, 12, 22**: Mid-infrared surveys. Useful for studying star formation, galaxy evolution, and the interstellar medium.

#### Optical:DSS
- **DSS, DSS1, DSS2**: Optical surveys. Useful for general-purpose sky surveys and historical data comparison.

#### Optical:SDSS
- **SDSSg, i, r, u, z**: Optical surveys. Useful for detailed studies of galaxy morphology, star formation, and large-scale structure.

#### OtherOptical
- **TESS, Mellinger, SHASSA**: Various optical surveys. Useful for studying variable stars, wide-field imaging, and specific emission lines.

#### ROSATDiffuse
- **RASS Background**: X-ray surveys. Useful for studying the diffuse X-ray background and large-scale structures.

#### ROSATw/sources
- **RASS-Cnt, PSPC, HRI**: X-ray surveys. Useful for studying individual X-ray sources and their properties.

#### Radio:GHz
- **GB6, VLA FIRST, NVSS**: Radio surveys. Useful for studying radio galaxies, quasars, and the interstellar medium.

#### Radio:GLEAM
- **GLEAM**: Low-frequency radio surveys. Useful for studying the large-scale structure of the universe and radio sources.

#### Radio:MHz
- **SUMSS, WENSS, TGSS ADR1, VLSSr**: Various radio surveys. Useful for studying radio sources and the interstellar medium.

#### SoftX-ray
- **SwiftXRT, HEAO 1 A-2**: Soft X-ray surveys. Useful for studying X-ray sources and their properties.

#### SwiftUVOT
- **UVOT**: Ultraviolet surveys. Useful for studying hot stars, star formation regions, and active galactic nuclei.

#### UV
- **GALEX, ROSAT WFC, EUVE**: Ultraviolet surveys. Useful for studying hot stars, star formation regions, and the interstellar medium.

#### X-ray:SwiftBAT
- **BAT SNR**: Hard X-ray surveys. Useful for studying high-energy phenomena and compact objects like black holes and neutron stars.

These survey types provide a comprehensive set of data that can be leveraged for various aspects of the **AstrID** project, from identifying and classifying stars to studying galaxies and other celestial phenomena.

---

## Star Data Headings Explanation and Usage

This section explains the headings used in the star data:

- **HIP**: The Hipparcos catalog number, a unique identifier for stars in the Hipparcos catalog.
  - **Usage**: Cross-reference additional data, ensure consistency across datasets.

- **RAhms**: Right Ascension in hours, minutes, and seconds, used to locate stars in the sky.
  - **Usage**: Map star positions, include as a feature for spatial pattern recognition.

- **DEdms**: Declination in degrees, minutes, and seconds, used in conjunction with RA to locate stars.
  - **Usage**: Precisely locate stars, include as a feature for spatial pattern recognition.

- **Vmag**: Visual magnitude, indicating the brightness of a star as seen from Earth.
  - **Usage**: Classify stars based on brightness, include as a feature to differentiate star types.

- **B-V**: Color index, the difference in magnitude between the B (blue) and V (visual) filters, indicating the star's color and temperature.
  - **Usage**: Classify stars based on color and temperature, include as a feature for classification.

- **_RA.icrs**: Right Ascension in degrees according to the International Celestial Reference System (ICRS).
  - **Usage**: Provide precise positional data, include as a feature for accurate positional information.

- **_DE.icrs**: Declination in degrees according to the International Celestial Reference System (ICRS).
  - **Usage**: Provide precise positional data, include as a feature for accurate positional information.

These headings provide essential information for identifying and analyzing stars in astronomical research.

---

##  Commonly Used Catalogs:

1. **Hipparcos Catalog (`I/239/hip_main`):**
   - **Description**: Contains high-precision astrometric data for over 100,000 stars.
   - **Data**: Includes positions, parallaxes, proper motions, and magnitudes.
   - **Usage**: Ideal for projects requiring precise astrometric data and stellar positions.

2. **2MASS Catalog (`II/246`):**
   - **Description**: The Two Micron All Sky Survey (2MASS) catalog contains near-infrared data for millions of objects.
   - **Data**: Includes J, H, and K band magnitudes.
   - **Usage**: Suitable for projects focusing on infrared observations and studies of stellar populations, star formation, and galactic structure.

### Factors to Consider:

1. **Type of Data Needed:**
   - **Astrometric Data**: If you need precise positions, parallaxes, and proper motions, the Hipparcos catalog is a good choice.
   - **Photometric Data**: If you need magnitudes in different bands, consider catalogs like 2MASS or others that provide photometric data.

2. **Wavelength Coverage:**
   - **Optical**: Hipparcos provides data in the optical range.
   - **Infrared**: 2MASS provides data in the near-infrared range.

3. **Catalog Coverage:**
   - **All-Sky Coverage**: Some catalogs, like 2MASS, cover the entire sky, while others may focus on specific regions.

4. **Project Goals:**
   - **Star Classification**: For classifying stars based on their properties, Hipparcos and 2MASS are both useful.
   - **Black Hole Identification**: You might need a combination of catalogs to gather comprehensive data for identifying black holes.

### Conclusion:
- **Hipparcos (`I/239/hip_main`)**: Use this catalog for precise astrometric data and stellar positions.
- **2MASS (`II/246`)**: Use this catalog for near-infrared photometric data.

You can also explore other catalogs available through Vizier based on your specific needs. Combining data from multiple catalogs can provide a more comprehensive dataset for your project.


---


## Compare star data to image, confirm coordinates
Compared using various websites and star maps:

Coordinates 18 18 36, -13 48 00, located in ESA Sky map pinpoint the exact same position on that star as the gridlines in the image in box [17].

https://sky.esa.int/esasky/?target=274.65%20-13.8&hips=PanSTARRS+DR1+color+(i%2C+r%2C+g)&fov=0.1517944745194883&cooframe=J2000&sci=true&lang=en


Coords on the Eagles Nebula:

https://sky.esa.int/esasky/?target=274.7083333333333%20-13.85&hips=PanSTARRS+DR1+color+(i%2C+r%2C+g)&fov=0.1517944745194883&cooframe=J2000&sci=true&lang=en



--- 
Using the Data in Our Project
1. Data Preprocessing:

- Clean and preprocess the data to handle any missing or inconsistent values.
- Normalize or standardize the features if necessary.

2. Feature Engineering:

- Create additional features if needed, such as combining RA and Dec into a single positional feature.
- Consider the physical significance of each feature and how it might help in classification.

3. Model Training:

- Use the features to train machine learning models such as decision trees, random forests, or neural networks.
- Evaluate the models using metrics like accuracy, precision, recall, and F1-score.

4. Star Classification:

- Use the trained models to classify stars into different types based on their features.
- Analyze the results to understand the characteristics of different star types.

5. Black Hole Identification:

- Extend the project to include features that might indicate the presence of black holes, such as unusual motion or gravitational effects on nearby stars.
- Use advanced machine learning techniques to identify potential black hole candidates.