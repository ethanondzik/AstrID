# SkyView
## Available Survey Types and Their Usage

In the context of the **AstrID** project, various survey types can be utilized to gather data for identifying and classifying astronomical objects. Below is a list of available survey types and their potential usage:

#### Allbands: GOODS/HDF/CDF
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

---

## Star Data Headings Explanation and Usage

This section explains the headings used in the star data inside each fits file:

- **RAJ2000**: Right Ascension in J2000 coordinates, used to pinpoint the star's position in the sky.
- **DEJ2000**: Declination in J2000 coordinates, used alongside RAJ2000 to locate the star.
- **_2MASS**: Identifier from the 2MASS survey, providing a unique reference for each star.
- **Jmag**: Magnitude in the J band, indicating the star's brightness in near-infrared light.
- **e_Jmag**: Error in J band magnitude, representing the uncertainty in the Jmag measurement.
- **Hmag**: Magnitude in the H band, another measure of the star's brightness in near-infrared light.
- **e_Hmag**: Error in H band magnitude, representing the uncertainty in the Hmag measurement.
- **Kmag**: Magnitude in the K band, indicating the star's brightness in the far-infrared spectrum.
- **e_Kmag**: Error in K band magnitude, representing the uncertainty in the Kmag measurement.
- **Qflg**: Quality flag, providing information on the reliability of the photometric measurements.
- **Rflg**: Read flag, indicating the number of times the star was observed.
- **Bflg**: Blend flag, showing whether the star's image was blended with another source.
- **Cflg**: Contamination flag, indicating potential contamination from nearby sources.
- **Xflg**: Extended source flag, identifying whether the star is part of an extended object.
- **Aflg**: Artifact flag, indicating the presence of artifacts in the star's image.

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