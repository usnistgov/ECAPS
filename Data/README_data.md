# Data in support of “Infrared Spectroscopy to Classify Polyolefins using Machine Learning”

## Background

Bradley P. Sutliff, Peter A. Beaucage, Debra J. Audus, Sara V. Orski, and Tyler B. Martin, "Sorting Polyolefins with Near-visible Infrared Spectroscopy: Identification of optimal data analysis pipelines and machine learning classifiers"  *Digital Discovery* **2024** *NUM* (NUM), PAGES DOI:

This submitted publication explores using machine learning to enhance the characterization potential of near-visible infrared spectroscopy when applied to polyolefin materials. It describes a methodology for correlating a NIR spectra with physical properties such as crystallization, density, and short chain branching of polyolefin species.

Data is provided both for reproducibility and to enable prototyping of the concepts discussed in the submitted manuscript. A code repository is provided at [ECAPS](https://github.com/usnistgov/ecaps), which includes a notebook illustrating the concepts within the submitted manuscript. Reproduction of the figures from the submitted manuscript including plotting of the data contained herein is also provided.

## Data descriptions

### Sample Information
The `SampleInformation.csv` file provides the original resin code from the manufacturer (where appropriate), the source, the sample code used to identify materials in this work, the major and minor polyolefin class as reported by the source, the physical form/shape of the sample when tested, and additional notes such as BigSMILES and alternative names.

This file is also used by the `1-ECAPS_PreProcessingExamples.ipynb` to generate a list of files for the code to automatically read into the document and generate an Xarray dataset.

### NIR
The `NIR` folder contains the spectroscopic information for the NIR measurements, after background subtraction was automatically applied by the instrument. They are labeled as `<SampleCode>_<Replicate>.csv` where each sample was measured 7 times, with the sample being shaken between each replicate measurement. Samples were characterized at wavelengths between approximately 4000 cm$^{-1}$ and 10000 cm$^{-1}$, and the intensity is given in terms of % reflectance. 

More details on how these measurement were obtained can be found in the corresponding submitted paper:
Bradley P. Sutliff, Peter A. Beaucage, Debra J. Audus, Sara V. Orski, and Tyler B. Martin, "Sorting Polyolefins with Near-visible Infrared Spectroscopy: Identification of optimal data analysis pipelines and machine learning classifiers"  *Digital Discovery* **2024** *NUM* (NUM), PAGES DOI:

## Contact

Bradley P. Sutliff, PhD  
Materials Science and Engineering Division  
Material Measurement Laboratory  
National Institute of Standards and Technology  

Email: Bradley.Sutliff@nist.gov  
GithubID: @bpsut  
Staff website: https://www.nist.gov/people/bradley-sutliff  

## How to cite

If you use the code, please cite our submitted manuscript once published:

Bradley P. Sutliff, Peter A. Beaucage, Debra J. Audus, Sara V. Orski, and Tyler B. Martin, "Sorting Polyolefins with Near-visible Infrared Spectroscopy: Identification of optimal data analysis pipelines and machine learning classifiers"  *Digital Discovery* **2024** *NUM* (NUM), PAGES DOI:

If you use the data, please cite:

Sutliff, Bradley P., Goyal, Shailja, Martin, Tyler B., Beaucage, Peter A., Audus, Debra J., Orski, Sara V. (2023), Correlating Near-Infrared Spectra to Bulk Properties in Polyolefins, National Institute of Standards and Technology, https://doi.org/10.18434/mds2-3022 (Accessed DATE)
