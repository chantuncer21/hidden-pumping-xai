# Unfolding the Hidden Role of Pumping in Nonlinear Hydroclimatic Interactions Using a Serial Hybrid Numerical and AI Modeling Framework

This repository contains the data and code supporting the manuscript:

**Unfolding the Hidden Role of Pumping in Nonlinear Hydroclimatic Interactions Using a Serial Hybrid Numerical and AI Modeling Framework**

**Authors:** Cihan Tuncer¹, Hakan Başağaoğlu², Chetan Sharma³, Debaditya Chakraborty⁴

1. Istanbul Water and Sewerage Administration, Istanbul, Türkiye
2. Edwards Aquifer Authority, San Antonio, TX, USA
3. Jodhpur Institute of Engineering and Technology, Jodhpur, Rajasthan, India
4. University of Texas at San Antonio, San Antonio, TX, USA

**Corresponding author:** Cihan Tuncer (cituncer@iski.gov.tr)

**Status:** Submitted to *Journal of Hydrology* [DOI:/Citation:]

## Overview

The study uses a serial hybrid framework in which a physics-based hydrological model (AquiMod 2) generates groundwater level (GWL) and runoff time series under five progressively intensified pumping scenarios. Tree-based ensemble models (Random Forest, Extremely Randomized Trees, XGBoost) are then trained to predict monthly GWL with an expanding-window validation strategy, and SHAP is used to explain the predictions. Pumping is excluded from the AI predictors to test whether lagged GWL can implicitly capture its effects.

## Requirements

### AquiMod 2 (required)

The numerical simulations require **AquiMod 2** to be installed. It is not included in this repository. Please download it from the British Geological Survey:

https://www.bgs.ac.uk/technologies/software/aquimod/#download

AquiMod 2 is distributed under its own license terms. Please follow the BGS instructions for installation and the AquiMod 2 User Manual for configuration. The model parameters used in this study are listed in Table 1 of the manuscript.

### Other software

- Python [3.12.3] with the packages listed in `requirements.txt` [e.g., numpy, pandas, scikit-learn, xgboost, shap, matplotlib]
- CLIGEN stochastic weather generator [if the climate generation step is rerun]

## Repository structure

├── data/ # Input for climate (CLIGEN model outputs)
├── graphs/ # Graphical results of rf_model.py 
├── lumped_model/ # AquiMod 2 model configuration file
├── rf_model.py/ # Scripts for Random Forest Model: AI modeling, validation and SHAP analysis
├── xgboost_model.py/ # Scripts for XGBoost Model: AI modeling, validation and SHAP analysis
├── ert_model.py/ # Scripts for Extremely Randomized Trees Model: AI modeling, validation and SHAP analysis
├── requirements.txt
└── README.md