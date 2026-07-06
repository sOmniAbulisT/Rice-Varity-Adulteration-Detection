# Mixed Rice Adulteration Detection

[![Language: R](https://img.shields.io/badge/Language-R-blue.svg)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview
In the context of global food safety and supply chain transparency, identifying agricultural product origin and detecting adulteration has become a critical issue. This project aims to solve the "adulteration" problem in the rice supply chain by developing an end-to-end Machine Learning Pipeline. It automatically classifies the origin of rice (Taiwan vs. Vietnam) and, for samples identified as "Mixed," accurately estimates the blending ratio along with a 95% Confidence Interval.

The system utilizes a **Two-Stage Prediction Architecture**, seamlessly integrating Multi-class Classification and Continuous Value Estimation (Regression). This "detect anomaly first, then quantify severity" design logic is highly applicable not only in agricultural biometrics but also in industrial smart manufacturing, yield analysis, and defect detection scenarios.

## Repository Structure
The project is built with a highly modular design, decoupling data cleaning, model training, and prediction business logic to ensure high maintainability and scalability:

```text
Rice-Variety-Adulteration-Detection/
├── data/
│   └── 2024SLTrainData.csv       # Raw training dataset (150 pure rice samples)
├── src/
│   ├── dataCleaning.R           # Data preprocessing, sampling, and mixed data augmentation
│   ├── classification.R         # Classification model arsenal (SVM, RF, LDA, GLMNET)
│   ├── ratioEstimation.R        # Linear regression models for continuous ratio estimation
│   └── main.R                   # Core MRPM pipeline 
└── README.md
```

## Contributors

This project was co-developed by:
* **Kun-Hong Liao** ([@sOmniAbulisT](https://github.com/sOmniAbulisT))
* **Yu-Syuan Wang** ([@SeanYW1999](https://github.com/SeanYW1999))

*This repository contains the refactored and modularized version of our original coursework.*