# firm-bankruptcy-predictor
Python code for bankruptcy prediction models using the Polish Companies Dataset.

## Overview
This repository contains the implementation code for a research paper evaluating three corporate bankruptcy prediction models using the Polish Companies Bankruptcy Dataset (UCI Machine Learning Repository).

Models implemented:
- Altman Z-Score (computed in SPSS)
- Logistic Regression (5 Altman variables)
- Logistic Regression (all available financial variables)

## Files
- `logistic_regression_5_var.py` - Logistic regression using the five Altman variables
- `logistic_regression_all_var.py` - Logistic regression using all available financial variables
- `practical_usecase.py` - Predicts bankruptcy probability for individual firms

## Data
Data sourced from the UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data

Code shown is for the T-1 (Year 5) dataset. The same procedure was repeated for Years 1–4 by changing the file path accordingly.

## Running the Use Case
To test the practical use case, run `logistic_regression_all_var.py` first, then run `practical_usecase.py`.

This requires a `Test_firm.xlsx` file placed in the `data/` folder, with two sheets named `Healthy Firm` and `Distressed Firm`, each containing one row of financial ratios using the same column names as the dataset.
