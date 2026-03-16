# AI-Based Document Tampering Detection

## Overview

Digital documents such as ID cards, certificates, and legal contracts are increasingly manipulated using image editing tools. Detecting these tampered documents manually is difficult and time-consuming. This project proposes an AI-based system that automatically detects tampered documents using machine learning and computer vision techniques.

## Problem Statement

Document forgery is a serious issue in financial institutions, government departments, and educational organizations. Traditional verification methods cannot reliably detect subtle manipulations such as copy-move attacks, splicing, or pixel-level edits. This project aims to build a deep learning-based system capable of identifying manipulated regions in digital documents.

## Target Audience

* Banks verifying loan documents
* Government agencies verifying ID cards
* Universities validating certificates
* Online platforms verifying uploaded documents

## Dataset

We use the **DocTamper Dataset** from Kaggle.

Dataset link:
https://www.kaggle.com/datasets/dinmkeljiame/doctamper

The dataset contains:

* Authentic documents
* Tampered documents
* Ground truth masks showing manipulated regions

## Technologies Used

* Python
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Deep Learning (CNN)

## Project Structure

```
data_scripts/
    download_dataset.py
    cleaning.py
    preprocessing.py
    eda.py
notebooks/
    eda_analysis.ipynb
report/
    report.tex
```

## Features

* Data collection pipeline
* Data cleaning and preprocessing
* Exploratory data analysis (EDA)
* Feature scaling and encoding
* AI-based tampering detection (future phase)

## Contributors

Add your group member names here.
Moaz Kashif 23L-2626
Umer Khalid 23L-2599

## License

This project is developed for academic purposes.
