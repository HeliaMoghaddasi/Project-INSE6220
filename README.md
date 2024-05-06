# Project: Iris Flower Classification

This project involves building a machine-learning model to classify iris flowers into different species based on their sepal and petal dimensions.

## Dataset

The Iris dataset contains measurements of sepal length, sepal width, petal length, and petal width for three species of iris flowers: Setosa, Versicolor, and Virginica.

* Features:
* Sepal Length (cm)
* Sepal Width (cm)
* Petal Length (cm)
* Petal Width (cm)
##
* Target Variable:
* Species (Setosa, Versicolor, Virginica)

## Methodology
* Data Loading and Exploration:
Download the dataset from the UCI repository.
Load the dataset into a pandas DataFrame.
Explore the dataset to understand its structure and distributions.

* Data Preprocessing:
Check for missing values and handle if necessary.
Encode the categorical target variable (Species) into numerical labels.
* Exploratory Data Analysis (EDA):
Visualize relationships between different features using scatter plots and pair plots.
Analyze distributions of individual features across different species.
* Model Building:
Split the dataset into training and testing sets.
Train a machine learning model (e.g., logistic regression) to classify iris species based on feature values.
* Model Evaluation:
Evaluate the trained model using accuracy score, confusion matrix, and classification report.
Tune hyperparameters if necessary to improve model performance.



## Usage

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
```

## Results

This approach provides a clear summary of the classification model's performance (accuracy) on the test set both before and after applying PCA to the Iris dataset, demonstrating its effectiveness in predicting iris species based on sepal and petal measurements.



## License

[ARCHIVE.ICS](https://archive.ics.uci.edu/)
