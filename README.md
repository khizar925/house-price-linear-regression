# House Price Prediction using Linear Regression from Scratch

A machine learning project that predicts house prices using a Linear Regression model built from scratch with NumPy, implementing gradient descent optimization without using Scikit-learn's pre-built models.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [How It Works](#how-it-works)
- [Making Predictions](#making-predictions)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a **Linear Regression model from scratch** using only NumPy and Pandas. Unlike typical implementations that use Scikit-learn's built-in models, this project demonstrates the underlying mathematics of linear regression by implementing:

- Custom hypothesis function (linear model)
- Mean Squared Error cost function
- Gradient descent optimization algorithm
- Min-Max feature scaling
- Train/test split for model validation

The model predicts house prices based on 13 features including area, bedrooms, bathrooms, and various amenities.

## Dataset

The project uses the `Housing.csv` dataset which contains information about houses with the following features:

**Continuous Features:**
- **price** - House price (target variable)
- **area** - Area of the house in square feet
- **bedrooms** - Number of bedrooms
- **bathrooms** - Number of bathrooms
- **stories** - Number of stories/floors
- **parking** - Number of parking spaces

**Binary Features (0 or 1):**
- **mainroad** - Whether the house is on the main road
- **guestroom** - Presence of a guest room
- **basement** - Presence of a basement
- **hotwaterheating** - Hot water heating availability
- **airconditioning** - Air conditioning availability
- **prefarea** - Located in a preferred area

**Categorical Features (One-Hot Encoded):**
- **furnishing_semi_furnished** - Semi-furnished status
- **furnishing_unfurnished** - Unfurnished status

The dataset is split into 80% training data and 20% test data using a random seed of 42 for reproducibility.

## Features

- **Custom Linear Regression Implementation** - Built from scratch using NumPy
- **Gradient Descent Optimization** - Implements batch gradient descent with 1000 iterations
- **Feature Scaling** - Min-Max normalization for continuous features
- **Train/Test Split** - 80/20 split with random shuffling
- **Cost Function Tracking** - Monitors Mean Squared Error throughout training
- **R² Score Evaluation** - Calculates R² for both training and test sets
- **Prediction Function** - Ready-to-use function for predicting new house prices
- **No Scikit-learn Models** - Demonstrates the mathematics behind linear regression

## Technologies Used

- **Python 3.x**
- **NumPy** - For numerical computations and linear algebra
- **Pandas** - For data loading and manipulation

**Note:** This project intentionally does not use Scikit-learn's pre-built models to demonstrate the underlying mathematics of machine learning algorithms.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/khizar925/house-price-linear-regression.git
cd house-price-linear-regression
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script to train the model and make predictions:

```bash
python main.py
```

The script will:
1. Load and preprocess the housing data
2. Split the data into training and testing sets
3. Train the Linear Regression model
4. Evaluate the model performance
5. Display predictions and metrics

## Model Performance

The model's performance is evaluated using the following metrics:

- **R² Score:** [Add your score]
- **Mean Absolute Error (MAE):** [Add your score]
- **Mean Squared Error (MSE):** [Add your score]
- **Root Mean Squared Error (RMSE):** [Add your score]

## Project Structure

```
house-price-linear-regression/
│
├── Housing.csv           # Dataset file
├── main.py              # Main script with model implementation
├── requirements.txt     # Project dependencies
├── .gitignore          # Git ignore file
└── README.md           # Project documentation
```