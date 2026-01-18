# Zomato Restaurant Analysis

This project analyzes restaurant data from Zomato for two major Indian cities: **Bangalore** and **Pune**. It involves data cleaning, exploratory data analysis (EDA), and machine learning modeling to predict the cost for two people.

## Project Structure

The project consists of several Python scripts, each serving a specific purpose in the data pipeline:

### 1. Data Cleaning
- **`DataCleaning.py`**: Performs initial cleaning on the individual datasets (Bangalore and Pune). It handles missing values for ratings, reviews, and other categorical features (filling with mean/mode).
- **`CDCleaning.py`**: Merges the two datasets into a combined dataset and performs cleaning on the merged data, ensuring consistency across common columns.

### 2. Exploratory Data Analysis (EDA)
- **`DataAnalysis.py`**: Conducts EDA on the individual datasets. It generates visualizations such as:
  - Histograms for pricing and ratings.
  - Correlation heatmaps.
  - Bar charts comparing total restaurants vs. those without delivery options across localities.
  - Word clouds for common restaurant names.
- **`CDAnalysis.py`**: Analyzes the merged (combined) dataset. It includes:
  - Correlation heatmaps for the combined data.
  - Scatter plots (e.g., Pricing vs. Ratings).
  - Outlier detection for price ranges.

### 3. Machine Learning Modeling
- **`CDModeling.py`**: Implements a machine learning pipeline using **XGBoost** to predict the `Pricing_for_2` (cost for two people).
  - **Features used**: Category, Locality, Dining Rating, Dining Review Count, Delivery Rating, Delivery Rating Count.
  - **Process**: 
    - Merges datasets.
    - Encodes categorical variables (`Category`, `Locality`) using `TargetEncoder`.
    - Splits data into training and testing sets.
    - Trains an `XGBRegressor` model.
    - Evaluates the model using R-squared score and cross-validation.
    - Plots Actual vs. Predicted values.

### 4. Utilities
- **`CombinedDataSet.py`**: A helper script to verify the merging process and ensure the dataset dimensions are correct.

## Prerequisites

To run this project, you need the following Python libraries:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost category_encoders wordcloud
```

## Setup & Usage

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd zomato_analysis
    ```

2.  **Prepare the Data**:
    - The scripts currently expect the following CSV files in the `D:\` drive:
        - `D:\Bangalore_Restaurants.csv`
        - `D:\Pune Restaurants.csv`
    - *Note: Please update the file paths in the scripts (`pd.read_csv(...)`) to match the location of your datasets.*

3.  **Run the analysis**:
    You can run individual scripts depending on the analysis you want to perform:

    ```bash
    # For individual city analysis
    python DataAnalysis.py

    # For combined dataset analysis
    python CDAnalysis.py

    # To train and evaluate the model
    python CDModeling.py
    ```

## Insights

- The project compares the dining and delivery landscapes of Bangalore and Pune.
- It identifies high-correlation features affecting pricing.
- The XGBoost model helps in estimating the cost for two based on restaurant characteristics.
