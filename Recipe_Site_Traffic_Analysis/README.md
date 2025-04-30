# Recipe Site Traffic Analysis

## Project Goal

To analyze recipe characteristics (nutritional information, category, servings) and build a machine learning model to predict whether a recipe will generate high traffic ('High') or low traffic ('Low') when featured on the website's homepage.

## Business Objective

The primary goal is to identify at least 80% of the recipes that would genuinely become popular (high recall for the 'High' class) while minimizing the number of unpopular recipes that are mistakenly promoted (maximizing precision at the target recall level).

## Data

The dataset (`recipe_site_traffic_2212.csv`) contains information about recipes, including:

* `recipe`: Unique identifier
* `calories`: Energy content
* `carbohydrate`: Carbohydrate content (g)
* `sugar`: Sugar content (g)
* `protein`: Protein content (g)
* `category`: Type of recipe (e.g., Breakfast, Chicken, Dessert)
* `servings`: Number of servings the recipe makes
* `high_traffic`: Target variable (initially 'High' or missing)

## Analysis & Modeling Steps

1.  **Data Validation:**
    * Checked data types and identified missing values (52 in nutritional columns, 373 in `high_traffic`).
    * Confirmed `recipe` ID is unique and suitable as a key.
    * Verified nutritional columns contained only non-negative values.
    * Identified inconsistencies in `category` ('Chicken' vs 'Chicken Breast') and `servings` (non-numeric entries like '4 as a snack').

2.  **Data Cleaning & Preparation:**
    * Standardized `category` by mapping 'Chicken Breast' to 'Chicken'.
    * Cleaned `servings` by converting valid entries to integers and dropping 3 rows with non-numeric text.
    * Imputed missing nutritional values using the median value for each recipe's category.
    * Filled missing `high_traffic` values with 'Low'.
    * Created a binary target variable `high_traffic_flag` (1 for 'High', 0 for 'Low').
    * Converted `category` to a categorical data type for modeling.

3.  **Exploratory Data Analysis (EDA):**
    * Visualized the distribution of `calories`, noting a strong right skew and outliers.
    * Examined recipe counts per category, finding 'Chicken' and 'Breakfast' most common.
    * Plotted `carbohydrate` vs. `protein`, showing no clear separation between high/low traffic based on these features alone.

4.  **Model Development:**
    * **Problem Type:** Binary Classification.
    * **Features:** `calories`, `carbohydrate`, `sugar`, `protein`, `servings`, `category`.
    * **Target:** `high_traffic_flag`.
    * **Preprocessing:** Used `StandardScaler` for numeric features and `OneHotEncoder` (dropping first category) for the `category` feature within a Scikit-learn pipeline.
    * **Models:**
        * Logistic Regression (Baseline)
        * Random Forest Classifier (Comparison)
    * **Data Split:** Split data into 80% training and 20% testing sets, stratified by the target variable.

5.  **Model Evaluation:**
    * Compared models using Accuracy, AUC, Precision, and Recall.
    * Logistic Regression showed better overall performance (Accuracy: 0.762, AUC: 0.843) compared to Random Forest (Accuracy: 0.688, AUC: 0.779).
    * At the default 0.5 threshold, Logistic Regression achieved Recall=0.800 (meeting the business goal) and Precision=0.807.
    * Tuned the Logistic Regression threshold to find the best precision while keeping Recall >= 0.80. The optimal threshold was found to be **0.520**, yielding Precision=0.829 and Recall=0.800.

## Conclusion & Recommendation

The Logistic Regression model, with a probability threshold adjusted to 0.520, is recommended for deployment. It successfully meets the primary business objective of identifying at least 80% of high-traffic recipes while achieving a precision of approximately 83% on the test set.

## Tools Used

* Python
* Pandas (Data manipulation and analysis)
* Matplotlib & Seaborn (Data visualization)
* Scikit-learn (Preprocessing, Model building, Evaluation)
