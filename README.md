## Project Title: Employee Turnover Prediction and Retention Analysis

---

## Project Overview

This project focuses on developing a robust machine learning solution to predict employee turnover and identify the key factors contributing to attrition. Leveraging a comprehensive dataset of HR records, the aim is to provide actionable insights that can empower organizations to implement proactive retention strategies, reduce costs associated with high turnover, and foster a more stable and satisfied workforce. The solution includes exploratory data analysis, advanced machine learning model development (Logistic Regression, Random Forest, XGBoost), and a user-friendly deployment on Streamlit Cloud.

---

## Problem Statement

Employee turnover is a significant challenge for organizations, leading to substantial costs in recruitment, training, and lost productivity. High attrition rates can also negatively impact team morale, institutional knowledge, and overall organizational performance. Without a clear understanding of the underlying causes and the ability to predict which employees are at risk of leaving, companies are often forced into reactive measures, which are less efficient and more costly than proactive retention efforts.

---

## Project Objective

The primary objectives of this project are:

1.  **Identify Key Drivers of Turnover**: To perform in-depth exploratory data analysis to uncover the most influential factors and patterns associated with employee departure.
2.  **Develop Predictive Models**: To build and evaluate machine learning models (e.g., Logistic Regression, Random Forest, XGBoost) capable of accurately predicting employee turnover with high precision and recall, especially for the minority class (employees who leave).
3.  **Provide Actionable Insights**: To translate model findings into clear, data-driven recommendations that HR departments and management can utilize to design and implement effective employee retention strategies.
4.  **Create an Accessible Prediction Tool**: To deploy an interactive web application that allows stakeholders to input employee data and receive real-time predictions and insights, thereby making the model's capabilities easily accessible and user-friendly.




## 1. Environment Setup and Data Loading

This section outlines the initial steps taken to prepare the development environment and load the HR dataset. It also provides a detailed description of each column within the dataset.

### Libraries Used:

The following Python libraries are used to facilitate data manipulation, numerical operations, and visualization:

* **Pandas**: For powerful data manipulation and analysis, especially with DataFrames.
* **NumPy**: Provides support for large, multi-dimensional arrays and mathematical functions.
* **Matplotlib.pyplot**: A comprehensive library for creating static, animated, and interactive visualizations.
* **Seaborn**: A statistical data visualization library that offers a high-level interface for drawing attractive and informative statistical graphics.

### Dataset Overview and Column Descriptions:

The dataset, `HR_comma_sep.csv`, is loaded to begin the analysis. An initial inspection of the dataset's head (first few rows) reveals the following columns and their descriptions:

* `satisfaction_level`: Represents the employee's satisfaction level, likely on a scale from 0 to 1.
* `last_evaluation`: Indicates the employee's last evaluation score, also likely on a scale from 0 to 1.
* `number_project`: The total number of projects an employee has worked on.
* `average_montly_hours`: The average number of hours an employee works per month.
* `time_spend_company`: The number of years an employee has spent in the company.
* `Work_accident`: A binary variable indicating whether the employee had a work accident (1 for yes, 0 for no).
* `left`: The target variable, indicating whether the employee left the company (1 for left, 0 for stayed). This is crucial for predicting employee turnover.
* `promotion_last_5years`: A binary variable indicating whether the employee was promoted in the last five years (1 for yes, 0 for no).
* `Department`: The department in which the employee works (e.g., sales, technical, support).
* `salary`: The employee's salary level, categorized as 'low', 'medium', or 'high'.


## 2. Data Inspection and Quality Check

Before proceeding with data analysis and modeling, it's crucial to inspect the dataset's structure and check for any missing values. This step ensures data integrity and helps in planning subsequent preprocessing steps.

### Data Structure:

The `df.info()` method provides a concise summary of the DataFrame. This includes the number of entries, the total number of columns, the number of non-null values for each column, and the data type (Dtype) of each column.

The output indicates:

* **Total Entries**: The dataset contains 14,999 entries (rows).
* **Columns**: There are 10 columns in total.
* **Data Types**:
    * `float64`: `satisfaction_level`, `last_evaluation`
    * `int64`: `number_project`, `average_montly_hours`, `time_spend_company`, `Work_accident`, `left`, `promotion_last_5years`
    * `object`: `Department`, `salary`
* **Memory Usage**: The dataset occupies approximately 1.1+ MB of memory.

This information is vital for understanding the nature of the data (numerical vs. categorical) and ensuring that all columns have the expected data types.

### Missing Values Check:

The `df.isnull().sum()` command is used to count the number of missing (null) values in each column of the DataFrame. A count of zero for all columns indicates a clean dataset with no missing entries.

The output confirms that there are **no missing values** across any of the columns:

* `satisfaction_level`: 0
* `last_evaluation`: 0
* `number_project`: 0
* `average_montly_hours`: 0
* `time_spend_company`: 0
* `Work_accident`: 0
* `left`: 0
* `promotion_last_5years`: 0
* `Department`: 0
* `salary`: 0

The absence of missing values simplifies the data preprocessing phase significantly, as no imputation or removal of rows/columns due to missing data will be required.


## 3. Descriptive Statistics of Numerical Features

After checking the data structure and for missing values, the next step is to examine the **descriptive statistics** of the numerical features in the dataset. This provides a quick quantitative summary, including central tendency, dispersion, and shape of the distribution of each column.

The `df.describe()` method generates these statistics for numerical columns:

* **`count`**: The number of non-null observations. This confirms that all numerical columns have 14,999 entries, aligning with our earlier missing values check.
* **`mean`**: The average value of the column. For example, the average satisfaction level among employees is approximately 0.61, and employees work, on average, around 201 hours per month.
* **`std`**: The standard deviation, which measures the amount of variation or dispersion of a set of values. A higher standard deviation indicates greater variability.
* **`min`**: The minimum value found in the column.
* **`25% (Q1)`**: The first quartile (25th percentile), below which 25% of the data falls.
* **`50% (Q2)`**: The median, or the second quartile (50th percentile), which is the middle value in the dataset.
* **`75% (Q3)`**: The third quartile (75th percentile), below which 75% of the data falls.
* **`max`**: The maximum value found in the column.

### Key Observations from Descriptive Statistics:

* **`satisfaction_level`**: Ranges from 0.09 to 1.00, with an average of 0.61. The standard deviation of 0.25 suggests a fair spread in satisfaction levels.
* **`last_evaluation`**: Ranges from 0.36 to 1.00, with an average of 0.72. The distribution appears relatively broad.
* **`number_project`**: Employees work on a minimum of 2 projects and a maximum of 7. The average is about 3.8 projects.
* **`average_montly_hours`**: Monthly hours range from 96 to 310, with an average of approximately 201 hours. This wide range suggests varied workloads.
* **`time_spend_company`**: Employees have spent between 2 and 10 years at the company, with an average of 3.5 years.
* **`Work_accident`**: Since this is a binary variable (0 or 1), the mean (0.14) indicates that roughly 14.5% of employees have had a work accident.
* **`left`**: As our target variable (0 or 1), the mean (0.238) signifies that about 23.8% of employees in this dataset have left the company. This highlights the class imbalance, which is an important consideration for modeling.
* **`promotion_last_5years`**: The mean (0.021) suggests that only about 2.1% of employees received a promotion in the last five years, indicating that promotions are relatively rare.

These statistics provide valuable insights into the distribution and characteristics of the numerical data, helping to identify potential outliers, skewness, or other patterns that might require further investigation or preprocessing.


## 4. Analysis of Employee Turnover (Target Variable)

A fundamental step in understanding employee attrition is to analyze the distribution of the target variable, 'left'. This variable indicates whether an employee has left the company (1) or stayed (0). This analysis helps to identify the balance between employees who stayed and those who departed, which is crucial for subsequent modeling efforts.

### Visualization: Employee Retention vs. Attrition

A count plot was generated to visually represent the number of employees who stayed versus those who left.

![count of target variable](https://github.com/Phenomkay/Employee-Retention-Forecast/blob/81f4060d31ca12b4e1e98b83c81d53f212e0abae/count%20of%20target%20variable.png)


* **X-axis**: Represents the 'left' variable, with labels 'Stayed' (0) and 'Left' (1).
* **Y-axis**: Shows the count of employees.

The plot clearly illustrates a significant imbalance, with a much larger number of employees having stayed compared to those who left.

### Quantitative Summary:

To provide exact figures, the counts and percentages for each category of the 'left' variable were calculated:

* **Counts**:
    * **0 (Stayed)**: 11428 employees
    * **1 (Left)**: 3571 employees

* **Percentages**:
    * **0 (Stayed)**: 76.19% of employees
    * **1 (Left)**: 23.81% of employees

### Key Insight:

This analysis confirms that approximately **76.19% of employees remained with the company**, while **23.81% departed**. This imbalance is a critical observation as it can impact the performance of predictive models, potentially leading to models that are biased towards the majority class (employees who stayed). Future modeling considerations will need to address this class imbalance, possibly through techniques like oversampling, undersampling, or using appropriate evaluation metrics.


## 5. Distribution of Numerical Features by Employee Status

To gain deeper insights into the factors influencing employee turnover, the distributions of key numerical features were examined, distinguishing between employees who stayed (`left = 0`) and those who left (`left = 1`). This analysis helps in identifying patterns and potential thresholds where employee behavior diverges significantly.

Kernel Density Estimate (KDE) plots were used for the following numerical features:

* `satisfaction_level`
* `last_evaluation`
* `number_project`
* `average_montly_hours`
* `time_spend_company`

### Key Observations:

* **`satisfaction_level`**:
    * Employees who stayed (`left = 0`) tend to have higher satisfaction levels, showing a peak around 0.7-0.9.
    * Employees who left (`left = 1`) show a distinct bimodal distribution with two significant peaks: one for very low satisfaction (around 0.1-0.2) and another for moderately high satisfaction (around 0.7-0.9). The peak at low satisfaction is particularly strong for those who left, suggesting that low satisfaction is a major driver of attrition.
* **`last_evaluation`**:
    * For employees who stayed, the distribution is somewhat normal, peaking around 0.7-0.8.
    * For those who left, there are peaks at both lower (around 0.5) and higher (around 0.8-0.9) evaluation scores. This bimodal pattern suggests that both underperformers and highly evaluated employees might be prone to leaving.
* **`number_project`**:
    * Employees who stayed primarily engaged in 3, 4, or 5 projects.
    * Employees who left show distinct peaks for those who worked on 2, 6, or 7 projects. This suggests that having too few (2) or too many (6, 7) projects might contribute to an employee's decision to leave.
* **`average_montly_hours`**:
    * Employees who stayed generally fall within the 150-250 hours/month range, with a peak around 200 hours.
    * Employees who left exhibit a bimodal distribution: one group worked very few hours (around 150), and another group worked very long hours (around 250-300). This indicates that both underutilization and overwork could be factors in attrition.
* **`time_spend_company`**:
    * Employees who stayed mostly have spent 2, 3, or 4 years at the company.
    * Employees who left show strong peaks at 3, 4, 5, and particularly 6 years. This suggests that the 3-6 year mark might be a critical period for employee retention.

These visualizations provide critical insights into the underlying patterns and differences between employees who stay and those who leave, highlighting features that are highly correlated with employee turnover.


## 6. Distribution of Categorical Features by Employee Status

To further understand the dynamics of employee turnover, the distribution of categorical features was analyzed in relation to whether an employee stayed (`left = 0`) or left (`left = 1`). This helps in identifying specific categories within each feature that are more prone to attrition.

Count plots were generated for the following categorical features:

* `Department`
* `salary`
* `promotion_last_5years`
* `Work_accident`

### Key Observations:

* **`Department` vs. Employee Status**:
    * The 'sales' department has the highest overall number of employees and also the highest number of employees who left.
    * Departments like 'technical' and 'support' also show significant numbers of employees leaving.
    * Conversely, 'management' and 'RandD' (Research and Development) departments appear to have lower attrition rates relative to their total employee count, suggesting better retention.

* **`salary` vs. Employee Status**:
    * Employees with 'low' and 'medium' salaries constitute the vast majority of those who left the company.
    * A significantly smaller proportion of employees with 'high' salaries departed, indicating that higher compensation is strongly correlated with employee retention.

* **`promotion_last_5years` vs. Employee Status**:
    * The vast majority of employees who left did not receive a promotion in the last five years.
    * Only a very small number of promoted employees decided to leave, highlighting that promotion is a strong factor in retaining employees.

* **`Work_accident` vs. Employee Status**:
    * A much larger proportion of employees who left had *not* experienced a work accident.
    * Employees who had a work accident appear less likely to leave compared to those who did not, suggesting that perhaps support or benefits post-accident contribute to retention, or those prone to accidents are in roles with inherently lower turnover.

These visualizations provide critical insights into which categories within each feature are most susceptible to employee turnover, offering actionable areas for HR interventions.


## 7. Correlation Analysis of Numerical Features

To understand the linear relationships between the numerical features and their relationship with the target variable (`left`), a correlation matrix was computed and visualized using a heatmap. Correlation coefficients range from -1 to 1, where:

* **1** indicates a perfect positive linear correlation.
* **-1** indicates a perfect negative linear correlation.
* **0** indicates no linear correlation.

### Heatmap Visualization:

The heatmap provides a clear visual representation of these correlations.

* **Color Scale**: Red shades indicate positive correlations, while blue shades indicate negative correlations. The intensity of the color corresponds to the strength of the correlation.
* **Annotations**: Each cell displays the correlation coefficient, rounded to two decimal places, for better readability.

### Key Observations from the Correlation Matrix:

* **`left` (Target Variable) Correlations**:
    * **Negative Correlation with `satisfaction_level` (-0.39)**: This is the strongest correlation with the `left` variable, indicating that employees with lower satisfaction levels are much more likely to leave the company. This aligns with findings from the distribution plots.
    * **Negative Correlation with `Work_accident` (-0.15)**: Employees who had a work accident are slightly less likely to leave. This reinforces the observation from the categorical plots.
    * **Negative Correlation with `promotion_last_5years` (-0.06)**: While weak, it suggests that employees who received a promotion are slightly less likely to leave, although its impact is minor compared to satisfaction.
    * **Weak Positive Correlations**: `number_project`, `average_montly_hours`, and `time_spend_company` show weak positive correlations with `left`, meaning that employees with a higher number of projects, more average monthly hours, or more time spent at the company have a very slightly increased tendency to leave.
    * **Negligible Correlation with `last_evaluation` (0.01)**: `last_evaluation` appears to have almost no linear relationship with whether an employee leaves.

* **Inter-Feature Correlations**:
    * **`number_project` and `average_montly_hours` (0.42)**: There is a moderate positive correlation, which is intuitive as more projects often lead to more working hours.
    * **`number_project` and `last_evaluation` (0.35)**: Employees with more projects tend to have higher last evaluation scores.
    * **`average_montly_hours` and `last_evaluation` (0.34)**: Similar to projects, more working hours are moderately correlated with higher evaluation scores.
    * **`time_spend_company` and `number_project` (0.20)**: Employees who have spent more time at the company tend to work on more projects.

This correlation analysis confirms the strong inverse relationship between employee satisfaction and attrition, and provides insights into other subtle linear dependencies within the dataset.


## 8. Data Preprocessing

Before training any machine learning models, the raw data needs to be preprocessed. This involves converting categorical variables into a numerical format that models can understand and splitting the data into training and testing sets.

The following preprocessing steps were applied:

### 8.1. Encoding Categorical Features

* **Ordinal Encoding for `salary`**: The `salary` column, which has an inherent order ('low', 'medium', 'high'), was converted into numerical values using ordinal encoding. This ensures that the model recognizes the natural hierarchy of salary levels:
    * 'low' was mapped to 0
    * 'medium' was mapped to 1
    * 'high' was mapped to 2

* **One-Hot Encoding for `Department`**: The `Department` column, being nominal (categories without inherent order), was transformed using one-hot encoding. This creates new binary columns for each department category, preventing the model from assuming any false ordinal relationships. The `drop_first=True` argument was used to avoid multicollinearity.

### 8.2. Feature and Target Separation

The preprocessed DataFrame was then split into features (`x`) and the target variable (`y`):

* **Features (`x`)**: All columns except the `left` column. These are the independent variables used to predict employee turnover.
* **Target (`y`)**: The `left` column, which indicates whether an employee left the company (1) or stayed (0).

### 8.3. Train-Test Split

The dataset was divided into training and testing sets to evaluate the model's performance on unseen data.

* **Split Ratio**: 80% of the data was allocated for training and 20% for testing (`test_size=0.2`).
* **Random State**: `random_state=42` was used to ensure reproducibility of the split.
* **Stratification**: Critically, the `stratify=y` parameter was used during the split. Given the class imbalance observed in the `left` variable (more employees stayed than left), stratification ensures that both the training and testing sets maintain the same proportion of 'left' (1) and 'stayed' (0) employees as the original dataset. This is vital for training and evaluating models effectively on imbalanced datasets.

After these preprocessing steps, the shapes of the resulting datasets are:

* **Training Features Shape**: (11999, 17)
* **Test Features Shape**: (3000, 17)

This indicates that the training set contains 11,999 samples with 17 features, and the test set contains 3,000 samples with 17 features. The increase in feature count (from 10 to 17) is due to the one-hot encoding of the 'Department' column.


## 9\. Model Training and Evaluation: Logistic Regression

With the data preprocessed, the next step involves training a machine learning model to predict employee turnover and evaluating its performance. Logistic Regression, a commonly used algorithm for binary classification, was chosen for this task.

### 9.1. Model Initialization and Training

  * **Model**: A `LogisticRegression` model was initialized from `sklearn.linear_model`.
  * **`max_iter=1000`**: The maximum number of iterations for the solver to converge was increased to 1000 to ensure the model has enough iterations to find a good solution.
  * **`class_weight='balanced'`**: This crucial parameter was set to 'balanced' to automatically adjust weights inversely proportional to class frequencies. This helps in mitigating the impact of the class imbalance (more employees stayed than left), preventing the model from being biased towards the majority class and improving its ability to correctly identify the minority class (employees who left).
  * **`random_state=42`**: A fixed random state ensures reproducibility of the model's training process.

The model was then trained using the `x_train` (features) and `y_train` (target) datasets.

### 9.2. Prediction

After training, the model was used to predict the `left` status for the unseen `x_test` dataset. These predictions (`lr_y_pred`) are then compared against the actual values (`y_test`) to evaluate the model's performance.

### 9.3. Model Evaluation

The performance of the Logistic Regression model was assessed using several key metrics:

  * **Accuracy Score**:

      * **Accuracy: 0.7687 (76.87%)**
      * Accuracy represents the proportion of correctly classified instances out of the total. While seemingly good, for imbalanced datasets, accuracy alone can be misleading as it can be high even if the minority class is poorly predicted.

  * **Confusion Matrix**:
    The confusion matrix provides a detailed breakdown of correct and incorrect predictions:

    ```
    [[1736  550]
     [ 144  570]]
    ```

      * **True Negatives (TN)**: 1736 employees who stayed (0) were correctly predicted as stayed.
      * **False Positives (FP)**: 550 employees who stayed (0) were incorrectly predicted as left (Type I error).
      * **False Negatives (FN)**: 144 employees who left (1) were incorrectly predicted as stayed (Type II error).
      * **True Positives (TP)**: 570 employees who left (1) were correctly predicted as left.

  * **Classification Report**:
    The classification report provides precision, recall, and F1-score for each class (0: stayed, 1: left):

    ```
                  precision    recall  f1-score   support

              0       0.92      0.76      0.83      2286
              1       0.51      0.80      0.62       714

       accuracy                           0.77      3000
      macro avg       0.72      0.78      0.73      3000
    weighted avg       0.82      0.77      0.78      3000
    ```

      * **Class 0 (Stayed)**:

          * **Precision (0.92)**: When the model predicts an employee will stay, it is correct 92% of the time.
          * **Recall (0.76)**: The model correctly identifies 76% of all employees who actually stayed.
          * **F1-score (0.83)**: The harmonic mean of precision and recall for Class 0.

      * **Class 1 (Left)**:

          * **Precision (0.51)**: When the model predicts an employee will leave, it is correct only 51% of the time. This means there are a significant number of false alarms (employees predicted to leave who actually stayed).
          * **Recall (0.80)**: The model successfully identifies 80% of all employees who actually left. This is a good recall for the minority class, indicating that the `class_weight='balanced'` parameter helped.
          * **F1-score (0.62)**: The harmonic mean for Class 1.

      * **Macro Avg**: Averages the metrics for each class without considering class imbalance.

      * **Weighted Avg**: Averages the metrics, weighting each by the number of true instances for each class.

### Summary of Model Performance:

The Logistic Regression model, with balanced class weights, achieved a decent overall accuracy of 76.87%. More importantly, its high recall for the minority class (employees who left) at 80% indicates its effectiveness in identifying potential attrition cases. However, the lower precision for the 'left' class (51%) suggests that while it catches many who leave, it also generates a notable number of false positives (predicting attrition when it doesn't occur). This trade-off between precision and recall is common in imbalanced classification and needs to be considered based on the business objective (e.g., is it more critical to catch all potential leavers, or to minimize false alarms?).


## 10\. Model Training and Evaluation: Random Forest Classifier

Following the Logistic Regression model, a more powerful ensemble method, the Random Forest Classifier, was employed to predict employee turnover. Random Forest models are known for their ability to handle complex relationships and often yield higher accuracy.

### 10.1. Model Initialization and Training

  * **Model**: A `RandomForestClassifier` was initialized from `sklearn.ensemble`.
  * **`n_estimators=100`**: This parameter specifies the number of trees in the forest. A value of 100 is a common starting point and generally provides good performance.
  * **`class_weight='balanced'`**: Similar to the Logistic Regression model, this crucial parameter was set to 'balanced' to account for the class imbalance in the target variable. This ensures that the model gives proper attention to both the majority (stayed) and minority (left) classes during training.
  * **`random_state=42`**: A fixed random state ensures the reproducibility of the model's training process and results.

The Random Forest model was then trained using the preprocessed `x_train` (features) and `y_train` (target) datasets.

### 10.2. Prediction

After training, the model made predictions (`y_pred_rf`) on the unseen `x_test` dataset.

### 10.3. Model Evaluation

The performance of the Random Forest Classifier was thoroughly evaluated using the same set of metrics:

  * **Accuracy Score**:

      * **Accuracy: 0.9910 (99.10%)**
      * This indicates an exceptionally high overall accuracy, suggesting that the model correctly classifies nearly all instances.

  * **Confusion Matrix**:
    The confusion matrix reveals the breakdown of correct and incorrect predictions:

    ```
    [[2283    3]
     [  24  690]]
    ```

      * **True Negatives (TN)**: 2283 employees who stayed (0) were correctly predicted as stayed. (Very high)
      * **False Positives (FP)**: Only 3 employees who stayed (0) were incorrectly predicted as left. (Extremely low - excellent precision for class 0)
      * **False Negatives (FN)**: 24 employees who left (1) were incorrectly predicted as stayed. (Very low - excellent recall for class 1)
      * **True Positives (TP)**: 690 employees who left (1) were correctly predicted as left. (Very high)

  * **Classification Report**:
    The classification report details precision, recall, and F1-score for each class:

    ```
                  precision    recall  f1-score   support

              0       0.99      1.00      0.99      2286
              1       1.00      0.97      0.98       714

       accuracy                           0.99      3000
      macro avg       0.99      0.98      0.99      3000
    weighted avg       0.99      0.99      0.99      3000
    ```

      * **Class 0 (Stayed)**:

          * **Precision (0.99)**: When the model predicts an employee will stay, it is correct 99% of the time.
          * **Recall (1.00)**: The model correctly identifies virtually all employees who actually stayed.
          * **F1-score (0.99)**: Nearly perfect F1-score for Class 0.

      * **Class 1 (Left)**:

          * **Precision (1.00)**: When the model predicts an employee will leave, it is correct 100% of the time. This is outstanding, meaning no false alarms for this class.
          * **Recall (0.97)**: The model successfully identifies 97% of all employees who actually left. This indicates very few missed attrition cases.
          * **F1-score (0.98)**: An exceptionally high F1-score for the minority class, demonstrating robust performance in identifying employees at risk of leaving.

### Summary of Random Forest Performance:

The Random Forest Classifier delivered exceptional performance in predicting employee turnover. With an accuracy of 99.10% and near-perfect precision and recall scores for both classes, especially for the crucial 'left' class, this model is highly effective. The low number of false positives and false negatives signifies its strong predictive power and ability to correctly identify both retention and attrition cases. This model appears to be highly suitable for the task of predicting employee turnover.


## 11\. Model Training and Evaluation: XGBoost Classifier

To further explore robust predictive models for employee turnover, an XGBoost (Extreme Gradient Boosting) Classifier was implemented. XGBoost is a highly efficient and effective open-source library that provides a gradient boosting framework.

### 11.1. Model Initialization and Training

  * **Model**: An `XGBClassifier` was initialized.
  * **`use_label_encoder=False`**: This parameter silences a deprecation warning, indicating that label encoding is not being used internally for the target variable (which is already numerical).
  * **`eval_metric='logloss'`**: Specifies the evaluation metric used during training. 'logloss' is a common metric for binary classification, measuring the prediction error.
  * **`scale_pos_weight=3.2`**: This is a crucial parameter for handling imbalanced datasets in XGBoost. It's manually calculated as the ratio of the number of negative samples (employees who stayed) to the number of positive samples (employees who left). In this case, approximately 11428 / 3571 $\\approx$ 3.2. This parameter gives more weight to the minority class (employees who left), helping the model to focus on correctly identifying them.
  * **`random_state=42`**: Ensures the reproducibility of the training process.

The XGBoost model was then trained using the `x_train` and `y_train` datasets.

### 11.2. Prediction

After training, the model generated predictions (`y_pred_xgb`) on the held-out `x_test` dataset.

### 11.3. Model Evaluation

The performance of the XGBoost Classifier was rigorously evaluated using the standard metrics:

  * **Accuracy Score**:

      * **Accuracy: 0.9863 (98.63%)**
      * This indicates a very high overall accuracy, suggesting the model is performing exceptionally well in classifying instances correctly.

  * **Confusion Matrix**:
    The confusion matrix provides a detailed breakdown of the model's predictions:

    ```
    [[2267   19]
     [  22  692]]
    ```

      * **True Negatives (TN)**: 2267 employees who stayed (0) were correctly predicted as stayed. (Very high)
      * **False Positives (FP)**: 19 employees who stayed (0) were incorrectly predicted as left. (Very low, indicating good precision for class 0)
      * **False Negatives (FN)**: 22 employees who left (1) were incorrectly predicted as stayed. (Very low, indicating good recall for class 1)
      * **True Positives (TP)**: 692 employees who left (1) were correctly predicted as left. (Very high)

  * **Classification Report**:
    The classification report presents precision, recall, and F1-score for each class:

    ```
                  precision    recall  f1-score   support

              0       0.99      0.99      0.99      2286
              1       0.97      0.97      0.97       714

       accuracy                           0.99      3000
      macro avg       0.98      0.98      0.98      3000
    weighted avg       0.99      0.99      0.99      3000
    ```

      * **Class 0 (Stayed)**:

          * **Precision (0.99)**: High confidence when predicting 'stayed'.
          * **Recall (0.99)**: Captures almost all employees who actually stayed.
          * **F1-score (0.99)**: Excellent balance of precision and recall.

      * **Class 1 (Left)**:

          * **Precision (0.97)**: When the model predicts an employee will leave, it is correct 97% of the time, signifying very few false alarms for attrition.
          * **Recall (0.97)**: The model successfully identifies 97% of all employees who actually left, indicating it's highly effective at catching potential leavers.
          * **F1-score (0.97)**: An outstanding F1-score for the minority class, showcasing strong performance on the critical attrition prediction task.

### Summary of XGBoost Performance:

The XGBoost Classifier demonstrated exceptional performance, with an accuracy of 98.63%. It achieved impressive precision and recall scores for both classes, particularly for the 'left' class (97% precision and 97% recall). This indicates that the model is highly effective at identifying employees at risk of attrition with very few false positives or false negatives. While slightly lower in overall accuracy than the Random Forest, its balanced performance metrics across both classes, especially for the minority class, are highly commendable and make it a robust model for this problem.


## 12. Cross-Validation and ROC-AUC Analysis

While single train-test splits provide a good initial assessment, **cross-validation (CV)** offers a more robust evaluation of a model's performance by training and testing it on different subsets of the data multiple times. This helps to ensure that the model's performance is consistent and not merely a result of a particular data split.

The **Receiver Operating Characteristic (ROC) curve** and its associated **Area Under the Curve (AUC)** score are particularly valuable metrics for binary classification, especially with imbalanced classes. ROC-AUC measures the model's ability to distinguish between the positive (left) and negative (stayed) classes across all possible classification thresholds. An AUC score closer to 1 indicates a better ability to discriminate.

### 12.1. Cross-Validation Strategy

* **`StratifiedKFold`**: A stratified K-fold cross-validation strategy was employed with 5 splits (`n_splits=5`). This is crucial for imbalanced datasets as it ensures that each fold maintains the same proportion of classes as the original dataset, preventing any fold from having an unrepresentative class distribution.
* **`shuffle=True`**: Shuffling the data before splitting helps in ensuring randomness across folds.
* **`random_state=42`**: Ensures reproducibility of the folds.

### 12.2. Random Forest Cross-Validation ROC-AUC

The Random Forest model was evaluated using this cross-validation strategy, with `roc_auc` as the scoring metric.

* **Individual Fold ROC-AUC Scores**:
    * 0.9915
    * 0.9923
    * 0.9948
    * 0.9957
    * 0.9937

* **Mean ROC-AUC Score**: 0.9936

The consistently high ROC-AUC scores across all folds, with a mean of 0.9936, indicate that the Random Forest model is highly robust and generalizes exceptionally well in distinguishing between employees who stay and those who leave.

### 12.3. XGBoost Cross-Validation ROC-AUC

Similarly, the XGBoost model was subjected to the same cross-validation evaluation.

* **Individual Fold ROC-AUC Scores**:
    * 0.9940
    * 0.9926
    * 0.9930
    * 0.9944
    * 0.9946

* **Mean ROC-AUC Score**: 0.9937

The XGBoost model also demonstrates outstanding and consistent performance across the folds, with a mean ROC-AUC score of 0.9937. This confirms its strong discriminative power and reliability.

### Conclusion from Cross-Validation:

Both the Random Forest and XGBoost models exhibit excellent and highly stable performance in predicting employee turnover, as evidenced by their high mean ROC-AUC scores across stratified cross-validation folds. The very slight difference in their mean AUC suggests both models are equally strong contenders for this prediction task, demonstrating superior generalization capabilities.


## 13. ROC Curve Comparison

To visually compare the discriminative power of the top-performing models (Random Forest and XGBoost), their Receiver Operating Characteristic (ROC) curves were plotted. The ROC curve illustrates the trade-off between the True Positive Rate (TPR, or Sensitivity) and the False Positive Rate (FPR, or 1 - Specificity) at various classification thresholds. The Area Under the Curve (AUC) provides a single scalar value that summarizes the overall performance.

### Visualization: ROC Curve Comparison

The plot displays the ROC curves for both models:

* **X-axis**: False Positive Rate (FPR)
* **Y-axis**: True Positive Rate (TPR)
* **Diagonal Dashed Line**: Represents a random classifier (AUC = 0.5). A good model's ROC curve should be as far as possible from this line and towards the top-left corner.

The curve for **Random Forest (AUC = 0.992)** and **XGBoost (AUC = 0.995)** are both positioned very close to the top-left corner of the plot, indicating excellent performance.

### Key Insights from ROC Curve Comparison:

* Both Random Forest and XGBoost models demonstrate **exceptional discriminatory ability**, as their ROC curves closely hug the top-left corner of the plot. This signifies that they are very good at distinguishing between employees who will stay and those who will leave.
* The **AUC scores (0.992 for Random Forest and 0.995 for XGBoost)** are remarkably high, further confirming their strong performance. An AUC value close to 1.0 indicates near-perfect classification.
* While both models perform outstandingly, the XGBoost model shows a **slight edge in AUC (0.995 vs 0.992)**, suggesting it is marginally better at discriminating between the classes across various thresholds. This aligns with the mean AUC scores observed during cross-validation.

This visual and quantitative comparison reinforces the conclusion that both Random Forest and XGBoost are highly effective models for predicting employee turnover, with XGBoost showing a marginal lead in overall discriminative performance.


## 14. Feature Importance Analysis

Understanding which features contribute most significantly to the model's predictions is vital for deriving actionable insights and making informed business decisions. For the Random Forest Classifier, feature importance scores were extracted to identify the most influential factors in predicting employee turnover.

### Methodology:

The `feature_importances_` attribute of the trained Random Forest model was used to obtain the importance score for each feature. These scores represent the average decrease in impurity (or Gini impurity for classification tasks) across all trees in the forest, weighted by the proportion of samples reaching that node. A higher score indicates a more important feature.

The features were then sorted by their importance in descending order, and the top 10 were visualized.

### Visualization: Top 10 Feature Importances

A horizontal bar plot was generated to display the top 10 most important features.

* **X-axis**: Importance Score
* **Y-axis**: Feature Name

The plot clearly highlights the hierarchy of influence among the features.

### Key Insights from Feature Importance:

The analysis of feature importance reveals the following critical drivers of employee turnover:

1.  **`satisfaction_level` (Importance: ~0.279)**: This is by far the most important feature. Employees' satisfaction levels play the most significant role in determining whether they leave the company. This reinforces earlier observations from the distribution plots and correlation analysis.
2.  **`time_spend_company` (Importance: ~0.240)**: The number of years an employee has spent at the company is the second most influential factor. This suggests that certain tenure milestones might be critical periods for attrition.
3.  **`number_project` (Importance: ~0.159)**: The number of projects an employee is involved in is also highly important, indicating that workload (too few or too many projects) can be a significant factor.
4.  **`average_montly_hours` (Importance: ~0.143)**: Closely related to `number_project`, the average monthly hours worked is another key determinant, highlighting the impact of work-life balance.
5.  **`last_evaluation` (Importance: ~0.133)**: Employee evaluation scores also hold substantial importance, suggesting that performance reviews are a significant factor in retention.

Following these top five numerical features, other factors like `salary`, `Work_accident`, and specific `Department` categories (e.g., technical, sales, support) also contribute, albeit to a lesser extent. `promotion_last_5years` appears to have the least impact among the listed features.

### Conclusion from Feature Importance:

The feature importance analysis provides clear, actionable insights for HR. **Employee satisfaction is the paramount factor influencing turnover**, followed closely by tenure, workload (projects and hours), and performance evaluations. Addressing these key areas would likely yield the most significant impact on improving employee retention.


## 15. Deployment with Streamlit Cloud and Custom UI

To make the employee turnover prediction model accessible and interactive, the project has been deployed as a user-friendly web application on Streamlit Cloud. The application features a custom user interface (UI) designed with a combination of CSS and HTML to enhance its aesthetic appeal and user experience, visually communicating the context of employee management.

### 15.1. Streamlit Cloud Overview

Streamlit Cloud provides a straightforward and efficient platform for deploying data science and machine learning applications directly from a GitHub repository. It streamlines the infrastructure setup, enabling quick deployment and easy sharing of interactive models and dashboards.

### 15.2. Custom User Interface (UI) Design

A key focus of the deployment was to create an intuitive and visually engaging interface. This was achieved by integrating custom CSS and HTML elements within the Streamlit application, allowing for:

* **Tailored Input Fields**: The application utilizes direct input fields for all user entries, providing a clear and precise way for users to interact with the model.
* **Enhanced Visuals**: Custom styling elements contribute to a polished and professional look.
* **Contextual Background**: An image depicting employees at work serves as a background, immediately setting the context for the HR-focused prediction task.

This combination of Streamlit's robust application capabilities with custom front-end styling results in a highly functional and aesthetically pleasing tool for predicting employee turnover.

### 15.3. Accessing the Live Application

You can interact with the deployed Employee Retention Forecast application live by clicking on the link below:

[**Employee Retention Forecast Application**](https://employee-retention-forecast.streamlit.app/)


## 16. Conclusion and Future Enhancements

This project successfully developed and deployed a machine learning model to predict employee turnover, providing valuable insights into the factors influencing retention. Through comprehensive data exploration, robust preprocessing, and the evaluation of multiple classification algorithms, a highly accurate and reliable prediction system was established.

### Key Learnings and Achievements:

* **Data-driven Insights**: The exploratory data analysis clearly identified critical factors such as `satisfaction_level`, `time_spend_company`, `number_project`, `average_montly_hours`, and `last_evaluation` as primary drivers of employee turnover.
* **Robust Model Performance**: Both Random Forest and XGBoost classifiers demonstrated exceptional predictive capabilities, achieving high accuracy, precision, and recall, particularly for identifying employees at risk of leaving. The XGBoost model showed a marginal edge in overall discriminative performance.
* **Effective Imbalance Handling**: The consistent use of strategies like `stratify` in train-test split and `class_weight='balanced'` or `scale_pos_weight` in models proved effective in addressing the inherent class imbalance in the attrition dataset.
* **Deployment and Accessibility**: The deployment on Streamlit Cloud with a custom UI makes the model accessible and actionable for stakeholders, enabling interactive predictions based on employee attributes.

### Challenges Encountered:

The primary challenge encountered during this project was in **designing and implementing a robust and aesthetically pleasing custom user interface (UI) for the Streamlit application**. This involved navigating various CSS and HTML integration complexities and resolving several error bugs to ensure a seamless and intuitive user experience. Overcoming these UI challenges was a significant learning curve and ultimately contributed to a more polished final product.

### Future Enhancements:

Several avenues could be explored to further enhance this project:

* **Advanced Feature Engineering**: Investigate creating more complex features from existing data, such as interaction terms or polynomial features, to potentially capture non-linear relationships.
* **Model Interpretability**: Delve deeper into model interpretability techniques (e.g., SHAP values, LIME) to provide more granular explanations for individual predictions, beyond just feature importance.
* **Real-time Data Integration**: Explore integrating the application with a live HR database for real-time predictions and continuous monitoring of employee retention risks.
* **A/B Testing of Interventions**: Once implemented, the model's predictions could be used to guide HR interventions, and the effectiveness of these interventions could be A/B tested to measure their impact on actual retention rates.
* **User Feedback and Iteration**: Gather feedback from HR professionals to refine the application's features and UI, making it even more practical and user-friendly.

This project serves as a solid foundation for proactive employee retention strategies, offering a powerful tool for predicting and understanding workforce dynamics.


## 17. Key Insights and Recommendations for Employee Retention

The comprehensive analysis of the HR dataset and the development of predictive models have yielded several critical insights into the drivers of employee turnover. Based on these findings, the following actionable recommendations are provided to help the company enhance employee retention strategies.

### Key Insights:

1.  **Satisfaction is the Foremost Driver of Attrition:**
    * Employee `satisfaction_level` is the most significant predictor of whether an employee will leave. While very low satisfaction is a clear indicator, a notable segment of employees with surprisingly high satisfaction levels also tend to leave. This suggests diverse reasons for departure, not solely direct dissatisfaction.

2.  **Critical Tenure Milestones:**
    * The `time_spend_company` is a highly influential factor, with a pronounced increase in attrition rates observed around the **3 to 6-year mark**. This period appears to be a critical juncture where employees evaluate their career path and future within the company.

3.  **Workload Extremes Lead to Turnover:**
    * Both **under-utilization** (employees with 2 projects and lower average monthly hours) and **overwork** (employees with 6-7 projects and very high average monthly hours) significantly contribute to attrition. A balanced workload is essential for retention.

4.  **Performance Evaluation's Dual Role:**
    * `last_evaluation` is an important predictor, indicating that employees at both ends of the performance spectrum (those with low and unusually high evaluation scores) are more prone to leaving. High-performing individuals might seek new challenges or better opportunities if their growth isn't adequately supported internally.

5.  **Compensation and Promotion are Powerful Retention Levers:**
    * Employees in **'low' and 'medium' salary bands** constitute the largest groups of leavers.
    * Conversely, **promotions** are highly effective in retaining employees, with a remarkably low attrition rate among those who have been promoted in the last five years.

6.  **Departmental Hotspots for Attrition:**
    * The **'sales', 'technical', and 'support' departments** exhibit higher absolute numbers of employee departures, indicating potential underlying issues specific to these areas that warrant further investigation.

7.  **Unexpected Impact of Work Accidents:**
    * Surprisingly, employees who **did not** experience a work accident are more likely to leave than those who did. This suggests that the company's support mechanisms or benefits provided after an accident might inadvertently contribute to loyalty among affected employees, or that roles with higher accident rates naturally have lower turnover.

### Recommendations:

Based on these insights, the following strategic recommendations are proposed:

1.  **Proactive Satisfaction Management:**
    * Implement regular, short-pulse satisfaction surveys to monitor employee sentiment continually.
    * Develop targeted retention programs for employees exhibiting low satisfaction.
    * Conduct "exit interviews" and "stay interviews" with high-performing, high-satisfaction employees who depart, to understand their diverse motivations and address potential gaps in career growth or recognition.

2.  **Targeted Tenure Engagement Programs:**
    * Introduce specific career development workshops, mentorship programs, or retention bonuses for employees nearing or within their 3-6 year tenure.
    * Encourage managers to have proactive career path discussions with employees at these critical milestones.

3.  **Optimize Workload Management:**
    * Implement workload assessment tools to identify and address instances of both employee under-utilization and overwork.
    * Promote flexible work arrangements and discourage excessive overtime to foster a healthy work-life balance.
    * For underloaded employees, assign challenging new projects or cross-functional training to enhance engagement and skill development.

4.  **Refine Performance Management and Growth Opportunities:**
    * Ensure that performance reviews are not only about past performance but also future growth and development within the company.
    * Create clear pathways for advancement and provide challenging assignments for high-performing employees to keep them engaged and prevent stagnation.

5.  **Re-evaluate Compensation and Promotion Structures:**
    * Conduct a thorough review of salary bands, especially for 'low' and 'medium' tiers, to ensure competitiveness within the industry.
    * Increase the visibility and fairness of promotion criteria and processes. Consider more frequent recognition programs.

6.  **Department-Specific Retention Audits:**
    * Initiate focused HR audits within the 'sales', 'technical', and 'support' departments to uncover specific pain points, challenges, or leadership issues contributing to higher attrition.
    * Develop tailored solutions and support systems for each department.

7.  **Investigate Work Accident Retention Effect:**
    * Analyze the specific support systems, compensation, or cultural aspects associated with employees who have experienced work accidents to understand what contributes to their higher retention. Explore if these positive elements can be generalized across the workforce.

By proactively addressing these key drivers, the company can significantly reduce employee turnover, foster a more engaged workforce, and realize long-term benefits in productivity and organizational stability.
