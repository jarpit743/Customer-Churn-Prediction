
# Customer Churn Prediction Project

A machine learning project aimed at predicting customer churn using data analysis and classification models, followed by deploying using Flask.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [About Dataset](#dataset-overview)
4. [Requirements](#requirements)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [Model Building](#model-building)
7. [API Development](#api-development)
8. [Steps to Run the Project](#steps-to-run-the-project)
9. [Model Evaluation](#model-evaluation)
10. [API Usage](#api-usage)
11. [Future Enhancements](#future-enhancements)
12. [License](#license)

## Project Overview

Customer churn refers to when customers stop using a product or service. This project aims to predict which customers are likely to churn using historical data and machine learning techniques. The process involves three main stages:
1. **Exploratory Data Analysis (EDA):** Understanding and visualizing the data.
2. **Model Building:** Training machine learning models to make predictions.
3. **Deployment:** Creating a Flask API for serving churn predictions.

The project will help businesses take proactive actions to retain customers and minimize churn rates.

## Project Structure

1. **Churn Analysis - EDA.ipynb**: 
   - This notebook performs the exploratory data analysis (EDA) of the dataset, exploring patterns, visualizing relationships, and preparing data for modeling.
   
2. **Churn_Analysis_Model_Building.ipynb**: 
   - This notebook contains all the steps for building machine learning models to predict customer churn. This includes data preprocessing, splitting the data, training multiple models, evaluating their performance, and selecting the best model.
   
3. **app.py**: 
   - A Python Flask app that hosts a simple web API. The API accepts customer data as input and provides a prediction on whether the customer is likely to churn or not, using the trained machine learning model.

## About Dataset

### Context
"Predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs." - [IBM Sample Data Sets]

This dataset, available on Kaggle, helps businesses understand customer behavior and predict potential churners based on historical data. The goal is to analyze customer data and build strategies for retention.

**Dataset Link:** [Telco Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### Content
Each row in the dataset represents a customer, with columns providing details about:

- **Churn**: Whether the customer has left in the last month.
- **Services**: Phone, multiple lines, internet services (DSL, fiber optic), online security, backup, device protection, tech support, and streaming services.
- **Account Information**: Tenure, contract type, payment method, paperless billing, monthly and total charges.
- **Demographics**: Gender, age, partner, and dependent status.


## Requirements

To run this project, you will need the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Flask
- pickle (for saving/loading the trained model)

You can install these packages by running:
```bash
pip install -r requirements.txt
```

## Exploratory Data Analysis (EDA)

In the **Churn Analysis - EDA.ipynb**, the following steps are performed:
1. **Data Cleaning**: Handling missing values, incorrect data types, and outliers.
2. **Descriptive Statistics**: Getting an overall picture of the dataset through statistics like mean, median, mode, etc.
3. **Correlation Analysis**: Understanding the relationship between various features and customer churn using correlation matrices.
4. **Visualizations**: Plotting histograms, bar charts, heatmaps, and other visualizations to analyze patterns and trends in the data.

## Model Building

In the **Churn_Analysis_Model_Building.ipynb**, the model building process includes:
1. **Data Preprocessing**: Encoding categorical variables, scaling numerical features, and handling missing data.
2. **Model Training**: Different models like Logistic Regression, Random Forest, and SVM are trained on the processed data.
3. **Model Evaluation**: The models are evaluated based on accuracy, precision, recall, and F1-score to select the best-performing model.
4. **Model Export**: The best model is saved using the `pickle` module for future predictions.

## API Development

In **app.py**, a Flask API is developed:
- **/predict endpoint**: This endpoint accepts customer data as a JSON payload and returns a churn prediction (either `Churn` or `No Churn`).
  
   Example JSON input:
   ```json
   {
     "gender": "Female",
     "SeniorCitizen": 0,
     "Partner": "Yes",
     "Dependents": "No",
     "tenure": 12,
     "PhoneService": "Yes",
     "MultipleLines": "No",
     "InternetService": "DSL",
     "TotalCharges": 500.00
   }
   ```

## Steps to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd customer-churn-prediction
   ```

2. **Install Dependencies**:
   Install the required Python libraries by running:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the EDA and Model Building Notebooks**:
   Open the `Churn Analysis - EDA.ipynb` and `Churn_Analysis_Model_Building.ipynb` notebooks in Jupyter to follow and run the analysis and modeling steps.

4. **Run the Flask API**:
   Once the model is built and saved, start the Flask app:
   ```bash
   python app.py
   ```
   The app will be hosted at `http://localhost:5000`.

## Model Evaluation

The following models were evaluated:
1. **Logistic Regression**: A simple linear classifier.
2. **Random Forest**: An ensemble learning method using multiple decision trees.
3. **Support Vector Machine (SVM)**: A classifier that finds the hyperplane separating classes.

The evaluation metrics used are:
- **Accuracy**: The overall correctness of the model.
- **Precision & Recall**: Precision measures false positives, while recall measures false negatives.
- **F1-Score**: A weighted average of precision and recall.
  
The model with the best F1-score was saved and deployed in the Flask app.

## API Usage

To get predictions, send a POST request to the **/predict** endpoint with a JSON payload containing customer details.

Example:
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 45,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "TotalCharges": 1200.50
}'
```

The response will be either `Churn` or `No Churn`.

## Future Enhancements

- **Feature Engineering**: Introduce more complex features and interactions to improve model performance.
- **Hyperparameter Tuning**: Use grid search or random search for model optimization.
- **Web Interface**: Add a simple HTML front-end to allow users to input customer data and get predictions through a web form.
- **Cloud Deployment**: Deploy the Flask API on cloud platforms like AWS, Heroku, or Azure for public access.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
