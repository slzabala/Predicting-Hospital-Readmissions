# Patient Readmission Prediction & Analysis
## Project 4

## Project Overview
This project uses machine learning and data visualization techniques to address the critical issue of hospital readmissions. Our goal was to build a predictive model that identifies patients at high risk of being readmitted within 30 days of discharge. With this, our goal was to help high risk regions and hospitals prioritize interventions, improve patient outcomes, and potentially reduce readmission rates and associated cost.

Also, along with our machine learning model, we perform a geographical analysis of hospital readmission data to identify regional variations and facilities with high or low readmission rates. This analysis provides insights for targeted interventions and resource allocation.

## Objectives: 
### This project will focus on the following goals. 
    1. Predict Patient Readmission 
    2. Data Driven Insights
    3. Geographical Analysis
    4. Model Optimization
    5. Actionable Insights

# Data Sources:
### Centers for Medicare & Medicaid Services (CMS)
    o Patient Data: Resources/patient_data.csv
    o Used for building and tuning the machine learning model.
    o The Hospital Readmission Reduction Program public dataset provides readmission rates across hospitals nationwide. 
    o https://data.cms.gov/provider-data/dataset/9n3s-kdb3#data-table
### Hugging Face
    o Hospital Readmissions Data: Resources/FY_2025_Hospital_Readmissions_Reduction_Program_Hospital.csv
    o Used for data cleaning, feature engineering, and geographical visualizations.
    o Public Readmission dataset.
    o https://huggingface.co/datasets/aai540-group3/diabetes-readmission
    
## Research Questions: 
    1. What variables has the strongest correlation with readmission risk?
    2. Does the length of stay influence the probability of readmission?
    3. How does hospital volume, as measured by the number of admissions and discharges, affect patient readmission rates?
    4. How do readmission rates vary across regions that could benefit from a predictive model to reduce readmissions in highest readmission rates areas?
    5. Which facilities have the highest readmission rates, where a readmission prediction model might have the greatest reduction in readmission rates and healthcare costs?
    
## Tools and Technologies:
    • Programing Languages: Python:
    • Data Cleaning: Pandas, Numpy
    • Data Processing & Metrics: Scikit-learn
    • Deep Learning: TensorFlow & Keras
    • Hyperparameter Tuning: Keras Tuner
    • File Handling: Pathlib
    • Data Visualization: Tableau
    
## Files:
### Hospital Level Analysis Folder:
• Resources Folder
    o FY_2025_Hospital_Readmissions_Reduction_Program_Hospital.csv: Original hospital-level readmission data.
• hospital_data_cleaning.ipynb: Notebook used to clean and transform the above CSV into hospital_data_cleaning.ipynb.
• hospital_clean.csv: The cleaned dataset ready for visualization and further analysis.
• Tableau Visualizations: hosp-vizs.twb file illustrating hospital-level analysis.
• Other Working Files: Various notebooks/scripts used for exploratory work and early model optimization not part of the final ML model.
### Neural Networks Model Folder:
    • Resources Folder
        o patient_data.csv: The patient data used to build the neural network model.
    • nn_model_analysis.ipynb: Main notebook for creating, tuning, and evaluating the neural network model.
### Project Proposal
        • Patient Readmission Prediction Project Proposal.docx: Outlines the project scope, objectives, and initial plan.
### Final Presentation
    • Final Presentation Patient Readmission.pptx: The polished presentation summarizing the project’s findings, actionable insights, and conclusions.
    
### Code Overview
### 1. Machine Learning Model
    This script performs the following steps:
    • Data Loading & Exploration:
        o Loads the patient data and previews it.
        o Calculates the percentage of patients readmitted to understand the target distribution.
    • Feature Selection:
        o Computes the correlation matrix with respect to the target (readmitted).
        o Filters features using a correlation threshold (0.05), retaining only those with meaningful correlation.
    • Data Preprocessing:
        o Splits the data into training and testing sets.
        o Applies StandardScaler to normalize the feature values.
    • Model Building & Hyperparameter Tuning:
        o Defines a function (create_model) that builds a TensorFlow Sequential neural network.
        o The architecture (number of layers, units per layer, and activation functions) is tuned using Keras Tuner (Hyperband) based on validation accuracy.
        o The model is compiled with binary cross-entropy loss, the Adam optimizer, and metrics including accuracy and precision.
    • Training & Evaluation:
        o Runs the hyperparameter search, retrieves the best model, and further trains it for 100 epochs.
        o Evaluates the final model on the test set.
        o Saves the final model to NN_2.h5.
## 2. Hospital Data Cleaning and Visualization
    This script performs the following steps:
    • Data Loading:
        o Loads the hospital readmissions dataset from a CSV file.
    • Data Cleaning & Feature Selection:
        o Selects a subset of relevant columns such as Facility Name, Facility ID, State, Number of Discharges, Excess Readmission Ratio, Predicted and Expected Readmission Rates, and Number of Readmissions.
    • Geographical Feature Engineering:
        o Defines lists of states for each census division and region.
        o Engineered functions (get_division and get_region) to map each state to its respective division and region.
        o Applies these functions to create new columns for Division and Region.
        o Drops rows with null values to ensure clean data.
    • Data Export:
        o Saves the cleaned and transformed dataset to Resources/hospital_clean.csv for further analysis and visualization.
    
## Installation and Usage
    1. Clone the Repository:
        • git clone https://github.com/slzabala/Predicting-Hospital-Readmissions/tree/main
    2. Set Up the Environment: 
        • Make sure you have Python 3.10+ installed, then run:
        • pip install -r requirements.txt
          o This should install the required project’s dependencies like pandas, numpy, scikit-learn, tensorflow, keras-tuner, etc..
    3. Run the Machine Learning Model Script:
        • nn_model_analysis.ipynb 
    This script will:
        • Load and preprocess the patient data.
        • Run hyperparameter tuning using Keras Tuner.
        • Train, evaluate, and save the final neural network model.
    4. Run the Hospital Data Cleaning and Visualization Script:
        • python hospital_data_cleaning.py
    This script will:
        • Clean the hospital dataset.
        • Transform the data with geographical features.
        • Export the cleaned data to Resources/hospital_clean.csv.
        
## Project Takeaways
    • Results and Impact: 
        o Current model accuracy is ~62%, indicating that while patterns are detected, further optimization is needed.
        o The visualizations highlight regional disparities in readmission rates, suggesting areas for targeted interventions.
    • Future Research and Optimization Suggestions: 
        o Add more relevant features, increase data size or explore with advanced feature engineering to capture interactions between variables.
        o Explore deeper and more complex neural networks and try more optimizers and different learning rates. 
        o Evaluate the model using precision, recall, and F1-score to ensure that the model reliably identifies patients with high readmission risk. 