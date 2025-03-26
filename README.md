# Predicting-Hospital-Readmissions
Project 4
 

Overview of the Analysis 

This analysis developed a deep learning model to predict patient readmission risk using electronic health record data. The goal was to create a tool that helps healthcare providers identify high-risk patients who may need additional care or monitoring after discharge. By preprocessing patient data, designing a neural network architecture, and optimizing model performance, we aimed to build an accurate predictive tool that can support clinical decision-making. 

 

Results 
Data Preprocessing 

Target Variable(s): 
The target variable is readmitted (1 = patient was readmitted, 0 = not readmitted) 

Feature Variable(s): 
All 151 columns from the original dataset except the target variable 

Key features include: 

Time in hospital 
Number of procedures and medications 
Diagnostic history 
Test results (A1C, glucose levels) 
Treatment history 

Variables Removed: 
None - all features were retained for analysis 
Note: The target variable was separated from features during processing 

 

Data Preprocessing Steps: 

Maintained all features as they showed predictive value 
Used stratified sampling to preserve the natural 46% readmission rate in train/test splits 
Scaled all features using StandardScaler for consistent numerical ranges 
Split data into 75% training and 25% testing sets 
Model Architecture and Training 
Neural Network Design: 

Input Layer: 151 neurons (matching the number of features) 

Hidden Layers: Dynamic architecture with 1-20 hidden layers (automatically optimized) 

Neuron counts ranging from 3-19 per layer (tuned during optimization) 

ReLU activation in most layers (automatically selected as optimal) 

Output Layer: 1 neuron with Sigmoid activation for binary classification 

Training Approach: 

Used Adam optimizer with binary crossentropy loss 
Implemented Hyperband algorithm for efficient hyperparameter tuning 
Trained for 50-100 epochs with validation monitoring 

Model Performance 

The optimized model achieved: 
Accuracy: 63.3% (initial evaluation) 
Loss: 0.644 

After full training: 
Accuracy: 62.8% 
Loss: 0.651 

Key Optimization Strategies: 

Automated hyperparameter tuning found optimal architecture (9 hidden layers) 
Stratified sampling maintained realistic class distribution 
Feature scaling ensured equal contribution from all variables 

Summary 

The deep learning model achieved consistent performance around 63% accuracy in predicting patient readmissions. While this exceeds a simple majority-class baseline (54%), there remains room for improvement. The automated tuning successfully identified an effective 9-layer architecture with varied neuron counts that balanced complexity and performance. 

 