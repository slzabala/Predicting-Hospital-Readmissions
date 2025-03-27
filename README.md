# Predicting-Hospital-Readmissions
**Project 4**
 

**Overview of the Analysis:**
The purpose of this analysis was to develop a deep learning model to predict whether a patient will be readmitted to a hospital based on various medical and demographic features. The goal was to:

Preprocess and clean patient data for model training.

Build and train a neural network model using TensorFlow/Keras.

Optimize the model using hyperparameter tuning with Keras Tuner.

Evaluate model performance on test data to determine accuracy and reliability.

The model was trained on a subset of 1,000 patient records, with hyperparameter tuning applied to improve performance.
 


**Data Preprocessing**
**Target Variable(s):** readmitted: Binary classification (whether a patient was readmitted or not).

**Feature Variable(s):** All other columns in the dataset except readmitted were used as features (150 features total).

**Variables Removed**
Only the target variable (readmitted) was explicitly removed from the input features.

Rows with null, negative, or duplicate values were filtered out to ensure data quality.

**2. Compiling, Training, and Evaluating the Model**

Initial Model Architecture

**Layers:**

**Input Layer**: 150 features → Dense (8 neurons, ReLU activation)

**Hidden Layers:**

Dense (10 neurons, tanh activation)

Dense (16 neurons, tanh activation)

**Output Layer**: 1 neuron (sigmoid activation for binary classification)

**Optimizer**: Adam

**Loss Function:** Binary cross-entropy

Epochs: 50

**Model Performance**

**Training Accuracy:** 96.69% (indicating severe overfitting)

**Test Accuracy:** 54.40% (poor generalization to unseen data)

**Loss:** 1.5654 (high, suggesting poor predictive capability)

**Hyperparameter Tuning (Keras Tuner)**

To improve performance, we used Keras Tuner to optimize:

Activation Functions (relu, tanh, sigmoid)

Number of Hidden Layers (1 to 6)

Neurons per Layer (1 to 10, in steps of 2)

**Best Model After Tuning:**

Activation: tanh

Layers: 6

Neurons per Layer: Varied (3 to 9)

Validation Accuracy: 65.20% (improvement over initial model)

Loss: 0.6460 (better than initial model)

**Steps Taken to Improve Performance**

Hyperparameter Tuning: Automated search for optimal layer sizes and activations.

Data Scaling: StandardScaler was applied to normalize features.

Stratified Splitting: Ensured balanced class distribution in train/test sets.

Early Stopping (Potential Future Improvement): Could prevent overfitting by halting training when validation accuracy plateaus.

**Summary** 

**Overall Model Performance**

The tuned model achieved 65.2% accuracy, a significant improvement over the initial model (54.4%).

However, the model still suffers from moderate overfitting (training accuracy was much higher than test accuracy).

The loss value (0.646) suggests room for improvement.
 
