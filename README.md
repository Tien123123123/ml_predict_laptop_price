Project: Laptop Price Prediction
- Project Objective
  -  The goal of this project is to build a predictive model that estimates the prices of laptops based on various features. Using a dataset of laptop specifications, we aim to train a Random Forest Regressor to provide accurate price predictions.
- Dataset
  - The dataset used in this project is titled laptop_price - dataset.csv. It contains various features related to laptop specifications, including but not limited to:
Screen size
CPU frequency
RAM
Storage types and sizes (SSD, HDD, Flash)
Weight
GPU type
Price (target variable)
Key Components
1. Data Preprocessing
The following preprocessing steps were implemented:

Memory Parsing: The split_memories function extracts and converts storage capacities from the Memory field into separate columns for SSD, HDD, and Flash storage.
Screen Resolution Parsing: The split_screen function extracts screen width and height from the ScreenResolution field.
Feature Selection: Unnecessary columns such as TypeName, OpSys, and Product were removed to focus on relevant features.
2. Model Training
The project utilizes a RandomForestRegressor from sklearn for predicting laptop prices. The model training process includes:

Splitting the dataset into training and testing sets (80% training, 20% testing).
Preprocessing numeric and categorical features using StandardScaler and OneHotEncoder.
Fitting the Random Forest model to the training data.
3. Model Evaluation
The performance of the model is evaluated using the following metrics:

Mean Squared Error (MSE)
Mean Absolute Error (MAE)
R² Score
4. Results Visualization
A comparison between actual and predicted prices is visualized using matplotlib. The graph displays:

Actual prices in red
Predicted prices in blue (dashed line)
5. Model Saving
The trained model is saved using pickle for future use.



Result:
![image](https://github.com/user-attachments/assets/3e639289-61ac-4af0-a657-1630ef731b0e)
