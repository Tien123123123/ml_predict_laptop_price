Project: Laptop Price Prediction
- Project Objective
  -  The goal of this project is to build a predictive model that estimates the prices of laptops based on various features. Using a dataset of laptop specifications, I aim to train a Random Forest Regressor to provide accurate price predictions.
- Dataset
  - The dataset used in this project is titled laptop_price - dataset.csv. It contains various features related to laptop specifications, including but not limited to:
1. Screen size
2. CPU frequency
3. RAM
4. Storage types and sizes (SSD, HDD, Flash)
5. GPU type
6. Price (target variable)
----------------------------------
1. Data Preprocessing
- Memory Parsing: The split_memories function extracts and converts storage capacities from the Memory field into separate columns for SSD, HDD, and Flash storage.
- Screen Resolution Parsing: The split_screen function extracts screen width and height from the ScreenResolution field.
- Feature Selection: Unnecessary columns such as TypeName, OpSys, and Product were removed to focus on relevant features.
2. Model Training: The project utilizes a RandomForestRegressor from sklearn for predicting laptop prices. The model training process includes:
  - Splitting the dataset into training and testing sets (80% training, 20% testing).
  - Preprocessing numeric and categorical features using StandardScaler and OneHotEncoder.
  - Fitting the Random Forest model to the training data.
3. Model Evaluation: The performance of the model is evaluated using the following metrics:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - R² Score  
4. Results Visualization: A comparison between actual and predicted prices is visualized using matplotlib. The graph displays:
Actual prices in red
Predicted prices in blue (dashed line)
5. Model Saving: The trained model is saved using pickle for future use.

Result:
- After Training and Validating:
  - ![image](https://github.com/user-attachments/assets/e5786a1f-6542-4fd1-813c-0d44bf4b98ed)

  - ![image](https://github.com/user-attachments/assets/4a8345e9-ee7d-4202-8922-1e6cb7f1adf3)

- After Predict:
![image](https://github.com/user-attachments/assets/d13a23b3-f5ee-4389-8a4e-7c384c0a23cb)

