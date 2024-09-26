Project: Laptop Price Prediction
- Project Objective
  -  The goal of this project is to build a predictive model that estimates the prices of laptops based on various features. Using a dataset of laptop specifications, My Goal is deploy Regressor model to provide accurate price predictions.
- Dataset
  - The dataset used in this project is titled laptop_price - dataset.csv. It contains various features related to laptop specifications, including but not limited to:
1. Screen size
2. CPU frequency
3. RAM
4. Storage types and sizes (SSD, HDD, Flash)
5. GPU type
6. Price (target variable)
----------------------------------
1. Data Argumentation
- Memory Parsing: The split_memories function extracts and converts storage capacities from the Memory field into separate columns for SSD, HDD, and Flash storage.
- Screen Resolution Parsing: The split_screen function extracts screen width and height from the ScreenResolution field.
- Feature Selection: Unnecessary columns such as TypeName, OpSys and Product were removed to focus on relevant features.
2. Model Training: The project utilizes a RandomForestRegressor from sklearn for predicting laptop prices. The model training process includes:
  - Splitting the dataset into training and testing sets (80% training, 20% testing).
  - Deploying Pipeline and Preprocessing numeric and categorical features using StandardScaler and OneHotEncoder.
  - Fitting the Random Forest model with Select K best to the training data.
3. Model Evaluation: The performance of the model is evaluated using the following metrics:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - R² Score  
4. Results Visualization: A comparison between actual and predicted prices is visualized using matplotlib. The graph displays:
  - Actual prices in red
  - Predicted prices in blue (dashed line)
5. Model Saving: The trained model is saved using pickle for future use.

Result:
- After Training and Validating:
  - ![image](https://github.com/user-attachments/assets/7c65959e-b03d-41a9-aa2d-42d8a7058f96)

  - ![image](https://github.com/user-attachments/assets/70bf4435-ce8a-4e75-8e43-9a281fe4bf9d)


- After Predict on Postman:
![image](https://github.com/user-attachments/assets/a2b58422-a196-48f4-afcc-926a9112eb75)


