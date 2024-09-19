import pickle
import pandas as pd

with open('model_filename.pkl', 'rb') as file:
    model = pickle.load(file)

x_new_data = pd.DataFrame({
    "Inches": [15.6],
    "Screen_Width": [1920],
    "Screen_Height": [1080],
    "CPU_Frequency (GHz)": [2.6],
    "RAM (GB)": [8],
    "SSD_GB": [512],
    "HDD_GB": [0],
    "Flash Storage_GB": [0],
    "Weight (kg)": [2.1],
    "CPU_Type": ["Intel Core i5"],
    "GPU_Type": ["Intel UHD Graphics 620"]
})

y_new_pred = model.predict(x_new_data)

# Output the prediction
print("Predicted Price:", y_new_pred)
