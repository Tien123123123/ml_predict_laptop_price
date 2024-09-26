import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import pickle

def split_memories(loc):
    s = loc.split(" + ")
    ssd, hdd, flash = 0, 0, 0
    for part in s:
        num_list = part.split(" ")
        if 'GB' in num_list[0]:
            gb = float(num_list[0].replace('GB', ''))
        elif 'TB' in num_list[0]:
            gb = float(num_list[0].replace('TB', '')) * 1000

        if "SSD" in part:
            ssd += gb
        elif "HDD" in part:
            hdd += gb
        elif "Flash Storage" in part:
            flash += gb
    return ssd, hdd, flash
def split_screen(loc):
    s = loc.split(" ")
    number = [part.split("x") for part in s if "x" in part][0]
    return number[0], number[1]

root = "C:/Users/Dell/Documents/AI/Python/Datasets/laptop_price - dataset.csv"
df = pd.read_csv(root)
drop_features = ["TypeName", "OpSys", "CPU_Company", "Memory", "ScreenResolution", "GPU_Company", "Product"]
df[["SSD_GB", "HDD_GB", "Flash Storage_GB"]] = df["Memory"].apply(lambda x:split_memories(x)).apply(pd.Series)
df[["Screen_Width", "Screen_Height"]] = df["ScreenResolution"].apply(lambda x:split_screen(x)).apply(pd.Series)
df = df.drop(drop_features, axis=1)

target = "Price (Euro)"
x = df.drop(target, axis=1)
y = df[target]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=43)
# Data preprocessing
# Numeric: Inches, Screen_Width, Screen_Height, CPU_Frequency (GHz), RAM (GB), SSD_GB, HDD_GB, Flash Storage_GB, Weight (kg)
# Norminal: Product, CPU_Type, GPU_Type
# Ordinal:
# Boolen:

numeric_transform = Pipeline(steps=[
    ("scaler", StandardScaler())
])
norminal_transform = Pipeline(steps=[
    ("scaler", OneHotEncoder(handle_unknown="ignore"))
])

preprocessing_data = ColumnTransformer([
    ("numeric", numeric_transform, ["Inches", "Screen_Width", "Screen_Height", "CPU_Frequency (GHz)", "RAM (GB)", "SSD_GB", "HDD_GB", "Flash Storage_GB", "Weight (kg)"]),
    ("norminal", norminal_transform, ["CPU_Type", "GPU_Type"])
])
# Grid Search
param = {
    'n_estimators': [50, 100, 150, 200, 250, 300],
    'max_depth': [15, 20, 25, 30],
}
model_reg = Pipeline(steps=[
    ("processing", preprocessing_data),
    ("feature_selection", SelectKBest(score_func=f_regression, k=20)),
    ("model", GridSearchCV(RandomForestRegressor(random_state=123), param_grid=param, verbose=2, cv=6,scoring='r2'))
    # ("model", RandomForestRegressor(random_state=123))
])
model_reg.fit(x_train, y_train)

# Metrics and estimator
grid_search = model_reg.named_steps['model']
print("Best Estimator:", grid_search.best_estimator_)
print("Best Score:", grid_search.best_score_)
y_pred = model_reg.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse}, MAE: {mae}, R2 Score: {r2}")


# Save the model
with open('model_filename.pkl', 'wb') as file:
    pickle.dump(model_reg, file)

# Visualize Predictions
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(
    range(len(y_test)),
    y_test,
    label="Acutal Value",
    color="red"
)
ax.plot(
    range(len(y_pred)),
    y_pred,
    label="Predicted Value",
    color="blue",
    linestyle="--"
)
ax.set_xlabel("index")
ax.set_ylabel("price")
ax.legend()
ax.grid(True)
plt.show()
