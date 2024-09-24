import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
from lazypredict.Supervised import LazyRegressor

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

root = "laptop_price - dataset.csv"
df = pd.read_csv(root)
drop_features = ["TypeName", "OpSys", "CPU_Company", "Memory", "ScreenResolution", "GPU_Company", "Product"]
df[["SSD_GB", "HDD_GB", "Flash Storage_GB"]] = df["Memory"].apply(lambda x:split_memories(x)).apply(pd.Series)
df[["Screen_Width", "Screen_Height"]] = df["ScreenResolution"].apply(lambda x:split_screen(x)).apply(pd.Series)
df = df.drop(drop_features, axis=1)

target = "Price (Euro)"
x = df.drop(target, axis=1)
y = df[target]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=43)

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

param = {
    # 'n_estimators': [50, 100, 150, 200],  # Số lượng cây
    # 'max_depth': [None, 10, 15, 20, 25],  # Độ sâu của cây
    # 'min_samples_split': [2, 5, 10],  # Số mẫu tối thiểu để chia nút
    # 'min_samples_leaf': [1, 2, 4, 6],  # Số mẫu tối thiểu ở mỗi lá
    # 'max_features': ['auto', 'sqrt', 'log2']  # Số đặc trưng được xem xét để chia
}
model_reg = Pipeline(steps=[
    ("processing", preprocessing_data),
    # ("model", GridSearchCV(RandomForestRegressor(random_state=123), param_grid=param, verbose=2, cv=6,scoring='r2'))
    ("model", RandomForestRegressor(random_state=123))
])
model_reg.fit(x_train, y_train)
y_pred = model_reg.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"MSE: {mse}, MAE: {mae}")
print(f"R2 score: {r2_score(y_test, y_pred)}")

# Save the model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model_reg, file)

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
