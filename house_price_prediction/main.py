import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("data.csv")

X = data[["area"]]
y = data["price"]

model = LinearRegression()

model.fit(X, y)

area = float(input("Enter house area in square feet: "))

new_house = pd.DataFrame(
    [[area]],
    columns=["area"]
)

predicted_price = model.predict(new_house)

print(f"Predicted house price: {predicted_price[0]:.2f}")