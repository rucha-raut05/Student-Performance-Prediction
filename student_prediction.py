import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Create simple dataset
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Marks": [10, 20, 30, 40, 50, 60, 70, 80]
}

df = pd.DataFrame(data)

# Step 2: Split data
X = df[["Hours"]]
y = df["Marks"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Predict
predictions = model.predict(X_test)

# Step 5: Evaluation
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# Step 6: Visualization
plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks Prediction")
plt.show()