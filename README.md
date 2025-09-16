# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## Program:
```
step-1:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

step-2:
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks_Scored":  [35, 40, 50, 55, 60, 65, 70, 80, 85, 95]
}
df = pd.DataFrame(data)

# Display dataset
print("Dataset:\n", df.head())
df
```

<img width="377" height="533" alt="image" src="https://github.com/user-attachments/assets/f5d2cc36-f32f-4638-9007-1b7b3c41a6ad" />

```
step-3:
X = df[["Hours_Studied"]]   # Independent variable
y = df["Marks_Scored"]      # Dependent variable

step-4:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
step-5:
model = LinearRegression()
model.fit(X_train, y_train)

step-6:
y_pred = model.predict(X_test)

step-7:
print("\nModel Parameters:")
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])

step-8:
print("\nEvaluation Metrics:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
```

<img width="923" height="158" alt="image" src="https://github.com/user-attachments/assets/b909e8b5-ccbf-48d7-b007-f5bbcda6d3fa" />

```

step-9:
plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', linewidth=2, label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression: Predicting Marks")
plt.legend()
plt.grid(True)
plt.show()

```

## Output:
<img width="952" height="677" alt="image" src="https://github.com/user-attachments/assets/cb931ca4-f7ce-4f5f-8ef9-dee18dc0dbcc" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
