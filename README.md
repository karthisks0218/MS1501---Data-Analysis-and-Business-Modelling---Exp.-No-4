# Experiment 4: Multiple Regression Analysis (Fish Dataset)

## Aim
To perform Multiple Linear Regression on a dataset imported from a CSV file to analyze how multiple features together predict a dependent variable.

## Dataset
The dataset used for this experiment is `fish.csv` which contains real-world physical measurements of various fish species:
* **Independent Variables ($X$):**
  * `L`: Mean length derived from `Length1`, `Length2`, and `Length3`
  * `Height`: Vertical height of the fish
  * `Width`: Diagonal width of the fish
* **Dependent Variable ($Y$):**
  * `Weight`: Weight of the fish (in grams)

## Algorithm / Steps Followed
1. Import the required Python libraries (`numpy`, `pandas`, `matplotlib`, `scikit-learn`).
2. Load the fish dataset into a Pandas DataFrame using `pd.read_csv()`.
3. Display the dataset and inspect its columns.
4. Define the dependent ($Y$) and independent ($X$) variables.
5. Train the Multiple Linear Regression model using `LinearRegression()`.
6. Display the model coefficients (`model.coef_`), intercept (`model.intercept_`), and regression equation.
7. Predict target variable (`Weight`) values using `model.predict()`.
8. Evaluate model accuracy using the $R^2$ score (`r2_score()`).
9. Plot actual vs. predicted values for visual comparison.

## Functions and Methods Used

| Category | Function / Method | Purpose / Description |
| :--- | :--- | :--- |
| **Data Creation / Handling** | `pd.read_csv()` | Load the CSV dataset into a Pandas DataFrame |
| | `np.array()` / `pd.DataFrame()` | Create and manage numerical arrays and tabular data for regression |
| **Model Building (Scikit-learn)** | `LinearRegression()` | Initialize the Multiple Linear Regression model |
| | `model.fit()` | Train the regression model using multiple independent variables ($X$) and target ($Y$) |
| | `model.predict()` | Predict dependent variable values ($Y$) using the trained model |
| | `r2_score()` | Evaluate the accuracy of the regression model using the $R^2$ score |
| **Model Attributes** | `model.intercept_` | Retrieve the intercept (constant term) of the regression equation |
| | `model.coef_` | Retrieve the coefficients (slopes) for each independent variable |
| **Visualization (Matplotlib)** | `plt.plot()` | Plot actual and predicted weight values for comparison |
| | `plt.title()`, `plt.xlabel()`, `plt.ylabel()`, `plt.legend()`, `plt.show()` | Configure titles, axis labels, legend, and display the plotted figure |

## Program & Output

# Step 1: Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Step 2: Load Dataset
df = pd.read_csv('Fish_length.csv')

# Step 3: Display Dataset
print(df.head())
print(df.columns)

# Step 4: Select Independent (X) and Dependent (Y) Variables

# Calculate average of Length1, Length2 and Length3
L = df[['Length1', 'Length2', 'Length3']].mean(axis=1)

# Independent variables
X = pd.DataFrame({
    'L': L,
    'Height': df['Height'],
    'Width': df['Width']
})

# Dependent variable
Y = df['Weight']

# Step 5: Build and Train Multiple Regression Model
model = LinearRegression()
model.fit(X, Y)

# Step 6: Display Model Coefficients and Intercept
print("\nIntercept (a):", model.intercept_)
print("Coefficients (b1, b2, b3):", model.coef_)

# Step 7: Form Regression Equation
print(f"\nRegression Equation: Weight = {model.intercept_:.2f} + "
      f"{model.coef_[0]:.2f}×L + "
      f"{model.coef_[1]:.2f}×Height + "
      f"{model.coef_[2]:.2f}×Width")

# Step 8: Predict Values
Y_pred = model.predict(X)
print("\nPredicted Weight Values:")
print(Y_pred[:10])

# Step 9: Evaluate Model Performance
r2 = r2_score(Y, Y_pred)
print(f"\nModel Performance (R² Score): {r2:.4f}")

# Step 10: Visualization - Actual vs Predicted Weights
plt.figure(figsize=(8, 6))
plt.scatter(range(len(Y)), Y, color='blue',
            label='Actual Weight')
plt.plot(range(len(Y)), Y_pred, color='red',
         linewidth=2, label='Predicted Weight')

plt.title("Actual vs Predicted Fish Weights (Multiple Regression)")
plt.xlabel("Data Points")
plt.ylabel("Weight (grams)")
plt.legend()
plt.show()

# Output

<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/45f73bfd-6336-4562-9612-a14786ea4b18" />


## Result

The Multiple Linear Regression model was successfully implemented using the Fish dataset (fish.csv). The model was used to predict the weight of fish based on its physical characteristics, such as Length1, Length2, and Height.

The model achieved an R² score of 0.8808, indicating that it explains approximately 88.08% of the variation in fish weight.

Hence, the experiment was successfully completed, and the model demonstrated a strong relationship between fish measurements and their weight.
