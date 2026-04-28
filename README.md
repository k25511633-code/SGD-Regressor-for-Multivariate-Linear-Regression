# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and scale the dataset features (Size, Bedrooms) with StandardScaler.
2.Train two SGDRegressor models separately for Price and Occupants
3.Accept new house size and bedroom inputs, then scale them.
4.Predict Price and Occupants using the trained models and display results.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
## Developed by: KAVIYA R
## RegisterNumber:  212225040179
*/
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("house.csv")

data.columns = data.columns.str.strip()

X = data[['Size', 'Bedrooms']]
y_price = data['Price']
y_occ = data['Occupants']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

price_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)
occ_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)
price_model.fit(X_scaled, y_price)
occ_model.fit(X_scaled, y_occ)

size = float(input("Enter house size: "))
bed = int(input("Enter number of bedrooms: "))

new_data = scaler.transform([[size, bed]])

pred_price = price_model.predict(new_data)
pred_occ = occ_model.predict(new_data)

print("Predicted Price:", pred_price[0])
print("Predicted Occupants:", round(pred_occ[0]))
```

## Output:


<img width="802" height="136" alt="Screenshot 2026-04-28 144701" src="https://github.com/user-attachments/assets/6cb669f2-7659-4b4f-842c-808c2ed64762" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
