# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date: 19/08/2025
Name: SHYAM S
Reg.No: 212223240156
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('/content/index_1.csv', parse_dates=[0])
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

data.head()
data.tail()

resampled_data = data['money'].resample('M').sum().to_frame()
resampled_data.reset_index(inplace=True)
resampled_data.rename(columns={'date': 'Month'

months = list(range(len(resampled_data)))
money = resampled_data['money'].tolist()
n = len(months)

X = [i - months[len(months) // 2] for i in months]
x2 = [i ** 2 for i in X]
xy = [i * j for i, j in zip(X, money)]

# Linear trend calculation
b = (n * sum(xy) - sum(money) * sum(X)) / (n * sum(x2) - (sum(X) ** 2))
a = (sum(money) - b * sum(X)) / n
linear_trend = [a + b * X[i] for i in range(n)]

# Polynomial trend calculation
x3 = [i ** 3 for i in X]
x4 = [i ** 4 for i in X]
x2y = [i * j for i, j in zip(x2, money)]

coeff = [[len(X), sum(X), sum(x2)],
         [sum(X), sum(x2), sum(x3)],
         [sum(x2), sum(x3), sum(x4)]]
Y = [sum(money), sum(xy), sum(x2y)]

A = np.array(coeff)
B = np.array(Y)

solution, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
a_poly, b_poly, c_poly = solution
poly_trend = [a_poly + b_poly * X[i] + c_poly * (X[i] ** 2) for i in range(len(X))]

print(f"Linear Trend: y={a:.2f} + {b:.2f}x")
print(f"\nPolynomial Trend: y={a_poly:.2f} + {b_poly:.2f}x + {c_poly:.2f}x²")

resampled_data['Linear Trend'] = linear_trend
resampled_data['Polynomial Trend'] = poly_trend
```

A - LINEAR TREND ESTIMATION
```
# Linear trend plot
plt.figure(figsize=(8, 6))
plt.plot(resampled_data['Month'], resampled_data['money'], 'bo-', label='Actual Sales', markersize=4)
plt.plot(resampled_data['Month'], resampled_data['Linear Trend'], 'r--', linewidth=2, label='Linear Trend')
plt.xlabel('Month')
plt.ylabel('Monthly Sales')
plt.title('Linear Trend Analysis')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
```
B- POLYNOMIAL TREND ESTIMATION
```
# Polynomial trend plot
plt.figure(figsize=(8, 6))
plt.plot(resampled_data.index, resampled_data['money'], 'o-b', label='Actual Data')   # blue dots + line
plt.plot(resampled_data.index, resampled_data['Polynomial Trend'], 'k--', label='Polynomial Trend (Degree 2)')  # black dashed line
plt.xlabel('Year')
plt.ylabel('Total Money')
plt.title('Polynomial Trend Estimation plot')
plt.legend()
plt.show()
```
### OUTPUT
A - LINEAR TREND ESTIMATION
<img width="713" height="584" alt="image" src="https://github.com/user-attachments/assets/e3a0384e-a954-4539-89a5-66766b7803f8" />

B- POLYNOMIAL TREND ESTIMATION
<img width="713" height="547" alt="image" src="https://github.com/user-attachments/assets/ee21ad60-c8d4-40cf-b10f-282b7635bebe" />


### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
