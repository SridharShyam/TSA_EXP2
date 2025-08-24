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

resampled_data = data['money'].resample('Y').sum().to_frame()
resampled_data.index = resampled_data.index.year
resampled_data.reset_index(inplace=True)
resampled_data.rename(columns={'date': 'Year'}, inplace=True)

years = resampled_data['Year'].tolist()
money = resampled_data['money'].tolist()
X = [i - years[len(years) // 2] for i in years]
x2 = [i ** 2 for i in X]
xy = [i * j for i, j in zip(X, money)]
n = len(years)

b = (n * sum(xy) - sum(money) * sum(X)) / (n * sum(x2) - (sum(X) ** 2))
a = (sum(money) - b * sum(X)) / n
linear_trend = [a + b * X[i] for i in range(n)]

x2 = [i ** 2 for i in X]
x3 = [i ** 3 for i in X]
x4 = [i ** 4 for i in X]
xy = [i * j for i, j in zip(X, money)]
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
resampled_data.set_index('Year', inplace=True)
```

A - LINEAR TREND ESTIMATION
```
plt.figure(figsize=(8, 6))
plt.plot(resampled_data.index, resampled_data['money'], 'o-b', label='Actual Data')   # blue dots + line
plt.plot(resampled_data.index, resampled_data['Linear Trend'], 'k--', label='Linear Trend')  # black dashed line
plt.xlabel('Year')
plt.ylabel('Total Money')
plt.title('Linear Trend Estimation (Degree 1) plot')
plt.legend()
plt.show()
```
B- POLYNOMIAL TREND ESTIMATION
```
plt.figure(figsize=(8, 6))
plt.plot(resampled_data.index, resampled_data['money'], 'o-b', label='Actual Data')   # blue dots + line
plt.plot(resampled_data.index, resampled_data['Linear Trend'], 'k--', label='Linear Trend')  # black dashed line
plt.xlabel('Year')
plt.ylabel('Total Money')
plt.title('Linear Trend Estimation (Degree 1) plot')
plt.legend()
plt.show()
```
### OUTPUT
A - LINEAR TREND ESTIMATION
<img width="713" height="547" alt="image" src="https://github.com/user-attachments/assets/8429f134-6ed8-447f-8ddd-c9630c4da868" />

B- POLYNOMIAL TREND ESTIMATION
<img width="713" height="547" alt="image" src="https://github.com/user-attachments/assets/43ed0d28-95de-4ef0-a71d-7dfea3119922" />

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
