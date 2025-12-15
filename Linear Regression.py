# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np

# %%
df = pd.read_csv("C:\\Users\\Lenovo\\Downloads\\Amazon (1).csv")

# %%
df.head()

# %%
df.info()

# %%
df.describe()

# %%
df.isna().sum()

# %%
data_filter = df[['Category', 'Brand', 'Country', 'Quantity', 'UnitPrice', 'Discount', 'Tax', 'ShippingCost', 'TotalAmount']]
data_filter

# %%
t_stat, p_value = stats.ttest_ind(data_filter['TotalAmount'], data_filter['UnitPrice'])
print(f'p-value: {p_value}')

# %%
t_stat, p_value = stats.ttest_ind(data_filter['TotalAmount'],  data_filter['Quantity'])
print(f'p-value: {p_value}')

# %%
t_stat, p_value = stats.ttest_ind(data_filter['TotalAmount'], data_filter['Discount'])
print(f'p-value: {p_value}')

# %%
t_stat, p_value = stats.ttest_ind(data_filter['TotalAmount'], data_filter['Tax'])
print(f'p-value: {p_value}')

# %%
t_stat, p_value = stats.ttest_ind(data_filter['TotalAmount'], data_filter['ShippingCost'])
print(f'p-value: {p_value}')

# %%
features = ['UnitPrice', 'Quantity', 'Discount', 'Tax', 'ShippingCost']
target = 'TotalAmount'

df_model = data_filter[features + [target]].copy()

# %%
df_model = pd.get_dummies(df_model, drop_first=True)
df_model

# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X = df_model[features]
y = df_model[target]
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# %%
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = np.mean(np.abs(y_test - y_pred))

print(f'R-squared: {r2}')
print(f'rmse: {rmse}')
print(f'mae: {mae}')


