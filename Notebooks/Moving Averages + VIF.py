#!/usr/bin/env python
# coding: utf-8

# In[1]:


##Libraries
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# In[2]:


#Import data
file_path = 'C:/Users/Maya/Documents/Maya/SummerSchool/Case/slim_files/teva_sales_small_normalized_slim.xlsx'
df = pd.read_excel(file_path)
#print(df)


# In[3]:


# Group by 'inn_normalized' and find the 'morion_id' with the highest 'sm_sales' within each group
max_sales_per_inn = df.loc[df.groupby('inn_normalized')['sm_sales'].idxmax()]

# Create a list of morion_id with highest sm_sales for each inn_normalized
morion_ids = max_sales_per_inn['morion_id'].tolist()

print(morion_ids)


# In[96]:


#Import data
file_path = 'C:/Users/Maya/Documents/Maya/SummerSchool/Case/112504_teva_sales_small_diff_pacf_lags.xlsx'
data = pd.read_excel(file_path)
data.head()


# In[97]:


#Simple Moving Average

# Find columns that match the Morion IDs
matching_columns = []
for morion_id in morion_ids:
    for col in data.columns:
        if f"morion_{morion_id}" in col:
            matching_columns.append(col)

# Define the moving average windows
windows = [3]

# Calculate moving averages for each matching column and each window size
for column_name in matching_columns:
    for window_size in windows:
        moving_average_column = f'{column_name}_MA_{window_size}'
        data[moving_average_column] = data[column_name].rolling(window=window_size).mean()

           # Fill the first two values with zeros
    data[moving_average_column].iloc[:2] = 0
    
# Display the first few rows of the dataframe to verify the moving averages
data.head()


# In[98]:


#Cumulative Moving Average

# Find columns that match the Morion IDs and exclude columns created for simple moving averages
matching_columns = []
for morion_id in morion_ids:
    for col in data.columns:
        if f"morion_{morion_id}" in col and "MA_" not in col:
            matching_columns.append(col)

# Calculate cumulative moving averages for each matching column
for column_name in matching_columns:
    if f'{column_name}_CMA' not in data.columns:
        moving_averages = []
        cum_sum = 0
        count = 0

        # Loop through the column elements
        for i in range(len(data[column_name])):
            count += 1
            cum_sum += data[column_name][i]
            window_average = cum_sum / count
            moving_averages.append(window_average)

        # Add the cumulative moving average to the DataFrame
        cumulative_average_column = f'{column_name}_CMA'
        data[cumulative_average_column] = moving_averages

# Display the first few rows of the dataframe to verify the moving averages
print(data.head())


# In[48]:


data.shape


# In[99]:


#Exponential Moving Average

# Define the smoothing factor for EMA
x = 0.5  # You can adjust this value as needed

# Calculate Exponential Moving Averages for each matching column
for column_name in matching_columns:
    if f'{column_name}_EMA' not in data.columns:
        moving_averages = []
        # Initialize the first value of EMA with the first value of the column
        moving_averages.append(data[column_name].iloc[0])

        # Loop through the column elements to calculate EMA
        for i in range(1, len(data[column_name])):
            window_average = round((x * data[column_name].iloc[i]) + ((1 - x) * moving_averages[-1]), 2)
            moving_averages.append(window_average)

        # Add the EMA to the DataFrame
        ema_column = f'{column_name}_EMA'
        data[ema_column] = moving_averages

# Display the first few rows of the dataframe to verify the moving averages
print(data.head())


# In[101]:


# Identify columns for morion_id_diff
diff_columns = [col for col in data.columns if '_diff' in col]

# List of other factors
other_factors = [
    'atc_code_5_normalized', 'brand_normalized', 'inn_normalized', 'nfc_1_normalized',
    'strength_unit_normalized', 'drug_size_unit_normalized', 'form_normalized',
    'strength_amount_normalized', 'pack_size_normalized', 'drug_size_amount_normalized'
]

# Columns that should not be removed
non_removable_columns = ['month', 'ar_lag_1', 'ar_lag_2']

# Function to calculate VIF and remove columns with VIF above 10
def calculate_vif(data, threshold=10.0):
    dropped = True
    while dropped:
        dropped = False
        vif = pd.DataFrame()
        vif["Variable"] = data.columns
        vif["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
        max_vif = vif["VIF"].max()
        if max_vif > threshold:
            dropped = True
            drop_column = vif.sort_values('VIF', ascending=False)["Variable"].iloc[0]
            print(f"Dropping {drop_column} with VIF {max_vif}")
            data = data.drop(columns=[drop_column])
    return data, vif

# Create a DataFrame for VIF calculation including all morion_id_diff and other factors
vif_columns = diff_columns + other_factors
vif_data = data[vif_columns].copy()

# Calculate VIF and remove high VIF columns
filtered_vif_data, final_vif = calculate_vif(vif_data)

# Add back the non-removable columns
final_columns = non_removable_columns + list(filtered_vif_data.columns)
filtered_vif_data = data[final_columns]

# Display the final data and VIF results
print(filtered_vif_data.head())
print(final_vif)


# In[102]:


filtered_vif_data.shape


# In[108]:


#Save as excel
output_file_path = 'C:/Users/Maya/Documents/Maya/SummerSchool/Case/FINAL_112504_teva_sales_small_diff_pacf_lags.xlsx'
filtered_vif_data.to_excel(output_file_path, index=False)


# In[109]:


#Save as csv
output_file_path = 'C:/Users/Maya/Documents/Maya/SummerSchool/Case/FINAL_112504_teva_sales_small_diff_pacf_lags.csv'
filtered_vif_data.to_csv(output_file_path, index=False)

