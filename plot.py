import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# ✅ Define Stock Data Folder
stock_folder = "/Users/colbywang/Google Drive/我的云端硬盘/Advanced NLP/Assignments/data files/organized/stock-data"

# ✅ List to Store 7-Day Percentage Changes
average_changes = []

# ✅ Loop Through Each Stock File
for stock_file in tqdm(os.listdir(stock_folder), desc="Processing Stock Data"):
    stock_path = os.path.join(stock_folder, stock_file)

    # ✅ Check if file is empty before reading
    if os.stat(stock_path).st_size == 0:
        continue

    # ✅ Read Stock Data
    stock_data = pd.read_csv(stock_path)

    # ✅ Check if required columns exist
    required_cols = {"Date", "Close", "10-K", "DEF 14A"}
    if not required_cols.issubset(stock_data.columns):
        continue

    # ✅ Find rows where "10-K" or "DEF 14A" is not "0"
    filing_rows = stock_data[(stock_data["10-K"] != "0") | (stock_data["DEF 14A"] != "0")].index

    # ✅ Compute 7-Day Stock Change
    for row in filing_rows:
        if row + 6 < len(stock_data):  # Ensure we have 7 days of data
            start_price = stock_data.iloc[row]["Close"]
            end_price = stock_data.iloc[row + 6]["Close"]

            # ✅ Compute Percentage Change
            percentage_change = ((end_price - start_price) / start_price) * 100
            average_changes.append(percentage_change)
        else:
            continue

import numpy as np
import scipy.stats as stats

# Convert average_changes to a NumPy array for calculations
average_changes = np.array(average_changes)

# Remove NaN values
average_changes = average_changes[~np.isnan(average_changes)]

# Compute descriptive statistics
mean_value = np.mean(average_changes)
median_value = np.median(average_changes)
q1 = np.percentile(average_changes, 25)
q3 = np.percentile(average_changes, 75)
iqr = q3 - q1  # Interquartile Range (IQR)
std_dev = np.std(average_changes, ddof=1)  # Sample standard deviation
coefficient_variation = std_dev / mean_value if mean_value != 0 else np.nan
skewness = stats.skew(average_changes)
kurtosis = stats.kurtosis(average_changes)

# Print descriptive statistics
print(f"Mean: {mean_value:.2f}")
print(f"Median: {median_value:.2f}")
print(f"Q1: {q1:.2f}")
print(f"Q3: {q3:.2f}")
print(f"IQR: {iqr:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")
print(f"Coefficient of Variation: {coefficient_variation:.2f}")
print(f"Skewness: {skewness:.2f}")
print(f"Kurtosis: {kurtosis:.2f}")

# ✅ Plot Box Plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.boxplot(average_changes, vert=False)
ax.set_title("Box Plot of 7-Day Stock Price Change After Filing")
ax.set_xlabel("Stock Price Change (%)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("boxplot-7-day-average-stock-price-change.png")
plt.show()

# ✅ Plot Histogram (Distribution)
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(average_changes, bins=50, alpha=0.75, edgecolor="black", density=True)
ax.set_title("Distribution of 7-Day Stock Price Change After Filing")
ax.set_xlabel("Stock Price Change (%)")
ax.set_ylabel("Density")
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("distribution-7-day-average-stock-price-change.png")
plt.show()
