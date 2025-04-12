# Week 3: Working with Libraries and Data

## Overview
This week introduces Python libraries essential for data manipulation and visualization, foundational for AI. Students will learn to handle numerical data with NumPy, manage datasets with pandas, visualize data using Matplotlib, and read/write CSV files. By the end, you'll be able to load, analyze, and visualize datasets like the Iris dataset.

## Objectives
- Understand and use NumPy for numerical operations.
- Manipulate datasets with pandas (Series, DataFrames, cleaning).
- Create visualizations (line plots, scatter plots, histograms) with Matplotlib.
- Read and write data to CSV files.
- Apply these skills in a mini-project to analyze and visualize a dataset.

## Topics
1. **NumPy**:
   - Arrays (1D, 2D, zeros, ones, random).
   - Array operations (addition, multiplication, dot product).
   - Indexing, slicing, and broadcasting.
2. **pandas**:
   - Series and DataFrames.
   - Loading datasets with `pd.read_csv()`.
   - Basic data cleaning (handling missing values).
   - Filtering and selecting data.
3. **Matplotlib**:
   - Line plots, scatter plots, histograms.
   - Customizing plots (labels, titles, colors).
   - Saving visualizations.
4. **Data I/O**:
   - Reading and writing CSV files with pandas.
   - Best practices for data workflows.

## Activities
- **Class Exercises**:
  - NumPy: Perform matrix addition, slice arrays, compute sums.
  - pandas: Filter a DataFrame, handle missing data, compute statistics.
  - Matplotlib: Plot a sine wave, create histograms, design scatter plots.
  - Data I/O: Read a CSV, modify data, save results.
- **Mini-Project**:
  - Load the Iris dataset (or similar).
  - Analyze petal length and width using pandas.
  - Create a scatter plot of petal length vs. width, colored by species.
  - Save the plot and a filtered dataset as files.

## Setup Instructions
1. **Install Python**: Ensure Python 3.8+ is installed ([download](https://www.python.org/downloads/)).
2. **Install Libraries**:
   ```bash
   pip install numpy pandas matplotlib