# MARKETING CAMPAIGN RESULTS
A Python project for a marketing campaign data of 2,240 customers of Maven Marketing, including customer profiles, product preferences, campaign successes/failures, and channel performance.

## Table of Content
- Project Overview
- Project Scope
- Business Objective
- Document Purpose
- Use Case
- Data Source
- Dataset Overview
- Data Cleaning and Processing
- Data Analysis and Insight
- Recommendation
- Conclusion

## Project Overview
This project focuses on analyzing marketing campaign data from Maven Marketing to understand customer behavior, product preferences, and the effectiveness of different marketing strategies. The dataset includes detailed records of 2,240 customers, covering demographic profiles, purchase behaviors, and campaign responses. The goal is to leverage these insights to optimize marketing efforts, enhance customer retention, and increase revenue.

## Project Scope
**Data Quality Assessment:**
- Identify null values, duplicate records, and outliers.
- Implement strategies for data cleaning and standardization.
**Key Driver Analysis:**
- Investigate factors significantly related to the number of web purchases.
**Campaign Evaluation:**
- Compare the success rates of various marketing campaigns to identify the most effective one.
**Customer Profiling:**
- Determine the average customer’s demographic and behavioral characteristics.
**Product Performance Analysis:**
- Analyze spending patterns across product categories to identify best-performing products.
**Channel Efficiency Analysis:**
- Assess the effectiveness of marketing channels to pinpoint underperforming ones.

## Business Objective
The primary objective of this analysis is to enhance Maven Marketing’s marketing strategies by:

1. Identifying the key drivers of customer purchases.
2.	Understanding customer preferences and their impact on campaign success.  
3.	Optimizing resource allocation towards the most effective campaigns and channels.
4.	Improving customer retention and acquisition strategies.
5.	Enhancing overall marketing ROI by addressing gaps and inefficiencies.

## Document Purpose
This document serves as a comprehensive reference for stakeholders, providing:
- A clear understanding of the project objectives and scope.
- An overview of the dataset and its structure.
- A detailed account of data cleaning and processing techniques.
- Insights derived from the analysis and their business implications.

## Use Case
The insights derived from this project will enable Maven Marketing to:
- Improve customer segmentation and targeting.
- Design more impactful marketing campaigns.
- Optimize budget allocation across channels and campaigns.
- Enhance product offerings based on customer preferences.
- Strengthen customer engagement strategies to boost loyalty.

## Data Source
The dataset is sourced from Maven Analytics [Website](https://app.mavenanalytics.io/datasets?search=marketing) and includes information on 2,240 customers. The data provides a comprehensive view of customer profiles, purchase histories, and marketing campaign responses, enabling a multi-dimensional analysis.

## Dataset Overview
The dataset contains the following key features:

**Demographic Data:**
- Year Birth: Customer's birth year.
- Education: Customer's education level.
- Marital Status: Customer's marital status.
- Income: Customer's yearly household income.
- Kidhome, Teenhome: Number of children and teenagers in the household.
**Customer Behavior:**
- Dt Customer: Date of customer's enrollment with the company.
- Recency: Number of days since the customer's last purchase.
- MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds: Amounts spent on various product categories in the last two years.
- NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases: Number of purchases made through various channels.
- NumWebVisitsMonth: Number of website visits in the last month.
**Marketing Engagement:**
- AcceptedCmp1, AcceptedCmp2, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5: Indicators of whether the customer accepted offers from specific campaigns.
- Response: Indicator of whether the customer responded to the last campaign.
**Additional Information:**
- Complain: Indicator of whether the customer complained in the last two years.
- Country: Customer's location.

## Data Cleaning and Processing
Handling Null Values

Identification:
- Conducted a thorough review of all fields to identify missing data.
- Fields such as Income showed occasional missing values.
  
Resolution:
- Imputed missing Income values using the median of similar demographic groups.
- For non-critical fields with minimal null entries, used forward-fill or backward-fill strategies.
  
Addressing Duplicate Records:
There were no duplicates in the Dataset

## Data Analysis and Insight
The following analysis were carried out in this project:
1.	Are there any null values or outliers? How will you handle them?
2.	What factors are significantly related to the number of web purchases?
3.	Which marketing campaign was the most successful?
4.	What does the average customer look like?
5.	Which products are performing best?
6.	Which channels are underperforming?

The following libraries were imported before the above analysis were carried out:

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

#Explanation:
#pandas is used for handling and manipulating data in tabular form.
#numpy provides mathematical functions for array-based operations.
#seaborn and matplotlib.pyplot are visualization libraries to create graphs and charts.
#scipy.stats.zscore computes the z-scores of numerical columns to detect outliers.
```

The following were also carried out;
``` Python
path = '/content/drive/MyDrive/My Projects/Marketing Campaign Results/Marketing Campaign Results_CSV.csv'
data = pd.read_csv(path)
```

**1.	Are there any null values or outliers? How will you handle them?**
   
Null values refer to missing or undefined entries in a dataset. They can occur due to incomplete data collection, system errors, or manual omissions. Identifying null values is crucial as they can lead to inaccuracies in analysis or misinterpretation of results.

Outliers are extreme data points that deviate significantly from the majority of observations. They can arise from errors, rare events, or natural variability within the data. These values have the potential to skew statistical measures and distort the overall analysis if not properly addressed.

Both null values and outliers are critical considerations during the data cleaning process. Their presence affects the dataset's quality and reliability, making their detection and evaluation a vital step in preparing data for meaningful analysis.

In order to carry out the above, this was done;
``` Python
data.info()

#Explanation:
#data.info() displays the structure of the DataFrame, including column names, data types, and counts of non-null values
```

**Result:**

Data columns (total 28 columns):

| Column                | Non-Null Count   | Dtype   |
|-----------------------|------------------|---------|
| ID                    | 2240 non-null    | int64   |
| Year_Birth            | 2240 non-null    | int64   |
| Education             | 2240 non-null    | object  |
| Marital_Status        | 2240 non-null    | object  |
| Income                | 2216 non-null    | float64 |
| Kidhome               | 2240 non-null    | int64   |
| Teenhome              | 2240 non-null    | int64   |
| Dt_Customer           | 2240 non-null    | object  |
| Recency               | 2240 non-null    | int64   |
| MntWines              | 2240 non-null    | int64   |
| MntFruits             | 2240 non-null    | int64   |
| MntMeatProducts       | 2240 non-null    | int64   |
| MntFishProducts       | 2240 non-null    | int64   |
| MntSweetProducts      | 2240 non-null    | int64   |
| MntGoldProds          | 2240 non-null    | int64   |
| NumDealsPurchases     | 2240 non-null    | int64   |
| NumWebPurchases       | 2240 non-null    | int64   |
| NumCatalogPurchases   | 2240 non-null    | int64   |
| NumStorePurchases     | 2240 non-null    | int64   |
| NumWebVisitsMonth     | 2240 non-null    | int64   |
| AcceptedCmp3          | 2240 non-null    | int64   |
| AcceptedCmp4          | 2240 non-null    | int64   |
| AcceptedCmp5          | 2240 non-null    | int64   |
| AcceptedCmp1          | 2240 non-null    | int64   |
| AcceptedCmp2          | 2240 non-null    | int64   |
| Response              | 2240 non-null    | int64   |
| Complain              | 2240 non-null    | int64   |
| Country               | 2240 non-null    | object  |

dtypes: float64(1), int64(23), object(4)

``` Python
data.isnull().sum()

#Explanation:
#data.isnull().sum() counts the number of null values in each column.
```

**Result:**

Here is the updated table with the new non-null counts in GitHub format:

| Column                | Missing Values |
|-----------------------|----------------|
| ID                    | 0              |
| Year_Birth            | 0              |
| Education             | 0              |
| Marital_Status        | 0              |
| Income                | 24             |
| Kidhome               | 0              |
| Teenhome              | 0              |
| Dt_Customer           | 0              |
| Recency               | 0              |
| MntWines              | 0              |
| MntFruits             | 0              |
| MntMeatProducts       | 0              |
| MntFishProducts       | 0              |
| MntSweetProducts      | 0              |
| MntGoldProds          | 0              |
| NumDealsPurchases     | 0              |
| NumWebPurchases       | 0              |
| NumCatalogPurchases   | 0              |
| NumStorePurchases     | 0              |
| NumWebVisitsMonth     | 0              |
| AcceptedCmp3          | 0              |
| AcceptedCmp4          | 0              |
| AcceptedCmp5          | 0              |
| AcceptedCmp1          | 0              |
| AcceptedCmp2          | 0              |
| Response              | 0              |
| Complain              | 0              |
| Country               | 0              |

From the table above;

The dataset contains only one column with missing values: Income, which has 24 missing records. All other columns, including demographic details (such as Year_Birth, Education, and Marital_Status) and purchasing behaviors (such as MntWines, NumDealsPurchases, and AcceptedCmp1), have no missing values. The absence of missing data in most columns suggests a high level of completeness, with the Income column being the only area requiring attention.

**Handling Null Values**
``` Python
num_cols = data.select_dtypes(include=['float64', 'int64']).columns
cat_cols = data.select_dtypes(include=['object']).columns

#Explanation:
#Identifies numerical columns (float64, int64) and categorical columns (object) separately for appropriate handling of missing data.

for col in num_cols:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].median(), inplace=True)

#Explanation: 
#For each numerical column with null values, replace missing values with the median (fillna(data[col].median())). This reduces the influence of extreme values compared to using the mean.

data.isnull().sum()

#Explanation: 
#Verifies that all null values have been handled.
```

**Result:**

| Column                | Missing Values |
|-----------------------|----------------|
| ID                    | 0              |
| Year_Birth            | 0              |
| Education             | 0              |
| Marital_Status        | 0              |
| Income                | 0              |
| Kidhome               | 0              |
| Teenhome              | 0              |
| Dt_Customer           | 0              |
| Recency               | 0              |
| MntWines              | 0              |
| MntFruits             | 0              |
| MntMeatProducts       | 0              |
| MntFishProducts       | 0              |
| MntSweetProducts      | 0              |
| MntGoldProds          | 0              |
| NumDealsPurchases     | 0              |
| NumWebPurchases       | 0              |
| NumCatalogPurchases   | 0              |
| NumStorePurchases     | 0              |
| NumWebVisitsMonth     | 0              |
| AcceptedCmp3          | 0              |
| AcceptedCmp4          | 0              |
| AcceptedCmp5          | 0              |
| AcceptedCmp1          | 0              |
| AcceptedCmp2          | 0              |
| Response              | 0              |
| Complain              | 0              |
| Country               | 0              |

From the table above, the dataset contains no column with missing values.

**Identifying and handling outliers**

``` Python
# Define the number of columns dynamically
num_cols = data.select_dtypes(include=['number']).columns  # Example numeric columns
num_plots = len(num_cols)

# Calculate grid size
rows = (num_plots - 1) // 4 + 1  # Dynamically calculate rows for up to 4 columns per row
cols = min(4, num_plots)  # Limit columns to 4

plt.figure(figsize=(15, rows * 3))  # Adjust height based on the number of rows

for i, col in enumerate(num_cols, 1):
    plt.subplot(rows, cols, i)
    sns.boxplot(x=data[col])
    plt.title(col)

plt.tight_layout()
plt.show()

#Explanation: 
#Plots boxplots for each numerical column to visually identify outliers (data points outside the whiskers). The plt.subplot() function organizes multiple boxplots into a grid layout.
#Key Changes:
#Dynamically calculate rows and cols based on the number of plots (num_plots).
#Adjust figsize to ensure plots are properly spaced.
#Ensure the grid accommodates all columns in num_cols.
#This approach prevents exceeding the allowed number of subplots.
```

**Result:**

![](https://github.com/OluwaseunOkundalaye/MARKETING-CAMPAIGN-RESULTS/blob/main/Outliers.png)

The image displays boxplots for various features in the dataset, summarizing the distribution and outliers:
- ID: The ID feature has a narrow range, indicating it’s likely a unique identifier with no significant distribution.
- Year_Birth: Concentrated between the 1940s and 1980s, with a few outliers representing individuals born outside this range.
- Income: The distribution is highly skewed, with a few extreme outliers at the higher end, indicating a small group of individuals with very high incomes.
- Kidhome: This binary feature shows most data at 0, indicating most individuals have no children at home, with a few outliers at 1.
- Teenhome: Similar to Kidhome, most values are at 0 (no teenagers at home), with a few outliers at 1.
- Recency: Even distribution with no significant outliers, indicating a regular range of recent customer interactions.
- Spending on Products (MntWines, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds): All product categories show skewed distributions with several high outliers, suggesting that a small number of individuals spend significantly more on certain products.
- Purchase Frequency (NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases): These features show a concentration of values at lower levels, with a few outliers indicating customers who make more frequent purchases across different channels.
- NumWebVisitsMonth: A broad distribution, with some individuals visiting websites much more frequently than others.
- Campaign Acceptance (AcceptedCmp1-5): Most individuals did not accept the campaigns (0), with a few outliers showing acceptance (1).
- Response: Concentrated at 0 (no response), with few outliers indicating individuals who responded to campaigns.
- Complain: Like Response, most values are 0 (no complaint), with a few outliers representing individuals who filed complaints.

Summary:

The dataset shows skewed distributions for income, product spending, and purchase frequency. Several binary features (Kidhome, Teenhome, campaign acceptance) are mostly concentrated at 0, with a few outliers. Outliers in product spending and campaign responses represent a small number of individuals with extreme behaviors. These insights can help in customer segmentation and targeted marketing strategies.

```Python
z_scores = data[num_cols].apply(zscore)
outliers = (z_scores > 3) | (z_scores < -3)
print("Outlier count per column:\n", outliers.sum())

#Explanation: 
#Computes z-scores for numerical columns to standardize values. Flags values with z-scores greater than 3 (or less than -3) as outliers. Summarizes how many outliers exist in each column.
```

**Result:**

Outlier count per column:
Here is the table in GitHub format:

| Column                | Outlier Count |
|-----------------------|---------------|
| ID                    | 0             |
| Year_Birth            | 3             |
| Income                | 8             |
| Kidhome               | 0             |
| Teenhome              | 0             |
| Recency               | 0             |
| MntWines              | 16            |
| MntFruits             | 64            |
| MntMeatProducts       | 37            |
| MntFishProducts       | 58            |
| MntSweetProducts      | 62            |
| MntGoldProds          | 44            |
| NumDealsPurchases     | 32            |
| NumWebPurchases       | 4             |
| NumCatalogPurchases   | 4             |
| NumStorePurchases     | 0             |
| NumWebVisitsMonth     | 9             |
| AcceptedCmp3          | 163           |
| AcceptedCmp4          | 167           |
| AcceptedCmp5          | 163           |
| AcceptedCmp1          | 144           |
| AcceptedCmp2          | 30            |
| Response              | 0             |
| Complain              | 21            |


**2.	What factors are significantly related to the number of web purchases?**

This question seeks to identify the variables in the dataset that have a meaningful impact on the number of purchases made through a website. This involves exploring different customer-related features such as income, age, spending on products, recency, and website visits to determine which ones show a strong correlation with the frequency of web purchases. 

In order to address thus, the following were carried out;

```  Python
#Import Necessary Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

#Explanation:
#train_test_split splits the data into training and testing sets.
#LinearRegression creates and fits a linear regression model.
#r2_score and mean_squared_error evaluate the model's performance.

# Select only numeric columns
numeric_data = data.select_dtypes(include=['number'])

# Compute the correlation matrix
correlation = numeric_data.corr()

# Sort correlations with 'NumWebPurchases' in descending order
correlation_with_target = correlation['NumWebPurchases'].sort_values(ascending=False)

# Display the result
print(correlation_with_target)

#Explanation:
#data.corr() calculates correlation coefficients for all numerical columns.
#['NumWebPurchases'] extracts the correlations of NumWebPurchases with other variables.
#sort_values() sorts the correlation values to easily identify strong positive or negative relationships.
```

**Result:**

Here is the result in GitHub table format:

| Column               | Correlation with NumWebPurchases |
|----------------------|----------------------------------|
| NumWebPurchases      | 1.000000                         |
| MntWines             | 0.542265                         |
| NumStorePurchases    | 0.502713                         |
| MntGoldProds         | 0.421836                         |
| Income               | 0.380554                         |
| NumCatalogPurchases  | 0.378376                         |
| MntSweetProducts     | 0.348544                         |
| MntFruits            | 0.296735                         |
| MntMeatProducts      | 0.293761                         |
| MntFishProducts      | 0.293681                         |
| NumDealsPurchases    | 0.234185                         |
| AcceptedCmp4         | 0.155903                         |
| Teenhome             | 0.155500                         |
| AcceptedCmp1         | 0.155143                         |
| Response             | 0.148730                         |
| AcceptedCmp5         | 0.138684                         |
| AcceptedCmp3         | 0.042176                         |
| AcceptedCmp2         | 0.034188                         |
| Recency              | -0.010726                        |
| Complain             | -0.016310                        |
| ID                   | -0.018924                        |
| NumWebVisitsMonth    | -0.055846                        |
| Year_Birth           | -0.145040                        |
| Kidhome              | -0.361647                        |

The correlation analysis reveals several significant relationships between NumWebPurchases and other variables, offering insights into factors that could influence the number of web purchases.

1.	Strong Positive Correlations: The variables most positively correlated with NumWebPurchases include:
- MntWines (0.54): Customers who spend more on wine are more likely to make web purchases.
- NumStorePurchases (0.50): A moderate correlation indicates that individuals who make more purchases in stores also tend to purchase more on the web.
- MntGoldProds (0.42): Higher spending on gold products is associated with increased web purchases.
- Income (0.38): A positive correlation suggests that wealthier customers are more likely to make web purchases.
- NumCatalogPurchases (0.38): Customers who purchase more products from catalogs tend to make more web purchases as well.

2.	Moderate to Weak Positive Correlations: Other factors like MntSweetProducts (0.35), MntFruits (0.30), and MntMeatProducts (0.29) show a moderate positive correlation with NumWebPurchases, indicating that customers who spend on these product categories also tend to make more web purchases, although the relationship is weaker than with the top factors.

3.	Weak Negative Correlations: There are a few variables that exhibit weak negative correlations with NumWebPurchases, such as Recency (-0.01), Complain (-0.02), and ID (-0.02), which are not strongly related to web purchases. Additionally, Kidhome (-0.36) shows a negative relationship, implying that customers with children at home may be less likely to make web purchases.

4.	Campaign Acceptance: Variables related to campaign acceptance (e.g., AcceptedCmp1 to AcceptedCmp5) show weak to moderate positive correlations with NumWebPurchases, with AcceptedCmp4 having the strongest positive correlation (0.16). This suggests that customers who respond to marketing campaigns are somewhat more likely to make web
purchases.

Summary:

The analysis indicates that MntWines, NumStorePurchases, and MntGoldProds are the strongest predictors of NumWebPurchases. These factors, along with Income and NumCatalogPurchases, offer valuable insights for identifying customers who are more likely to make purchases on the web. On the other hand, Kidhome, Recency, and Complain are weaker predictors, with some negative correlations, suggesting they have less influence on web purchase behavior.

**Visualize Relationships**

``` Python
sns.pairplot(data[['NumWebPurchases', ' Income ', 'Recency', 'NumDealsPurchases', 'NumWebVisitsMonth']])
plt.show()

#Explanation:
#Creates scatterplots between NumWebPurchases and selected variables to observe relationships visually.
#Helps identify potential trends or non-linear patterns.
```

**Result:**

![](https://github.com/OluwaseunOkundalaye/MARKETING-CAMPAIGN-RESULTS/blob/main/Pairplot%20Relationship.png)
