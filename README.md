# MARKETING CAMPAIGN RESULTS
A Python project for a marketing campaign data of 2,240 customers of Maven Marketing, including customer profiles, product preferences, campaign successes/failures, and channel performance.

## Table of Content
- [Project Overview]()

- [Project Scope]()

- [Business Objective]()

- [Document Purpose]()

- [Use Case]()

- [Data Source]()

- [Dataset Overview]()

- [Data Cleaning and Processing]()

- [Data Analysis and Insight]()

- [Recommendation]()

- [Conclusion]()

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

1.	[Are there any null values or outliers? How will you handle them?](https://github.com/OluwaseunOkundalaye/MARKETING-CAMPAIGN-RESULTS#1are-there-any-null-values-or-outliers-how-will-you-handle-them)

2.	[What factors are significantly related to the number of web purchases?](https://github.com/OluwaseunOkundalaye/MARKETING-CAMPAIGN-RESULTS#2what-factors-are-significantly-related-to-the-number-of-web-purchases)

3.	[Which marketing campaign was the most successful?](https://github.com/OluwaseunOkundalaye/MARKETING-CAMPAIGN-RESULTS#3which-marketing-campaign-was-the-most-successful)

4.	[What does the average customer look like](https://github.com/OluwaseunOkundalaye/MARKETING-CAMPAIGN-RESULTS#4what-does-the-average-customer-look-like)

5.	[Which products are performing best?](https://github.com/OluwaseunOkundalaye/MARKETING-CAMPAIGN-RESULTS#5which-products-are-performing-best)

6.	[Which channels are underperforming?](https://github.com/OluwaseunOkundalaye/MARKETING-CAMPAIGN-RESULTS#6which-channels-are-underperforming)

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

### 1.	Are there any null values or outliers? How will you handle them?**
   
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


### 2.	What factors are significantly related to the number of web purchases?**

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

The pairplot of the variables NumWebPurchases, Income, Recency, NumDealsPurchases, and NumWebVisitsMonth visually shows the relationships between these features. Here's a breakdown of the key observations:

**1.	NumWebPurchases vs. Other Variables:**
- NumWebPurchases vs. Income: The scatterplot shows no clear linear relationship between NumWebPurchases and Income. While there are a few high-income outliers, the data for NumWebPurchases is spread across all income levels, indicating that income does not strongly determine web purchase behavior.
- NumWebPurchases vs. Recency: There's no clear trend between the number of web purchases and recency (the number of days since last purchase). The scatterplot is dispersed, suggesting that recency does not directly influence web purchases.
- NumWebPurchases vs. NumDealsPurchases: There is a moderate positive trend, with customers who make more purchases in deals also making more web purchases. This suggests that promotions or deals may encourage customers to buy more online.
- NumWebPurchases vs. NumWebVisitsMonth: This plot shows a clear positive trend, indicating that customers who visit the website more frequently are more likely to make web purchases. This suggests a strong relationship between website engagement and purchase behavior.

**2.	Variable Relationships:**
- Income vs. Recency: The plot shows a scattered distribution, with no clear pattern between Income and Recency.
- NumDealsPurchases vs. NumWebVisitsMonth: There's a visible positive relationship, with customers who engage more with deals also visiting the website more frequently, suggesting that promotional activities drive both visits and purchases.

**3.	Distributions:**
- Most variables, particularly NumDealsPurchases and NumWebVisitsMonth, have a skewed distribution, with many customers making fewer purchases and visits, and a smaller group making high numbers. Recency shows a more uniform distribution, while Income is highly right-skewed with a concentration of lower-income individuals.

Summary:

The pairplot highlights several key relationships, particularly the strong connection between NumWebPurchases and NumWebVisitsMonth. It also indicates that NumDealsPurchases positively impacts web purchases, while Income and Recency show weaker associations. Understanding these patterns can help in developing targeted marketing strategies, such as incentivizing frequent website visits and promoting deals to increase web purchases.

**Scatter Plot**

``` Python
sns.scatterplot(x=' Income ', y='NumWebPurchases', data=data)
plt.title('Income vs. NumWebPurchases')
plt.xlabel('Income')
plt.ylabel('Number of Web Purchases')
plt.show()

#Explanation: 
#Plots Income (independent variable) against NumWebPurchases (dependent variable) to examine the relationship.
```
![](https://github.com/OluwaseunOkundalaye/MARKETING-CAMPAIGN-RESULTS/blob/main/Scattered%20Plot%20Relationship.png)

The scatter plot of Income vs. NumWebPurchases shows the relationship between a customer's income and the number of web purchases they make. Here are the key observations:
1. Distribution of Data:
- The plot reveals that most customers have relatively low incomes (around 0 to 100,000), and their number of web purchases tends to be lower, with many customers making 0 to 5 purchases.
- A small number of customers with higher incomes (above 200,000) have a higher number of web purchases, with a few extreme outliers making up to 25 web purchases. This indicates that, while most customers with higher incomes tend to make more purchases, this is not a consistent pattern.

2. No Clear Linear Relationship:
- Overall, there is no clear or strong linear relationship between Income and NumWebPurchases. While income might influence purchasing behavior to some extent, it does not appear to be a strong predictor of how many purchases a customer makes.
- This is supported by the fact that there is no visible trend or clustering of points in the plot, especially for the majority of customers who make fewer web purchases.

Conclusion:

The scatter plot suggests that income alone does not appear to strongly influence the number of web purchases. Factors other than income, such as product offerings, promotions, or website engagement, may play a more significant role in driving web purchases. This analysis indicates that further investigation into these other factors would be beneficial to fully understand the drivers of web purchases.

**Prepare Data for Regression**
``` Pyhton
#Select relevant predictors
X = data[[' Income ', 'Recency', 'NumDealsPurchases', 'NumWebVisitsMonth']]  # Adjust columns as needed
y = data['NumWebPurchases']
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Explanation:
#X includes independent variables (predictors).
#y is the dependent variable (NumWebPurchases).
#train_test_split divides the data into training and testing sets to evaluate the regression model.
```

**Perform Linear Regression**
```Python
# Initialize and fit the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Coefficients
print("Coefficients:", regressor.coef_)
print("Intercept:", regressor.intercept_)

#Explanation:
#LinearRegression() initializes the regression model.
#fit(X_train, y_train) trains the model using the training data.
#Prints the model coefficients (impact of each predictor) and intercept (constant term).
```

**Result:**

Coefficients: [ 4.23976361e-05 -5.29364539e-04  3.54974807e-01  9.61439886e-02]

Intercept: 0.5573787799195022

**Evaluate the Model**
``` Python
#Make predictions
y_pred = regressor.predict(X_test)

# Evaluate performance
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("R-squared:", r2)
print("Mean Squared Error:", mse)

#Explanation:
#predict(X_test) generates predictions for the test data.
#r2_score() evaluates how well the model explains variability in the target variable.
#mean_squared_error() measures the average squared difference between actual and predicted values.
```

**Result:**

R-squared: 0.26274398252805076

Mean Squared Error: 5.90055581330441


**Data Preparation and Regression Model Evaluation Report**

**Data Preparation**

The dataset was prepared for linear regression by selecting relevant predictors (independent variables) and the dependent variable. The independent variables include Income, Recency, NumDealsPurchases, and NumWebVisitsMonth, which are assumed to influence the number of web purchases a customer makes. The dependent variable, NumWebPurchases, represents the number of web purchases made by a customer.

The data was split into training and testing sets using the train_test_split function. The training set contains 70% of the data, while the testing set holds the remaining 30%. This division is essential to train the regression model on one portion of the data and evaluate its performance on another, ensuring an unbiased assessment.

**Linear Regression Model**

The Linear Regression model was trained using the training data. The coefficients and intercept for the model were printed to understand the influence of each predictor on the target variable:

**Coefficients:**
- Income: 4.24e-05
- Recency: -5.29e-04
- NumDealsPurchases: 0.354975
- NumWebVisitsMonth: 0.096144
  
These coefficients indicate the magnitude and direction of the impact each predictor has on NumWebPurchases. For example, an increase in NumDealsPurchases is associated with a rise in NumWebPurchases, while Recency has a negative association.
- Intercept: 0.5574, indicating the baseline number of web purchases when all predictors are zero.

**Model Evaluation**

After fitting the model, I made predictions on the testing set and evaluated the model’s performance using two metrics:
- R-squared: 0.2627, suggesting that approximately 26.27% of the variability in the number of web purchases is explained by the predictors in the model. While this is not a very high value, it indicates that the model has some predictive power, but there are likely other important factors not included in the model.
- Mean Squared Error (MSE): 5.9006, which represents the average squared difference between the actual and predicted number of web purchases. A higher value of MSE suggests that there is considerable error in the model's predictions.

Conclusion:

The linear regression model provides some insights into the predictors of NumWebPurchases, but the relatively low R-squared value indicates that the model does not fully explain the variability in the target variable. Further model refinement and inclusion of additional predictors could potentially improve its accuracy. The MSE indicates room for improvement in predicting the number of web purchases more precisely.

### 3.	Which marketing campaign was the most successful?**

The question, "Which marketing campaign was the most successful?" focuses on evaluating the outcomes of different campaigns to identify the one that achieved the best results. Success is typically measured by metrics such as increased sales, higher customer engagement, or a strong return on investment (ROI). To determine this, it is important to analyze data like revenue generated, the number of new customers acquired, and the effectiveness of the campaign in meeting its intended goals.

Additionally, the evaluation must consider factors like the cost of implementing each campaign, the target audience reached, and the timing or channels used. Metrics such as ROI provide insight into the efficiency of each campaign, while external influences like seasonal trends or market conditions help contextualize the results. This analysis ensures a fair comparison, allowing for a well-supported conclusion on which campaign was the most impactful.

The following were done to address the analysis:

**Calculate Acceptance Rates for Each Campaign**

``` Python
# Calculate the acceptance rate for each campaign
campaign_columns = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
acceptance_rates = data[campaign_columns].mean()  # Calculate the mean for each campaign column (acceptance rate)
acceptance_rates = acceptance_rates.sort_values(ascending=False)  # Sort campaigns by acceptance rate

# Display acceptance rates
print(acceptance_rates)

#Explanation:
#campaign_columns is a list of columns representing the campaigns (AcceptedCmp1 to AcceptedCmp5).
#data[campaign_columns].mean() calculates the mean (acceptance rate) for each campaign column, where 1 indicates acceptance and 0 indicates non-acceptance.
#sort_values(ascending=False) sorts the campaigns from the highest to the lowest acceptance rate.
```

**Result:**
- AcceptedCmp4    0.074554
- AcceptedCmp3    0.072768
- AcceptedCmp5    0.072768
- AcceptedCmp1    0.064286
- AcceptedCmp2    0.013393

**Visualize Acceptance Rates Using Bar Plot**

``` Python
# Create a bar plot for campaign acceptance rates
plt.figure(figsize=(10, 6))
sns.barplot(x=acceptance_rates.index, y=acceptance_rates.values, palette="Blues_d")
plt.title('Acceptance Rates by Marketing Campaign')
plt.xlabel('Campaign')
plt.ylabel('Acceptance Rate')
plt.xticks(rotation=45)
plt.show()

#Explanation:
#sns.barplot() creates a bar plot showing the acceptance rates for each campaign.
#acceptance_rates.index contains the campaign names (e.g., AcceptedCmp1, AcceptedCmp2), and acceptance_rates.values contains the corresponding acceptance rates.
#plt.figure(figsize=(10, 6)) adjusts the plot size for better readability.
#plt.xticks(rotation=45) rotates the campaign names for better visualization.
```

![](https://github.com/OluwaseunOkundalaye/MARKETING-CAMPAIGN-RESULTS/blob/main/Acceptance%20rate%20by%20marketing%20campaign.png)

The provided bar chart illustrates the acceptance rates for five different marketing campaigns, identified as AcceptedCmp4, AcceptedCmp3, AcceptedCmp5, AcceptedCmp1, and AcceptedCmp2. The acceptance rate, displayed on the y-axis, quantifies the proportion of customers who accepted each campaign's offer.

From the chart, it is evident that campaigns AcceptedCmp4, AcceptedCmp3, and AcceptedCmp5 achieved the highest acceptance rates, with very similar performance near the 0.07 mark. Campaign AcceptedCmp1 follows closely behind with a slightly lower acceptance rate. However, AcceptedCmp2 had a significantly lower acceptance rate compared to the other campaigns, indicating a lack of effectiveness. These findings suggest that while most campaigns performed relatively well, AcceptedCmp2 requires further investigation to understand its underperformance.

**Combine Campaign Performance with Demographic Insights**

``` Python
# Group by 'Education' to see how acceptance rates vary by customer education level
campaign_by_education = data.groupby('Education')[campaign_columns].mean()

# Visualize the campaign acceptance rates by education level
campaign_by_education.plot(kind='bar', figsize=(12, 6), colormap='viridis')
plt.title('Campaign Acceptance Rates by Education Level')
plt.ylabel('Acceptance Rate')
plt.xlabel('Education Level')
plt.xticks(rotation=45)
plt.show()

#Explanation:
#data.groupby('Education')[campaign_columns].mean() groups the data by education level and calculates the mean (acceptance rate) for each campaign for each education group.
#plot(kind='bar') creates a bar plot to visualize how the acceptance rates differ across education levels.
#colormap='viridis' provides a color palette for the bars.
#plt.xticks(rotation=45) rotates the education levels for better readability.
```

![](https://github.com/OluwaseunOkundalaye/MARKETING-CAMPAIGN-RESULTS/blob/main/Campaign%20acceptance%20rate%20by%20educational%20level.png)

Key Observations:

1.	Basic Education:
- This group overwhelmingly accepted AcceptedCmp3, with an acceptance rate exceeding 0.10.
- No significant responses were observed for other campaigns.

2.	2n Cycle:
- Campaigns AcceptedCmp3 and AcceptedCmp4 had comparable acceptance rates (around 0.07).
- AcceptedCmp1 and AcceptedCmp5 were moderately accepted, while AcceptedCmp2 had minimal response.

3.	Graduation:
-	AcceptedCmp3 and AcceptedCmp4 had the highest acceptance rates (approximately 0.08).
-	AcceptedCmp5 also performed well, followed by AcceptedCmp1.
-	AcceptedCmp2 showed the least acceptance in this category.

4.	Master:
-	AcceptedCmp3, AcceptedCmp4, and AcceptedCmp5 showed strong acceptance rates, clustered around 0.07–0.08.
-	AcceptedCmp1 was moderately accepted, while AcceptedCmp2 lagged significantly.

5.	PhD:
-	The acceptance rates for AcceptedCmp3, AcceptedCmp4, and AcceptedCmp5 were consistently high, around 0.08.
-	Similar to other education levels, AcceptedCmp1 had a moderate response, and AcceptedCmp2 remained the least accepted.

Summary:

Across most education levels, AcceptedCmp3, AcceptedCmp4, and AcceptedCmp5 are the most successful campaigns, particularly for Basic, Graduation, Master, and PhD levels.
AcceptedCmp2 consistently underperforms, indicating possible issues with the campaign's targeting or messaging.
There is a notable variance in acceptance rates by education, suggesting that certain campaigns resonate better with specific groups.

### 4.	What does the average customer look like?**

This question aims to identify the key characteristics that define the typical customer in the dataset. This includes analyzing demographic factors such as age, gender, marital status, and education level, as well as behavioral patterns like purchasing habits, campaign acceptance rates, and product preferences. These insights help paint a clear picture of the customer base's overall composition.

The following were carried out in order to address this analysis;

``` Python
# Calculate mean, median, and standard deviation for numerical columns
numerical_stats = data.describe()  # Default gives count, mean, std, min, 25%, 50%, 75%, max
mean_age = np.mean(data['Year_Birth'])
median_income = np.median(data[' Income '])
print("Mean Age: ", mean_age)
print("Median Income: ", median_income)

#Explanation:
#data.describe() calculates common descriptive statistics (mean, standard deviation, min, max, and percentiles) for numerical columns.
#np.mean() and np.median() are used to compute the mean and median of specific columns like age (Year_Birth) and income.
#This helps identify the central tendency and spread of numerical features in the dataset.
```

**Result:**
- Mean Age:  1968.8058035714287
- Median Income:  51381.5

**Identify Most Common Values for Categorical Columns**

``` Python
# Find the most frequent (mode) values for categorical columns
mode_marital_status = data['Marital_Status'].mode()[0]  # Mode returns a Series, we select the first value
mode_education = data['Education'].mode()[0]
mode_country = data['Country'].mode()[0]

print("Most common Marital Status: ", mode_marital_status)
print("Most common Education Level: ", mode_education)
print("Most common Country: ", mode_country)

#Explanation:
#mode() finds the most frequent value(s) for categorical columns. For example, marital status, education, and country.
#Since mode() returns a Series (in case of ties), we select the first value using [0] to get the most common category.
```

**Result:**
- Most common Marital Status:  Married
- Most common Education Level:  Graduation
- Most common Country:  Spain

**Combine Findings into a Profile Summary**

``` Python
# Combine the findings into a summary of the average customer profile
average_customer_profile = {
    'Average Age': 2024 - mean_age,  # Calculate age from birth year (assuming current year is 2024)
    'Median Income': median_income,
    'Most Common Marital Status': mode_marital_status,
    'Most Common Education Level': mode_education,
    'Most Common Country': mode_country
}

# Print the profile summary
print("\nAverage Customer Profile Summary:")
for key, value in average_customer_profile.items():
    print(f"{key}: {value}")

#Explanation:
#I created a dictionary average_customer_profile to combine the key findings (average age, median income, most common marital status, education level, and country).
#The age is calculated by subtracting the Year_Birth from the current year (2024 in this case).
#The profile summary is printed by iterating over the dictionary and displaying each key-value pair.
```

**Result:**
- Average Customer Profile Summary:
- Average Age: 55.19419642857133
- Median Income: 51381.5
- Most Common Marital Status: Married
- Most Common Education Level: Graduation
- Most Common Country: Spain

**Visualize Customer Profile (for better understanding)**

``` Python
# Visualize some key aspects of the average customer profile
# Plot age distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Year_Birth'], bins=30, kde=True, color='blue')
plt.title('Customer Age Distribution')
plt.xlabel('Year of Birth')
plt.ylabel('Frequency')
plt.show()

# Plot income distribution
plt.figure(figsize=(10, 6))
sns.histplot(data[' Income '], bins=30, kde=True, color='green')
plt.title('Customer Income Distribution')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.show()

# Bar plot for marital status and education level
plt.figure(figsize=(10, 6))
sns.countplot(x='Marital_Status', data=data, palette='Blues_d')
plt.title('Marital Status Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Education', data=data, palette='Purples_d')
plt.title('Education Level Distribution')
plt.show()

#Explanation:
#sns.histplot() is used to create histograms with kernel density estimation (KDE) for continuous features like age and income.
#sns.countplot() is used to visualize the distribution of categorical features like marital status and education level.
#These visualizations help to better understand the distribution of customer characteristics.
```

![](https://github.com/OluwaseunOkundalaye/MARKETING-CAMPAIGN-RESULTS/blob/main/Customer%20age%20distribution.png)

The histogram above illustrates the distribution of customer ages based on their years of birth. The data shows a concentration of customers born between 1960 and 1980, with the peak frequency around the mid-1970s. This indicates that the majority of the customer base is likely between the ages of 40 and 60 years old, making middle-aged adults the predominant group. There is a noticeable decline in representation for customers born after 1980 and minimal presence of customers born before 1940, reflecting the smaller proportion of younger and older customers within the dataset.

![](https://github.com/OluwaseunOkundalaye/MARKETING-CAMPAIGN-RESULTS/blob/main/Customer%20income%20distribution.png)

The image provided is a histogram displaying the income distribution among customers. The x-axis represents income ranging from 0 to 600,000, while the y-axis indicates the frequency of customers within each income bracket. The bars in the histogram are green, with a green density plot line overlaying them. The chart, titled "Customer Income Distribution," shows that the majority of the customers have incomes below 100,000, and there is a significant drop in frequency as the income increases beyond this point. This trend suggests a highly skewed income distribution, with most customers clustered at the lower end of the income scale.

![](https://github.com/OluwaseunOkundalaye/MARKETING-CAMPAIGN-RESULTS/blob/main/Marital%20status%20distribution.png)

The bar chart, titled "Marital Status Distribution," illustrates the frequency of individuals across various marital statuses. The x-axis lists eight categories: Married, Single, Widow, Divorced, Together, Alone, YOLO, and Absurd, while the y-axis represents the count, ranging from 0 to 900. The highest count is observed in the "Married" category, with over 800 individuals, followed by the "Together" category, slightly above 600. "Single" has approximately 450 individuals, and "Divorced" about 300. The "Widow" category stands at around 100, and the categories "Alone," "YOLO," and "Absurd" each have fewer than 50 individuals.

![](https://github.com/OluwaseunOkundalaye/MARKETING-CAMPAIGN-RESULTS/blob/main/Education%20level%20distribution.png)

The image above is a bar chart titled "Education Level Distribution." It displays the count of individuals across different education levels, ranging from "Graduation" to "Basic." The x-axis categorizes the education levels as "Graduation," "Master," "PhD," "2nd Cycle," and "Basic," while the y-axis shows the count of individuals, ranging from 0 to 1200. The "Graduation" level has the highest count, with over 1000 individuals, followed by the "PhD" level with around 600. The "Master" level has approximately 400 individuals, the "2nd Cycle" has about 200, and the "Basic" education level has slightly over 100 individuals. The bars are shaded in different shades of purple for distinction.

### 5.	Which products are performing best?**

This question seeks to identify the top-performing items in terms of key business metrics such as sales volume, revenue, or profitability. Analyzing product performance helps determine which items contribute the most to the company’s success and can guide decisions on inventory management, marketing strategies, and resource allocation.

The following were carried out in order to address this analysis;

**Sum Spending Across Product Categories**

``` Python
# Sum the spending for each product category
product_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
total_spending = data[product_columns].sum()  # Sum spending for each product category

print("Total spending per product category:\n", total_spending)

#Explanation:
#data[product_columns].sum(): Sums up the total spending for each product category. The list product_columns contains the columns related to product spending.
#This step helps to determine the total revenue generated from each product category across all customers.
```

**Result:**

| Product Category     | Total Spending |
|----------------------|----------------|
| MntWines             | 680,816        |
| MntFruits            | 58,917         |
| MntMeatProducts      | 373,968        |
| MntFishProducts      | 84,057         |
| MntSweetProducts     | 60,621         |
| MntGoldProds         | 98,609         |


**Visualize Total Spend per Product Category**

``` Python
# Plot the total spending for each product category using a bar plot
plt.figure(figsize=(10, 6))
total_spending.plot(kind='bar', color='skyblue')  # Plot as a bar chart
plt.title('Total Spending per Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Spending')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.show()

#Explanation:
#total_spending.plot(kind='bar', color='skyblue'): Creates a bar plot to visualize the total spending in each product category. The kind='bar' argument specifies that the #plot will be a bar chart, and color='skyblue' sets the bar color.
#plt.xticks(rotation=45, ha='right'): Rotates the x-axis labels (product categories) for better readability, especially when there are long labels.
#This visualization allows you to quickly compare the total spending for each product category
```

![](https://github.com/OluwaseunOkundalaye/MARKETING-CAMPAIGN-RESULTS/blob/main/Total%20spending%20per%20product%20category.png)

The bar chart titled "Total Spending per Product Category" presents a breakdown of consumer expenditures across six distinct product categories: MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, and MntGoldProds. The data reveals that MntWines leads significantly in total spending, indicating a strong consumer preference or higher price points in this category. Following this, MntMeatProducts also shows substantial spending, while categories like MntFruits have the lowest spending, suggesting either less consumer interest or more affordable prices.

**Visualize Total Spend as a Pie Chart**

``` Python
# Plot the total spending as a pie chart
plt.figure(figsize=(8, 8))
total_spending.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set3', len(total_spending)))
plt.title('Spending Distribution Across Product Categories')
plt.ylabel('')  # Remove the y-axis label for a cleaner look
plt.show()

#Explanation:
#total_spending.plot(kind='pie'): Plots the total spending as a pie chart to show the distribution of spending across product categories.
#autopct='%1.1f%%': Displays the percentage of total spending for each product category.
#startangle=90: Rotates the chart to start the first slice at 90 degrees for better alignment.
#colors=sns.color_palette('Set3', len(total_spending)): Uses Seaborn's color palette to color the pie chart slices.
#This pie chart helps visualize the proportion of spending on each product category.
```

![](https://github.com/OluwaseunOkundalaye/MARKETING-CAMPAIGN-RESULTS/blob/main/Spending%20distribution%20across%20product%20category.png)

The pie chart provides a comprehensive overview of spending habits across various product categories. According to the chart, the majority of spending is allocated to MntWines, which accounts for 50.2% of the total expenditure. This is followed by MntMeatProducts, which constitute 27.6% of the spending. Other categories such as MntFishProducts, MntSweetProducts, MntGoldProds, and MntFruits receive comparatively smaller shares, with percentages of 6.2%, 4.5%, 7.3%, and 4.3%, respectively. 

**Analyze Purchasing Trends Over Time**

``` Python
# Convert 'Dt_Customer' to datetime if it's not already in that format
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'])

# Extract the year and month from the 'Dt_Customer' column for trend analysis
data['Year_Month'] = data['Dt_Customer'].dt.to_period('M')  # Create a 'Year_Month' column

# Calculate the total spending for each month
monthly_spending = data.groupby('Year_Month')[product_columns].sum()

# Plot monthly spending trends for each product category
monthly_spending.plot(figsize=(12, 8), marker='o')
plt.title('Monthly Spending Trends per Product Category')
plt.xlabel('Month')
plt.ylabel('Total Spending')
plt.legend(title='Product Categories', labels=product_columns)
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

#Explanation:
#pd.to_datetime(): Converts the Dt_Customer column (which contains the enrollment date) to the datetime type if it's not already in that format.
#data['Year_Month'] = data['Dt_Customer'].dt.to_period('M'): Creates a new column (Year_Month) to extract the year and month from the Dt_Customer column. This will allow grouping the data by month.
#data.groupby('Year_Month')[product_columns].sum(): Groups the data by the new Year_Month column and calculates the total spending for each product category in each month.
#monthly_spending.plot(): Plots the monthly spending trends for each product category.
#This step allows us to understand how spending on each product category evolves over time, highlighting any seasonal trends.
```

![](https://github.com/OluwaseunOkundalaye/MARKETING-CAMPAIGN-RESULTS/blob/main/Monthly%20spending%20trend%20per%20product%20category.png)

The line graph titled "Monthly Spending Trends per Product Category" showcases the total spending trends for various product categories from July 2013 to April 2014. The x-axis represents the months, while the y-axis denotes the total spending. Each product category, including MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, and MntGoldProds, is represented by a distinct colored line. The graph reveals that MntWines consistently has the highest spending, with noticeable peaks and troughs, indicating significant consumer demand. MntMeatProducts also demonstrate considerable variation but at a lower level compared to MntWines. The other categories exhibit relatively lower and more stable spending trends.

### 6.	Which channels are underperforming?
This question focuses on identifying the sales or distribution channels that are not meeting expected performance metrics such as sales volume, revenue, customer engagement, or profitability. Understanding channel performance is critical for identifying inefficiencies and areas requiring improvement in a company’s marketing, sales, or delivery strategies.
The following were carried out in order to address this analysis;
Compare Total Purchases Across Channels
# Calculate the total number of purchases in each channel
channel_columns = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
total_purchases = data[channel_columns].sum()  # Sum purchases for each channel

print("Total purchases per channel:\n", total_purchases)
Explanation:
data[channel_columns].sum(): Sums up the total purchases for each channel (Web, Catalog, and Store) by summing across the relevant columns (NumWebPurchases, NumCatalogPurchases, NumStorePurchases).
This step helps to identify the overall purchases made through each channel.
Result:
Total purchases per channel:
 NumWebPurchases         9150
NumCatalogPurchases     5963
NumStorePurchases        12970

Calculate the Conversion Rate for the Web Channel
# Assuming 'NumWebVisits' represents the number of visits to the website
data['WebConversionRate'] = data['NumWebPurchases'] / data['NumWebVisitsMonth']  # Calculate the conversion rate

# Calculate the average conversion rate across all customers
average_conversion_rate = data['WebConversionRate'].mean()

print("Average Web Conversion Rate: {:.2f}%".format(average_conversion_rate * 100))
Explanation:
data['WebConversionRate']: Calculates the conversion rate for each customer on the web channel by dividing the number of web purchases (NumWebPurchases) by the number of web visits (NumWebVisits).
data['WebConversionRate'].mean(): Calculates the average conversion rate across all customers. This gives an idea of how effectively the web channel is converting visits into purchases.
Result:
Average Web Conversion Rate: inf%
The outcome of Average Web Conversion Rate: inf% suggests that some values in the NumWebVisitsMonth column are likely zero. When you divide by zero, it results in an infinite value (inf), which is why we're seeing this in the output. To address this, the following was carried out.
# Avoid division by zero by replacing 0 in 'NumWebVisitsMonth' with NaN or a small value
data['WebConversionRate'] = np.where(data['NumWebVisitsMonth'] == 0, np.nan, data['NumWebPurchases'] / data['NumWebVisitsMonth'])

# Calculate the average conversion rate, ignoring NaN values
average_conversion_rate = data['WebConversionRate'].mean()

# Print the average conversion rate as a percentage
print("Average Web Conversion Rate: {:.2f}%".format(average_conversion_rate * 100))
Explanation:
np.where(): This ensures that when NumWebVisitsMonth is zero, the conversion rate is set to NaN rather than causing division by zero errors.
Mean Calculation: The mean() function automatically ignores NaN values, so you won’t be affected by missing or invalid data when calculating the average conversion rate.
Result:
Average Web Conversion Rate: 109.21%

Visualize Total Purchases Across Channels
# Plot the total purchases for each channel using a bar plot
plt.figure(figsize=(10, 6))
total_purchases.plot(kind='bar', color='lightcoral')
plt.title('Total Purchases per Channel')
plt.xlabel('Channel')
plt.ylabel('Total Purchases')
plt.xticks(rotation=45, ha='right')
plt.show()
Explanation:
total_purchases.plot(kind='bar', color='lightcoral'): Creates a bar plot to visualize the total purchases for each channel (Web, Catalog, Store). The color='lightcoral' adds color to the bars.
plt.xticks(rotation=45, ha='right'): Rotates the x-axis labels to improve readability, especially when the labels are long.
This bar plot helps to quickly compare the performance of each channel based on the total number of purchases.
