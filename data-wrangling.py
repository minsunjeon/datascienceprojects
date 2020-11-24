#!/usr/bin/env python
# coding: utf-8

# 
# <h1 align=center><font size=5>Data Analysis with Python</font></h1>

# <h1>Data Wrangling</h1>

# <h3>What is the fuel consumption (L/100k) rate for the diesel car?</h3>

# <h3>Import data</h3>
# <p>
# Import the "Automobile Data Set" from the following link: <a href="https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data">https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data</a>. 
# 
# </p>

# <h4>Import pandas</h4> 

# In[1]:


import pandas as pd
import matplotlib.pylab as plt


# <h2>Reading the data set from the URL and adding the related headers.</h2>

# URL of the dataset

# This dataset was hosted on IBM Cloud object 

# In[ ]:


filename = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"


#  Python list <b>headers</b> containing name of headers 

# In[ ]:


headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]


# Use the Pandas method <b>read_csv()</b> to load the data from the web address. Set the parameter  "names" equal to the Python list "headers".

# In[ ]:


df = pd.read_csv(filename, names = headers)


#  Use the method <b>head()</b> to display the first five rows of the dataframe. 

# In[ ]:



df.head()


# <h2 id="identify_handle_missing_values">Identify and handle missing values</h2>
# 
# 
# <h3 id="identify_missing_values">Identify missing values</h3>
# <h4>Convert "?" to NaN</h4>
# In the car dataset, missing data comes with the question mark "?".
# Replace "?" with NaN (Not a Number), which is Python's default missing value marker, for reasons of computational speed and convenience. Use the function: 
#  <pre>.replace(A, B, inplace = True) </pre>
# to replace A by B

# In[ ]:


import numpy as np

# replace "?" to NaN
df.replace("?", np.nan, inplace = True)
df.head(5)


# 
# 
# <h4>Evaluating for Missing Data</h4>
# 
# The missing values are converted to Python's default. Use Python's built-in functions to identify these missing values. There are two methods to detect missing data:
# <ol>
#     <li><b>.isnull()</b></li>
#     <li><b>.notnull()</b></li>
# </ol>
# The output is a boolean value indicating whether the value that is passed into the argument is in fact missing data.

# In[ ]:


missing_data = df.isnull()
missing_data.head(5)


# "True" stands for missing value, while "False" stands for not missing value.

# <h4>Count missing values in each column</h4>
# <p>
# Using a for loop in Python, quickly figure out the number of missing values in each column. As mentioned above, "True" represents a missing value, "False"  means the value is present in the dataset.  In the body of the for loop the method  ".value_counts()"  counts the number of "True" values. 
# </p>

# In[ ]:


for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")    


# Based on the summary above, each column has 205 rows of data, seven columns containing missing data:
# <ol>
#     <li>"normalized-losses": 41 missing data</li>
#     <li>"num-of-doors": 2 missing data</li>
#     <li>"bore": 4 missing data</li>
#     <li>"stroke" : 4 missing data</li>
#     <li>"horsepower": 2 missing data</li>
#     <li>"peak-rpm": 2 missing data</li>
#     <li>"price": 4 missing data</li>
# </ol>

# Whole columns should be dropped only if most entries in the column are empty. In our dataset, none of the columns are empty enough to drop entirely.
# There is some freedom in choosing which method to replace data; however, some methods may seem more reasonable than others. Each of the following method will be applied to many different columns:
# 
# <b>Replace by mean:</b>
# <ul>
#     <li>"normalized-losses": 41 missing data, replace them with mean</li>
#     <li>"stroke": 4 missing data, replace them with mean</li>
#     <li>"bore": 4 missing data, replace them with mean</li>
#     <li>"horsepower": 2 missing data, replace them with mean</li>
#     <li>"peak-rpm": 2 missing data, replace them with mean</li>
# </ul>
# 
# <b>Replace by frequency:</b>
# <ul>
#     <li>"num-of-doors": 2 missing data, replace them with "four". 
#         <ul>
#             <li>Reason: 84% sedans is four doors. Since four doors is most frequent, it is most likely to occur</li>
#         </ul>
#     </li>
# </ul>
# 
# <b>Drop the whole row:</b>
# <ul>
#     <li>"price": 4 missing data, simply delete the whole row
#         <ul>
#             <li>Reason: price is what we want to predict. Any data entry without price data cannot be used for prediction; therefore any row now without price data is not useful to us</li>
#         </ul>
#     </li>
# </ul>

# <h4>Calculate the average of the column </h4>

# In[ ]:


avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)


# <h4>Replace "NaN" by mean value in "normalized-losses" column</h4>

# In[ ]:


df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)


# <h4>Calculate the mean value for 'bore' column</h4>

# In[ ]:


avg_bore=df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)


# <h4>Replace NaN by mean value</h4>

# In[ ]:


df["bore"].replace(np.nan, avg_bore, inplace=True)


# <h4>Replace NaN in "stroke" column by mean value</h4>

# In[ ]:


avg_stroke = df["stroke"].astype("float").mean(axis = 0)
print("Average of stroke:", avg_stroke)

# replace NaN by mean value in "stroke" column
df["stroke"].replace(np.nan, avg_stroke, inplace = True)


# <h4>Calculate the mean value for the  'horsepower' column:</h4>

# In[ ]:


avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
print("Average horsepower:", avg_horsepower)


# <h4>Replace "NaN" by mean value:</h4>

# In[ ]:


df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)


# <h4>Calculate the mean value for 'peak-rpm' column:</h4>

# In[ ]:


avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
print("Average peak rpm:", avg_peakrpm)


# <h4>Replace NaN by mean value:</h4>

# In[ ]:


df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)


# To see which values are present in a particular column, use the ".value_counts()" method:

# In[ ]:


df['num-of-doors'].value_counts()


# Four doors are the most common type. Use the ".idxmax()" method to calculate for us the most common type automatically:

# In[ ]:


df['num-of-doors'].value_counts().idxmax()


# In[ ]:


#replace the missing 'num-of-doors' values by the most frequent 
df["num-of-doors"].replace(np.nan, "four", inplace=True)


# Finally, drop all rows that do not have price data:

# In[ ]:


# simply drop whole row with NaN in "price" column
df.dropna(subset=["price"], axis=0, inplace=True)

# reset index, because we droped two rows
df.reset_index(drop=True, inplace=True)


# In[ ]:


df.head()


# Now, the dataset has no missing values.

# <h3 id="correct_data_format">Correct data format</h3>
# <p>The last step in data cleaning is checking and making sure that all data is in the correct format (int, float, text or other).</p>
# 
# In Pandas, these methods are used:
# <p><b>.dtype()</b> to check the data type</p>
# <p><b>.astype()</b> to change the data type</p>

# <h4>List the data types for each column</h4>

# In[ ]:


df.dtypes


# <p>Some columns are not of the correct data type. Numerical variables should have type 'float' or 'int', and variables with strings such as categories should have type 'object'. For example, 'bore' and 'stroke' variables are numerical values that describe the engines, so their data types are expected to be of the type 'float' or 'int'; however, they are shown as type 'object'. Convert data types into a proper format for each column using the "astype()" method.</p> 

# <h4>Convert data types to proper format</h4>

# In[ ]:


df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")


# <h4>List the columns after the conversion</h4>

# In[ ]:


df.dtypes


# 
# 
# Now, dataset is cleaned with no missing values and all data in its proper format.

# <h2 id="data_standardization">Data Standardization</h2>
# <p>
# Data is usually collected from different agencies with different formats.Transform the data into a common format which allows for meaningful comparison.
# 
# 

# <p>The formula for unit conversion is:<p>
# L/100km = 235 / mpg

# In[ ]:


df.head()


# In[ ]:


# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]

# check your transformed data 
df.head()


# Transform mpg to L/100km in the column of "highway-mpg", and change the name of column to "highway-L/100km".

# In[ ]:


# transform mpg to L/100km by mathematical operation (235 divided by mpg)
df["highway-mpg"] = 235/df["highway-mpg"]

# rename column name from "highway-mpg" to "highway-L/100km"
df.rename(columns={'"highway-mpg"':'highway-L/100km'}, inplace=True)

# check your transformed data 
df.head()


# Double-click <b>here</b> for the solution.
# 
# <!-- The answer is below:
# 
# # transform mpg to L/100km by mathematical operation (235 divided by mpg)
# df["highway-mpg"] = 235/df["highway-mpg"]
# 
# # rename column name from "highway-mpg" to "highway-L/100km"
# df.rename(columns={'"highway-mpg"':'highway-L/100km'}, inplace=True)
# 
# # check your transformed data 
# df.head()
# 
# -->
# 

# <h2 id="data_normalization">Data Normalization</h2>
# Transforming values of several variables into a similar range. Typical normalizations include scaling the variable so the variable average is 0, scaling the variable so the variance is 1, or scaling variable so the variable values range from 0 to 1.
# 

# In[ ]:


# replace (original value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()


# Normalize the column "height".

# In[ ]:


df['height'] = df['height']/df['height'].max() 
# show the scaled columns
df[["length","width","height"]].head()


# "length", "width" and "height" were normalized in the range of [0,1].

# <h2 id="binning">Binning</h2>
# Transforming continuous numerical variables into discrete categorical 'bins', for grouped analysis.
# Use the Pandas method 'cut' to segment the 'horsepower' column into 3 bins.
# 
# 

#  Convert data to correct format 

# In[ ]:


df["horsepower"]=df["horsepower"].astype(int, copy=True)


# Plot the histogram of horspower, to see what the distribution of horsepower looks like.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"])

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# <p>Create 3 bins of equal size bandwidth. Use numpy's <code>linspace(start_value, end_value, numbers_generated</code> function.</p>
# <p>Set start_value=min(df["horsepower"]) to include the minimum value of horsepower.</p>
# <p>Set end_value=max(df["horsepower"])to include the maximum value of horsepower .</p>
# <p>Since we are building 3 bins of equal length, there should be 4 dividers, so numbers_generated=4.</p>

# Build a bin array, with a minimum value to a maximum value, with bandwidth calculated above. The bins will be values used to determine when one bin ends and another begins.

# In[ ]:


bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
bins


# Set group  names:

# In[ ]:


group_names = ['Low', 'Medium', 'High']


# Apply the function "cut" the determine what each value of "df['horsepower']" belongs to. 

# In[ ]:


df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
df[['horsepower','horsepower-binned']].head(20)


# See the number of vehicles in each bin.

# In[ ]:


df["horsepower-binned"].value_counts()


# Plot the distribution of each bin.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# <p>
#     The last column provides the bins for "horsepower" with 3 categories ("Low","Medium" and "High"). 
# </p>
# <p>
#    The intervals were successfully narrowed down from 57 to 3!
# </p>

# <h3>Bins visualization</h3>
# Normally, a histogram is used to visualize the distribution of bins created above. 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot

a = (0,1,2)

# draw historgram of attribute "horsepower" with bins = 3
plt.pyplot.hist(df["horsepower"], bins = 3)

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# The plot above shows the binning result for attribute "horsepower". 

# <h2 id="indicator">Indicator variable (or dummy variable)</h2>
# <b>What is an indicator variable?</b>
# <p>
#     An indicator variable (or dummy variable) is a numerical variable used to label categories. They are called 'dummies' because the numbers themselves don't have inherent meaning. 
# </p>
# 
# <b>Why use indicator variables?</b>
# <p>
#     So we can use categorical variables for regression analysis in the later modules.
# </p>
# <b>Example</b>
# <p>
#     We see the column "fuel-type" has two unique values, "gas" or "diesel". Regression doesn't understand words, only numbers. To use this attribute in regression analysis, we convert "fuel-type" into indicator variables.
# </p>
# 
# <p>
#     Use the panda's method 'get_dummies' to assign numerical values to different categories of fuel type. 
# </p>

# In[ ]:


df.columns


# Get indicator variables and assign it to data frame "dummy_variable_1" 

# In[ ]:


dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()


# Change column names for clarity 

# In[ ]:


dummy_variable_1.rename(columns={'fuel-type-diesel':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)
dummy_variable_1.head()


# We now have the value 0 to represent "gas" and 1 to represent "diesel" in the column "fuel-type". We will now insert this column back into our original dataset. 

# In[ ]:


# merge data frame "df" and "dummy_variable_1" 
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)


# In[ ]:


df.head()


# The last two columns are now the indicator variable representation of the fuel-type variable. It's all 0s and 1s now.

# Create indicator variable to the column of "aspiration": "std" to 0, while "turbo" to 1.

# In[ ]:



dummy_variable_2=pd.get_dummies(df["aspiration"])
dummy_variable_2.rename(columns={'std':'aspiration-std','turbo':'aspiration-turbo'},inplace=True)
dummy_variable_2.head()


# Merge the new dataframe to the original dataframe then drop the column 'aspiration

# In[ ]:



df=pd.concat([df,dummy_variable_2],axis=1)
df.drop('aspiration',axis=1,inplace=True)


# Save the new csv 

# In[ ]:


df.to_csv('clean_df.csv')

