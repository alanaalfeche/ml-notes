import pandas as pd

### Creating data
'''
There are two core objects in pandas: DataFrame and Series

DataFrame is a table, containing an array of individual entries, each of which has a certain value. 
Each entry corresponds to a row (or record) in the table and a column. 

Series is a sequence of data value - a list. It doesn't have a column name, but it has an 'overall name'.
In an essence, Series is a single column of a DataFrame, and a DataFrame is collection of Series "glued together."

The list of row labels used in a DataFrame is called an Index, which can be assigned using the index parameter.
'''
# DataFrame examples
# ...as dictionary
pd.DataFrame({
    'Bob': ['I liked it.', 'It was awful.'],
    'Sue': ['Pretty good.', 'Bland.']},
    index=['Product A', 'Product B'])

# ...as list
pd.DataFrame([[30, 21]], columns=['Apples', 'Bananas'])

# Series examples
pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'], name='Product A')

# or...
sales = [30, 35, 40]
year = ['2015 Sales', '2016 Sales', '2017 Sales']
recipe = pd.Series(sales, index=year, name='Product ')

### Read data into a DataFrame
wine_reviews = pd.read_csv(file_path)

# or - Read data into a DataFrame with specified index
wine_reviews = pd.read_csv(file_path, index_col=0)

# Determine the number of rows and columns
wine_reviews.shape

# Return the first five rows
wine_reviews.head()

### Write DataFrame to csv
df.to_csv(file)path

### Indexing data
'''
Two ways of selecting a specific Series out of a DataFrame
1. reviews.country
2. reviews['country']

An advantage of [] operator is it can do the following: rewiews['country']['providence'].
This is not plausible using the (.) operator. 

For more advanced operations, use iloc and loc operators. 
Both follows row-first, column-second which is opposite to native Python. 

iloc views a dataset as a big matrix (a list of lists).
By contrast, loc uses the indices to navigate a dataset (often easier).

iloc uses Python stdlib indexing scheme where 0:10 returns 0,...,9
loc indexes inclusively where 0:10 returns 0,...,10
'''

# iloc for index-based selection 
wine_reviews.iloc[0] # returns first row of data in a DataFrame
wine_reviews.iloc[:,0] # returns all values in the first column of data in a DataFrame - (:) operator returns everything
wine_reviews.iloc[1:3, 0] # returns the second and third entries of the first column
wine_reviews.iloc[[1, 2, 3], 2] # returns first, second, and third entries form the third column
wine_reviews.iloc[-5:] # returns the last five row of data in a DataFrame

# loc for label-based selection
wine_reviews.loc[0, 'country'] # returns the first country in a DataFrame
wine_reviews.loc[[1, 3, 4]] # returns entries indexed as 1, 3, and 4
wine_reviews.loc[:, ['taster_name', 'points']] # returns all entries in the two specified columns

# for iloc and loc, chaining selection is also possible
# both returns the first country in the country column
wine_reviews.country.loc[0]
wine_reviews.country.iloc[0]

# set index
wine_reviews.set_index("title")

### Conditional Selection
wine_reviews.loc[wine_reviews.country == 'Italy'] # returns all records where country is Italy
wine_reviews.loc[(wine_reviews.country == 'Italy') & (wine_reviews.points >= 90)]
wine_reviews.loc[wine_reviews.country.isin(['Italy', 'France'])]
wine_reviews.loc[wine_reviews.price.notnull()]

### Assigning Data
wine_reviews['critic'] = 'everyone' # assigning constant value
wine_reviews['index_backwards'] = range(len(wine_reviews), 0, -1) # assigning with iterable values

### Summary Functions
# numerical data
wine_reviews.points.describe() # returns count, mean, 75%, and max
wine_reviews.points.mean() # returns average

# string data
wine_reviews.taster_name.describe() # returns count, unique, top, frequency
wine_reviews.taster_name.unique() # returns list of unique values from a column 
wine_reviews.taster_name.value_counts() # returns list of unique and how often they occur in the dataset

### Maps
'''
In mathematic, "map" is a term used for taking a set of values which is then "mapped" to another set of values
There are two mapping methods available: map() and apply()

Both returns new Series and transformed DataFrames - does not modify original data.
'''
wine_reviews_mean = wine_reviews.points.mean()

# map expects a single value from the Series and return a transformed version of that value 
wine_reviews.points.map(lambda p: p - wine_reviews_mean)

# in contrary, apply can transform a whole DataFrame 
def remean_points(row):
    row.points = row.points - wine_reviews_mean
    return row

wine_reviews.apply(remean_points, axis='columns') # axis='index' transforms each column

# Pando also has built-in speedy operators that can do simple operations
wine_reviews.points - wine_reviews_mean # returns a new Series with mean subtracted from points column
wine_reviews.country + " - " + wine_reviews.region # returns a new Series with string from two columns joined with '-'
bargain_idx = (wine_reviews.points / wine_reviews.price).idxmax() # idx returns the index of the row
bargain_wine = reviews.loc[bargain_idx, 'title'] # returns the wine name which is the most bargain

# Cool use of map to find the frequency count of a word in wine description
n_trop = wine_reviews.description.map(lambda desc: "tropical" in desc).sum()
n_fruity = wine_reviews.description.map(lambda desc: "fruity" in desc).sum()
descriptor_counts = pd.Series([n_trop, n_fruity], index=['tropical', 'fruity'])

### Grouping
# for single-label index
wine_reviews.groupby('points').points.count() # groupby() equivalent of value_counts() 
wine_reviews.groupby('points').size() # returns size of group without specified index 
wine_reviews.groupby('winery').apply(lambda df: df.title.iloc[0]) # returns first wine reviewed from each winery
wine_reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()]) # returns the best wine by country and provine
wine_reviews.groupby('country').price.agg([len, min, max]) # returns a statistical summary of the dataset

# for multi-label indexes
countries_reviewed = wine_reviews.groupby(['country', 'province']).description.agg([len])
type(countries_reviewed.index) # returns pandas.core.indexes.multi.MultiIndex
countries_reviewed.reset_index() # convert MultiIndex to regular index 

### Sorting
# groupby returns data in index order, not in value order 
countries_reviewed.sort_values(by='len') # returned sorted data by values
countries_reviewed.sort_values(by='len', ascending=False) # returnes sorted data in descending order
# returned sorted data by more than one column at a time
# this sorts by length then country
countries_reviewed.sort_values(by=['len', 'country']) 
countries_reviewed.sort_index() # return sorted data by index

# A bizarre example
best_rating_per_price = reviews.groupby('price')['points'].max().sort_index()

### Data Types 
wine_reviews.price.dtype # returns data type of the column
wine_reviews.dtype # returns all data types of every column in a data frame
wine_reviews.points.astype('str') # convert column into another data type
# Note that dtype on string returns an object data type

### Missing Values
# Note that missing values has value NaN and are of floay64 dtype 
# returns entries with country as NaN
wine_reviews[pd.isnull(reviews.country)] 
wine_reviews[wine_reviews.price.isnull()]
wine_reviews.price.isnull().sum() # my preference 
pd.isnull(wine_reviews.price).sum()

# replace values
wine_reviews.country.fillna("Unknown") # replace missing values
wine_reviews.reviewer.replace("oldname", "newname") # replace old values

# Good example of function interpolation
wine_reviews_per_region = wine_reviews.region_1.fillna('Unknown').value_counts().sort_values(ascending=False)

### Renaming
# renames lets you rename index (row and column) or columms of a DataFrame
wine_reviews.rename(columns={'points': 'score'})
wine_reviews.rename(index={0: 'first', 1: 'second'})
wine_reviews.rename_axis('wines', axis='rows').rename_axis('fields', axis='columns')

### Combining
# The are three ways to combine DataFrames and/or Series (in increasing of complexity): concat, join, and merge
# concat() requires datasets to have the same columns. 
pd.concat([dataset1, dataset2])

# join() lets you combine two different datasets which have index in commond
left = canadian_youtube.set_index(['title', 'trending_date'])
right = british_youtube.set_index(['title', 'trending_date'])
left.join(right, lsuffix='_CAN', rsuffix='_UK') # l/r-suffix is necessary because both datasets have the same column names
