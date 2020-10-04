# # Get our environment set up 
# The first thing we'll need to do is load in the libraries and dataset we'll be using. We'll be working with a dataset containing information on earthquakes that occured between 1965 and 2016.

import pandas as pd
import numpy as np
import seaborn as sns
import datetime

# read in our data
earthquakes = pd.read_csv("../input/earthquake-database/database.csv")
np.random.seed(0)

# # 1) Check the data type of our date column
# You'll be working with the "Date" column from the `earthquakes` dataframe.  Investigate this column now: does it look like it contains dates?  What is the dtype of the column?

earthquakes['Date'] # returns Name: Date, Length: 23412, dtype: object

# # 2) Convert our date columns to datetime
# Most of the entries in the "Date" column follow the same format: "month/day/four-digit year".  However, there are that follows a completely different pattern.

date_lengths = earthquakes.Date.str.len()
date_lengths.value_counts() # returns 10 and 24
indices = np.where([date_lengths == 24])[1]
print('Indices with corrupted data:', indices)
earthquakes.loc[indices]

# create a new column "date_parsed" in the `earthquakes` dataset that has correctly parsed dates in it.  
earthquakes.loc[3378, "Date"] = "02/23/1975"
earthquakes.loc[7512, "Date"] = "04/28/1985"
earthquakes.loc[20650, "Date"] = "03/13/2011"
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format="%m/%d/%Y")

# # 3) Select the day of the month
# Create a Pandas Series `day_of_month_earthquakes` containing the day of the month from the "date_parsed" column.

day_of_month_earthquakes = earthquakes.date_parsed.dt.day

# # 4) Plot the day of the month to check the date parsing
# Plot the days of the month from your earthquake dataset.

# remove na's
day_of_month_landslides = day_of_month_earthquakes.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)