###################################################
# Rating Products
###################################################

# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating

############################################
# Practice: User and Time Weighted Course Score Calculation
############################################

# region import & read

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv('Measurement_Problems/Rating_Products/dataset/course_reviews.csv')

# endregion

# (50+ Hours) Python A-Z™: Data Science & Machine Learning
# Rate: 4.8 (4.764925)
# Total Rate: 4611
# Rate Percentages: 75, 20, 4, 1, <1
# Approximate Numerical Equivalents: 3458, 922, 184, 46, 6

# region analyzing data

df.head()
df.shape  # 4323 reviews available.
df['Rating'].value_counts()  # Distribution of Ratings in the data set (how many in which area).

df['Questions Asked'].value_counts()  # Asking questions distributions.

df.groupby('Questions Asked').agg({'Questions Asked': 'count',
                                   'Rating': 'mean'})  # The number of asking questions & the average of the rates in the question breakdown.

df.head()

# The aim is to calculate the rate awarded for this course

# endregion

####################
# Average
####################

# region Standard Average

df['Rating'].mean()

"""

When an average calculation is made directly in this way, 
then we may miss the recent trend for the relevant product 
or service in terms of customers. So we're missing the satisfaction trend.
For example: 
There may be a very high level of satisfaction in the first 3 months, 
but there may also be some problems with this product in the last three months of a year. 
Therefore, the positive or negative trends that may arise in relation to the presentation 
or handling of the product will lose their effect.
  
-- Instead of just averaging points, what else can be done?
-- What can we do to better reflect the current trend on the average?

"""

# endregion

####################
# Time-Based Weighted Average
####################

# region Converting type of df['Timestamp'] to datetime64[ns]

df.head()
df.info()

"""
Here, we have some structural problem.
The type of Timestamp variable here is given as object.
Since we want to do a time dependent operation, we have to convert Timestamp to time variable.

"""

df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# endregion

# region Creating current_day & df['days']

current_date = pd.to_datetime('2021-02-10 0:0:0')  # We set current_date as the max date in the dataset.

df['days'] = (current_date - df['Timestamp']).dt.days  # Returns how many days ago the comments were made, in days.
df['days'].head()
df['days'].tail()

# How do we access the comments made in the last 30 days on this dataset?
df[df['days'] <= 30]
df[df['days'] <= 30].count()  # number of comments made in the last 30 days.

# endregion

# region Time Based Average

df.loc[df['days'] <= 30, 'Rating'].mean()

df.loc[(df['days'] > 30) & (df['days'] <= 90), 'Rating'].mean()

df.loc[(df['days'] > 90) & (df['days'] <= 180), 'Rating'].mean()

df.loc[(df['days'] > 180), 'Rating'].mean()

# endregion

# region Time-Based Weighted Average

"""
As the data of the past months are examined, it is observed that there is an increase in the satisfaction of this service.
We should focus on these timeframes at different levels and set weights for each timeframe.
We have 4 different time zones. While we care more about one, we want to care less about the other. 
Here, we express our desire to focus on the current comments by giving it the highest point.
We refine the calculation according to time.
"""

df.loc[df['days'] <= 30, 'Rating'].mean() * 28 / 100 + \
df.loc[(df['days'] > 30) & (df['days'] <= 90), 'Rating'].mean() * 26 / 100 + \
df.loc[(df['days'] > 90) & (df['days'] <= 180), 'Rating'].mean() * 24 / 100 + \
df.loc[(df['days'] > 180), 'Rating'].mean() * 22 / 100


# endregion

# region Time Based Weighted Average Function

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df['days'] <= 30, 'Rating'].mean() * w1 / 100 + \
           dataframe.loc[(df['days'] > 30) & (df['days'] <= 90), 'Rating'].mean() * w2 / 100 + \
           dataframe.loc[(df['days'] > 90) & (df['days'] <= 180), 'Rating'].mean() * w3 / 100 + \
           dataframe.loc[(df['days'] > 180), 'Rating'].mean() * w4 / 100


time_based_weighted_average(df)

time_based_weighted_average(df, 30, 26, 22, 22)

# endregion

####################
# User-Based Weighted Average
####################

# region User-Based Weighted Average

# Question : Should everyone's score have the same weight?

# Weighting of Rates according to the rate of users watching the course.

df.groupby('Progress').agg({'Rating': 'mean'})

df.loc[df['Progress'] <= 10, 'Rating'].mean() * 22 / 100 + \
df.loc[(df['Progress'] > 10) & (df['days'] <= 45), 'Rating'].mean() * 24 / 100 + \
df.loc[(df['Progress'] > 40) & (df['days'] <= 75), 'Rating'].mean() * 26 / 100 + \
df.loc[(df['Progress'] > 75), 'Rating'].mean() * 28 / 100

# endregion

# region User-Based Weighted Average Function

def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return dataframe.loc[df['Progress'] <= 10, 'Rating'].mean() * w1 / 100 + \
           dataframe.loc[(df['Progress'] > 10) & (df['days'] <= 45), 'Rating'].mean() * w2 / 100 + \
           dataframe.loc[(df['Progress'] > 40) & (df['days'] <= 75), 'Rating'].mean() * w3 / 100 + \
           dataframe.loc[(df['Progress'] > 75), 'Rating'].mean() * w4 / 100


user_based_weighted_average(df, 20, 24, 26, 30)

# endregion

####################
# Weighted Rating
####################

# region Weighted Rating

"""
By combining our Time-Based and User-Based calculations, 
we will perform calculations using a single function.
Considering these metrics, we make a refinement.
By bringing the two calculations together, we realize an average calculation
that is more reliable, more accurate, and includes more factors.
"""

def course_weighted_rating(dataframe, time_w=50, user_w=50):
    return time_based_weighted_average(dataframe) * time_w / 100 + user_based_weighted_average(dataframe) * user_w / 100


course_weighted_rating(df)

course_weighted_rating(df, time_w=40, user_w=60)

# endregion
