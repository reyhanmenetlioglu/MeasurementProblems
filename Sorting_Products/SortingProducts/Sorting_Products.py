###################################################
# Sorting Products
###################################################

# region import & read

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("Measurement_Problems/Sorting_Products/SortingProducts/dataset/product_sorting.csv")
print(df.shape)
df.head(10)

# endregion

####################
# Sorting by Rating
####################

# region Sorting by Rating

# When sorting by rating:
# Two factors (the number of purchases and the number of comments) will be overlooked.
# Both of these factors need to be considered.

df.sort_values("rating", ascending=False).head()

# endregion

####################
# Sorting by Comment Count or Purchase Count
####################

# region Sorting by Comment Count or Purchase Count

# There are some deficiencies here, as in ranking by rating.

df.sort_values("purchase_count", ascending=False).head()
df.sort_values("commment_count", ascending=False).head()

# endregion

####################
# Sorting by Rating, Comment and Purchase
####################

# region Sorting by Rating, Comment and Purchase

# The scales of the variables are different from each other, there is a standardization process to be done here.
# That is, the rating variable consists of numbers between 1 and 5
# Therefore, other variables should be converted to variables between 1 and 5.
# We do this standardization process with the MinMaxScaler method.


df['purchase_count_scaled'] = MinMaxScaler(feature_range=(1,5)). \
    fit(df[['purchase_count']]). \
    transform(df[['purchase_count']])

df['comment_count_scaled'] = MinMaxScaler(feature_range=(1,5)). \
    fit(df[['commment_count']]). \  # Fitted, giving scales, drawing the relevant conversion scheme, changes...
    transform(df[['commment_count']])  # What was done in the fit section was transformed in the transform section.

df.describe().T

# Since each of these 3 variables is now on the same scale, we can do weighting.

(df['comment_count_scaled'] * 32 / 100 + \
 df['purchase_count_scaled'] * 26 / 100 + \
 df['rating'] * 42 / 100)

# The value we calculate is the score, not the rating.

# endregion

# region weighted_sorting_score Function

def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    return (dataframe['comment_count_scaled'] * w1 / 100 +
            dataframe['purchase_count_scaled'] * w2 / 100 +
            dataframe['rating'] * w3 / 100)

df['weighted_sorting_score'] = weighted_sorting_score(df)

df.sort_values('weighted_sorting_score', ascending=False).head(20)

df[df["course_name"].str.contains("Veri Bilimi")].sort_values("weighted_sorting_score", ascending=False).head(20)

# We embodied some abstract factors, scaled them, reduced them to the same scale.
# We have made a more precise calculation that can format their effects.
# According to this, we can perform a sorting operation.

# endregion

####################
# Bayesian Average Rating Score
####################

# region bayesian_average_rating Function

# Sorting products with 5 Star Rated
# Sorting Products According to Distribution of 5 Star Rating

# Can we refine the ratings from a different perspective?
# Can we make a ranking by focusing only on rating?
# We will refocus on rating while ranking.
# bayesian_average_rating calculates the weighted probabilistic average using the distribution information of the stars.
# We will get a score with the #bayesian_average_rating function.
# This calculation can also be used as the final average rating of a product as it gives the average value associated with the rating.
# It can also be considered as a score because it shows the existing rates lower than they are.

def bayesian_average_rating(n, confidence=0.95):  # n denotes star values and observation frequencies.
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)  # confidence allows you to get the value of the z table value to be calculated.
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

df.head()

df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                "2_point",
                                                                "3_point",
                                                                "4_point",
                                                                "5_point"]]), axis=1)

df.sort_values("weighted_sorting_score", ascending=False).head(20)
df.sort_values("bar_score", ascending=False).head(20)

df[df["course_name"].index.isin([5, 1])].sort_values("bar_score", ascending=False)  # İki kurs arasında gözlem amaçlı.

# If we only rank with Bayesian Average Rating Score, we will again miss some factors.
# For this reason, we should use a hybrid method.

# Bar Score also gives us a chance to raise promising ones even if they are new in the dataset.
# So, when the Bar Score is considered as a weighting factor in a hybrid model,
# it also pushes up products with higher potential but not yet sufficient social proof

# Actually, we indirectly increase the weight of the rating.
# But Rating only refers to average in previous calculation, but refers to potential in Bar Score calculation.

# endregion

####################
# Hybrid Sorting: BAR Score + Other Factors
####################

# Rating Products
# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating
# - Bayesian Average Rating Score

# Sorting Products
# - Sorting by Rating
# - Sorting by Comment Count or Purchase Count
# - Sorting by Rating, Comment and Purchase
# - Sorting by Bayesian Average Rating Score (Sorting Products with 5 Star Rated)
# - Hybrid Sorting: BAR Score + Other Factors

# region hybrid_sorting_score Function : Hybrid Sorting: BAR Score + Other Factors

def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                     "2_point",
                                                                     "3_point",
                                                                     "4_point",
                                                                     "5_point"]]), axis=1)
    wss_score = weighted_sorting_score(dataframe)

    return bar_score*bar_w/100 + wss_score*wss_w/100


df["hybrid_sorting_score"] = hybrid_sorting_score(df)

df.sort_values("hybrid_sorting_score", ascending=False).head(20)

df[df["course_name"].str.contains("Veri Bilimi")].sort_values("hybrid_sorting_score", ascending=False).head(20)

# endregion