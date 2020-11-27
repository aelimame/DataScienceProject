import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import median, mean

def convert_string_column_to_boolean_column(df, column_name, true_values, false_values):
    df[column_name] = df[column_name].apply(lambda val: True if val in true_values else (False if val in false_values else None))
    return df

# Load dataset
df = pd.read_csv("src_data/train.csv")

print("dtypes:")
print(df.dtypes)

print("sample:")
print(df.iloc[0])

# Exploration of Location Public Visibility
print("Unique value counts (pre-clean): \n {}".format(df['Location Public Visibility'].value_counts()))

# Clean field
df = convert_string_column_to_boolean_column(df, 'Location Public Visibility', ['Enabled', 'enabled'], ['Disabled', 'disabled'])
print("Unique value counts (post-clean): \n {}".format(df['Location Public Visibility'].value_counts()))


ax = sns.barplot(x="Location Public Visibility", y="Num of Profile Likes", data=df, estimator=mean)
plt.show()

ax = sns.barplot(x="Location Public Visibility", y="Num of Profile Likes", data=df, estimator=median)
plt.show()