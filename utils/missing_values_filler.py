#### FOR MAC OSX USERS
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import KNNImputer
from imblearn.under_sampling import RandomUnderSampler

class MissingValuesFiller():
    def fill_missing_values(self, data, to_fill_column_name, to_fill_row_value, label_column_name):
        print("\nMissing values filling started ...\n")

        # Create a deep copy
        df = data.copy(deep=True)
        cat_column_name = to_fill_column_name + ' Code'

        print("Initial value counts:\n{}\n".format(data[to_fill_column_name].value_counts()))

        # Build numerical category
        df[to_fill_column_name] = pd.Categorical(df[to_fill_column_name])
        df[cat_column_name] = df[to_fill_column_name].cat.codes
        cat_map = dict(enumerate(df[to_fill_column_name].cat.categories))
        to_fill_code = [code for code, cat_name in cat_map.items() if cat_name == to_fill_row_value][0]

        print("Categorical map:\n{}\n".format(cat_map))
        print("Code to be filled: {}\n".format(to_fill_code))

        # Creating subset df for undersampling
        df = df[[cat_column_name, label_column_name]]
        to_fill_df = df[df[cat_column_name] == to_fill_code] # TODO: Will we need that? The undersampling procedure will mess up indexes..
        under_df = df[df[cat_column_name] != to_fill_code]
        under_df.reset_index(inplace=True)

        print("Value counts before undersampling:\n{}\n".format(df[cat_column_name].value_counts()))

        # Build data struc fo undersampling
        undersampler = RandomUnderSampler(sampling_strategy='majority')
        x = [under_df[cat_column_name].tolist()]
        x = np.array(x).T
        y = under_df[label_column_name].tolist()

        # Verify data integrity
        for i, r in under_df.iterrows():
            if r[cat_column_name] != x[i][0] or r[label_column_name] != y[i]:
                print("{} : {} --> {}".format(i, r[cat_column_name], r[label_column_name]))
                print("{} : {} --> {}".format(i, x[i][0], y[i]))
                raise RuntimeError("Data integrity error when building up undersampling dataframe.")

        # Undersampling
        under_x, under_y = undersampler.fit_resample(x, y)

        #  Build undersampled dataframe
        under_x = list(np.concatenate(under_x.T).flat)
        under_df = pd.DataFrame(columns=[cat_column_name, label_column_name])
        under_df[cat_column_name] = under_x
        under_df[label_column_name] = under_y

        print("Value counts after undersampling:\n{}\n".format(under_df[cat_column_name].value_counts()))

        print("\nMissing values filling ended ...\n")
        return None

############
### TEST ###
############

raw_df = pd.read_csv("src_data/train.csv")

# Merge all unknowns
raw_df['Profile Category'] = raw_df['Profile Category'].replace(r'^\s*$', 'unknown', regex=True)

mvf = MissingValuesFiller()
print(mvf.fill_missing_values(raw_df, 'Profile Category', 'unknown', 'Num of Profile Likes'))

# nu_df = profile_df[profile_df['Profile Category Code'] != 3]
# print("Values counts (before):\n{}".format(nu_df['Profile Category Code'].value_counts()))
# undersampler = RandomUnderSampler(sampling_strategy='not minority')
# category_codes = [nu_df['Profile Category Code'].tolist()]
# category_codes = np.array(category_codes).T
# likes = nu_df['Num of Profile Likes'].tolist()
# category_codes_under, likes_under = undersampler.fit_resample(category_codes, likes)
# category_codes_under = list(np.concatenate(category_codes_under.T).flat)
# under_profile_df = pd.DataFrame(columns=["Profile Category Code","Num of Profile Likes"])
# under_profile_df["Profile Category Code"] = category_codes_under
# under_profile_df["Num of Profile Likes"] = likes_under
# print("Values counts (after):\n{}".format(under_profile_df['Profile Category Code'].value_counts()))