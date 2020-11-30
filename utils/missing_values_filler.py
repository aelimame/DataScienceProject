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
    def fill_missing_values(self, data, to_fill_column_name, to_fill_row_value, label_column_name, k, debug=False):
        if debug: print("\nMissing values filling started ...\n")

        # Create a deep copy
        df = data.copy(deep=True)
        cat_column_name = to_fill_column_name + ' Code'

        if debug: print("Initial value counts:\n{}\n".format(data[to_fill_column_name].value_counts()))

        # Build numerical category
        df[to_fill_column_name] = pd.Categorical(df[to_fill_column_name])
        df[cat_column_name] = df[to_fill_column_name].cat.codes
        cat_map = dict(enumerate(df[to_fill_column_name].cat.categories))
        to_fill_code = [code for code, cat_name in cat_map.items() if cat_name == to_fill_row_value][0]

        if debug: print("Categorical map:\n{}\n".format(cat_map))
        if debug: print("Code to be filled: {}\n".format(to_fill_code))

        # Creating subset df for undersampling
        df = df[[cat_column_name, label_column_name]]
        to_fill_df = df[df[cat_column_name] == to_fill_code] # TODO: Will we need that? The undersampling procedure will mess up indexes..
        under_df = df[df[cat_column_name] != to_fill_code]
        under_df.reset_index(inplace=True)

        if debug: print("Value counts before undersampling:\n{}\n".format(df[cat_column_name].value_counts()))

        ###################
        ## Undersampling ##
        ###################

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

        if debug: print("Value counts after undersampling:\n{}\n".format(under_df[cat_column_name].value_counts()))

        ################
        ## Imputation ##
        ################

        # Build imputation struct
        start_index = under_df.shape[0]
        merged_df = under_df.append(to_fill_df.reset_index(), ignore_index=True, sort=False)
        merged_df[cat_column_name] = merged_df[cat_column_name].replace(to_replace=to_fill_code, value=np.nan)

        if debug: print("Value counts before imputation:\n{}\n".format(merged_df[cat_column_name].value_counts()))

        imputer = KNNImputer(n_neighbors=k, missing_values=np.nan)
        imputed_data = imputer.fit_transform(merged_df)
        imputed_df = pd.DataFrame(imputed_data, columns=merged_df.columns)

        # Rounding neighborhood outputs
        imputed_df[cat_column_name] = imputed_df[cat_column_name].round(0).astype(int)

        # Verify data integrity
        for i, r in merged_df.iterrows():
            if r[label_column_name] != imputed_df.at[i, label_column_name]:
                print("{} : {} --> {}".format(i, r[cat_column_name], r[label_column_name]))
                print("{} : {} --> {}".format(i, imputed_df.at[i, cat_column_name],imputed_df.at[i, label_column_name]))
                raise RuntimeError("Data integrity error when building up imputed dataframe.")

        # Verify data integrity & build imputed dataframe
        assert np.isnan(merged_df.at[start_index, cat_column_name])
        assert np.isnan(merged_df.at[start_index + 1, cat_column_name])
        assert not np.isnan(merged_df.at[start_index - 1, cat_column_name])
        for ((i, r0), (j, r1)) in zip(imputed_df[start_index:].iterrows(), to_fill_df.iterrows()):
            if data.at[j, to_fill_column_name] == to_fill_row_value and not (float(r0[label_column_name]) == float(r1[label_column_name]) == float(data.at[j, label_column_name])):
                print("{} : {} --> {}".format(i, r0[cat_column_name], r0[label_column_name]))
                print("{} : {} --> {}".format(j, r1[cat_column_name], r1[label_column_name]))
                print("{} : {} --> {}".format(i, data.at[j, to_fill_column_name], data.at[j, label_column_name]))
                raise RuntimeError("Data integrity error when building up imputed dataframe.")

            # Build imputed dataframe
            data.at[j, to_fill_column_name] = cat_map[int(r0[cat_column_name])]

        if debug: print("Value counts after imputation:\n{}\n".format(data[to_fill_column_name].value_counts()))

        if debug: print("\nMissing values filling ended ...\n")
        return data

############
### TEST ###
############

# raw_df = pd.read_csv("src_data/train.csv")

# # Merge all unknowns
# raw_df['Profile Category'] = raw_df['Profile Category'].replace(r'^\s*$', 'unknown', regex=True)

# print(raw_df['Profile Category'].value_counts())
# print(raw_df.shape[0])

# mvf = MissingValuesFiller()
# filled_df = mvf.fill_missing_values(raw_df, 'Profile Category', 'unknown', 'Num of Profile Likes', 5)

# print(filled_df['Profile Category'].value_counts())
# print(filled_df.shape[0])