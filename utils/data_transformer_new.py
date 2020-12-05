from time import strptime
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline

# TODO: Maybe remove this and use SimpleImputer?
from utils.missing_values_filler import MissingValuesFiller
mvf = MissingValuesFiller()

ACCOUNT_AGE_SINCE_DATE = dt.datetime(2021, 1, 1) # Jan 1st, 2021 is the date we measure account "age" from
DEFAULT_NUM_LANGUAGES_TO_FEATUREIZE = 9

# TODO Hardcoded outliers limits (Based on manuel data analysis for the moment)
mum_profile_likes_upper_limit = 200000
num_followers_lower_limit = 0
num_followers_upper_limit = 45000000
num_people_following_lower_limit = 0
num_people_following_upper_limit = 550000
num_status_updates_lower_limit = 0
num_status_updates_upper_limit = 1500000
num_direct_messages_lower_limit = 0
num_direct_messages_upper_limit=75000
avg_daily_profile_clicks_lower_limit = 0
avg_daily_profile_clicks_upper_limit = 17



#Custom Transformer that modifies colour columns
class ColorsTransformer( BaseEstimator, TransformerMixin ):
    def __init__( self, feature_names ):
        self._feature_names = feature_names

    def split_colour_column(self, column_name):
        def parse_color(two_char_code):
            if not two_char_code or (two_char_code[0] == 'n'):
                return 0
            return int(two_char_code, 16) / 255

        colours_rgb = X[column_name]
        X[column_name+'_r'] = colours_rgb.apply(lambda colour: parse_color(str(colour)[0:2]) )
        X[column_name+'_g'] = colours_rgb.apply(lambda colour: parse_color(str(colour)[2:4]) )
        X[column_name+'_b'] = colours_rgb.apply(lambda colour: parse_color(str(colour)[4:6]) )
        return self.df

    def fit( self, X, y = None ):
        return self

    def transform( self, X, y = None ):
        for column_name in self._feature_names:
            self.split_colour_column(column_name)
            X.drop(column_name, axis=1, inplace=True)
        return X.values

#Languages Transformer that modifies colour columns
class LanguagesTransformer( BaseEstimator, TransformerMixin ):
    def __init__( self, feature_names, num_languages_to_featureize = DEFAULT_NUM_LANGUAGES_TO_FEATUREIZE ):
        self._feature_names = feature_names

    def fit( self, X, y = None ):
        # Drop any country code (e.g. "en-gb" -> "en")
        X[self._feature_names] = X[self._feature_names].apply(lambda strVal: strVal[0:2].lower() )
        # Find the top n languages
        self._top_n_languages = X[self._feature_names].value_counts(normalize=True).keys().values[0:n]
        return self

    def transform( self, X, y = None ):
        # Drop any country code (e.g. "en-gb" -> "en")
        X[self._feature_names] = X[self._feature_names].apply(lambda strVal: strVal[0:2].lower() )
         # If User Language is other than one of the top n languages, replace it with "other"
        X[self._feature_names] = X[self._feature_names].apply(lambda strVal: strVal[0:2] if ( strVal[0:2] in self._top_n_languages ) else "other" )
        # One-hot vector for each of top N languages
        one_hot_top_n_languages = pd.get_dummies(X[self._feature_names], prefix='Language')
        X.drop(self._feature_names, axis=1, inplace=True)
        X = X.join(one_hot_top_n_languages)
        return X.values

#Custom Transformer that modifies location columns
class LocationsTransformer( BaseEstimator, TransformerMixin ):
    def __init__( self, feature_names ):
        self._feature_names = feature_names

    def fit( self, X, y = None ):
        return self

    def transform( self, X, y = None ):
        # X['Has Location'] = self.df['Location'].apply(lambda location: False if pd.isnull(location) else True )
        # TODO: Use geocoding results to add more info?
        X.drop('Location', axis=1, inplace=True)

        # We are going to bin into one-hour timezones, and one-hot
        # Maybe try this to fill missing values?
        # X = self.mvf.fill_missing_values(X, 'UTC Offset', NaN, 'Num of Profile Likes', 5)

        # Using a random number between -8 UTC and + 10 UTC because that encompases most of the
        # land mass of the Earth.
        # X['UTC Offset'] = X['UTC Offset'].fillna( np.random.randint( -8, 11 )*60*60 )
        # X['UTC Offset'] = X['UTC Offset'].fillna(UTC_FOR_NA)

        # floor so we group 1/2 hour offsets
        # X['UTC Offset'] = np.floor((X['UTC Offset']/60/60)).astype(int)

        #one_hot_categories = pd.get_dummies(X['UTC Offset'], prefix='UTC Offset')
        X.drop('UTC Offset', axis=1, inplace=True)
        #X = X.join(one_hot_categories)

        # Ensure all the important columns are there
        #        for i in np.arange(-12, 13):
        #            if 'UTC Offset_'+str(i) not in X:
        #               X['UTC Offset_'+str(i)] = 0

        X.drop('User Time Zone', axis=1, inplace=True)

        # This column uses "Enabled" and "Disabled" (and "??") rather than "True" and "False"
        X['Location Public Visibility'] = X['Location Public Visibility'].apply(lambda strVal: str(strVal).lower() == 'enabled' ).astype(bool).astype(int)

        return X.values

#Custom Transformer that modifies textual columns
class TextTransformer( BaseEstimator, TransformerMixin ):
    def __init__( self, feature_names ):
        self._feature_names = feature_names

    def fit( self, X, y = None ):
        return self

    def transform( self, X, y = None ):
        for feature_name in self._feature_names:
            X['Has '+feature_name] = X[feature_name].apply(lambda urlVal: 1 if str(urlVal) else 0 )
            X['Has '+feature_name].fillna(0, inplace=True)
            X.drop(feature_name, axis=1, inplace=True)
        return X.values

#Custom Transformer that modifies our date time columns
class DateTimeTransformer( BaseEstimator, TransformerMixin ):
    def __init__( self, feature_names ):
        self._feature_names = feature_names

    def fit( self, X, y = None ):
        return self

    def transform( self, X, y = None ):
        # Our dates come in the format:
        # Wed Jul 20 07:46:18 +0000 2011
        def days_since_fixed_date(datetime_str):
            dt_split = datetime_str.split(' ')

            year = int(dt_split[5])
            month = strptime(dt_split[1],'%b').tm_mon
            day = int(dt_split[2])

            times = dt_split[3].split(':')
            hour = int(times[0])
            mins = int(times[1])
            secs = int(times[2])

            thedate = dt.datetime(year, month, day, hour, mins, secs)
            difference = ACCOUNT_AGE_SINCE_DATE - thedate
            return difference.total_seconds() / dt.timedelta(days=1).total_seconds()

        X['Account Age Days'] = X['Profile Creation Timestamp'].apply(lambda x: days_since_fixed_date(x) )
        X.drop('Profile Creation Timestamp', axis=1, inplace=True)
        return X.values

#Custom Transformer that modifies our categorical columns
class CategoricalTransformer( BaseEstimator, TransformerMixin ):
    def __init__( self, feature_names ):
        self._feature_names = feature_names

    def fit( self, X, y = None ):
        # On Fit, we also want to try to impute values of Profile Category based on Num of Status Updates
        X['Profile Category'] = X['Profile Category'].replace(r'^\s*$', 'unknown', regex=True)
        X = mvf.fill_missing_values(X, 'Profile Category', 'unknown', 'Num of Status Updates', 1)
        return self

    def transform( self, X, y = None ):
        X['Is Profile View Size Customized?'].fillna(False, inplace=True)
        X['Is Profile View Size Customized?'] = X['Is Profile View Size Customized?'].astype(bool)

        X['Profile Cover Image Status'].fillna('Not set', inplace=True)
        X['Profile Cover Image Status'] = X['Profile Cover Image Status'].apply(lambda strVal: (str(strVal).lower() == 'set') )

        X['Profile Verification Status'] = X['Profile Verification Status'].apply(lambda strVal: str(strVal).lower() )

        X['Profile Category'] = X['Profile Category'].replace(r'^\s*$', 'unknown', regex=True)

        return X.values

#Custom Transformer that modifies our numerical columns
class NumericalTransformer( BaseEstimator, TransformerMixin ):
    def __init__( self, feature_names ):
        self._feature_names = feature_names

    def handle_outliers(self, df, col_name, lower_limit, upper_limit, remove=False):
        # Remove
        if remove:
            outliers = df[((df[col_name] > upper_limit) | (df[col_name] < lower_limit))]
            print('Droped {:} outilers. Based on column {:}'.format(len(outliers), col_name))
            df = df.drop(outliers.index)
        # Clip ?
        else:
            df[col_name] = df[col_name].clip(lower_limit, upper_limit)

        return df

    def fit( self, X, y = None ):
        X = self.handle_outliers( X, 'Avg Daily Profile Clicks', avg_daily_profile_clicks_lower_limit, avg_daily_profile_clicks_upper_limit )
        X = self.handle_outliers( X, 'Num of Status Updates',    num_status_updates_lower_limit,       num_status_updates_upper_limit )
        X = self.handle_outliers( X, 'Num of Followers',         num_followers_lower_limit,            num_followers_upper_limit )
        X = self.handle_outliers( X, 'Num of People Following',  num_people_following_lower_limit,     num_people_following_upper_limit )
        X = self.handle_outliers( X, 'Num of Direct Messages',   num_direct_messages_lower_limit,      num_direct_messages_upper_limit )
        return self

    def transform( self, X, y = None ):
        # Don't need to do anything, since the SimpleImputer will handle imputation of missing values
        X['Avg Daily Profile Visit Duration in seconds'] = X['Avg Daily Profile Visit Duration in seconds']
        X['Avg Daily Profile Clicks'] = X['Avg Daily Profile Clicks']
        X['Num of Status Updates']    = X['Num of Status Updates']
        X['Num of Followers']         = X['Num of Followers']
        X['Num of People Following']  = X['Num of People Following']
        X['Num of Direct Messages']   = X['Num of Direct Messages']

        return X.values


#Color features
color_features = ['Profile Text Color', 'Profile Page Color', 'Profile Theme Color']

#Language features
language_features = ['Language']

#Location features
location_features = ['Location', 'UTC Offset', 'User Time Zone']

#Textual features to pass down the categorical pipeline
textual_features  = ['Location Public Visibility', 'Personal URL']

#Date features to pass down the categorical pipeline
datetime_features = ['Profile Creation Timestamp']

#Categrical features to pass down the categorical pipeline
categorical_features = ['Is Profile View Size Customized?', 'Profile Cover Image Status',
                        'Profile Verification Status', 'Profile Category']

#Numerical features to pass down the numerical pipeline
numerical_features   = ['Avg Daily Profile Visit Duration in seconds', 'Avg Daily Profile Clicks', 'Num of Status Updates',
                        'Num of Followers', 'Num of People Following', 'Num of Direct Messages']



#Defining the steps in the colors pipeline
colors_pipeline    = Pipeline( steps = [ ( 'colors_transformer',    ColorsTransformer(    color_features    ) ) ] )
language_pipeline  = Pipeline( steps = [ ( 'languages_transformer', LanguagesTransformer( language_features ) ) ] )
location_pipleline = Pipeline( steps = [ ( 'location_transformer',  LocationsTransformer( location_features ) ) ] )
textual_pipleline  = Pipeline( steps = [ ( 'text_transformer',      TextTransformer(      textual_features  ) ) ] )
datetime_pipleline = Pipeline( steps = [ ( 'datetime_transformer',  DateTimeTransformer(  datetime_features ) ) ] )

#Defining the steps in the categorical pipeline
categorical_pipeline = Pipeline( steps = [ ( 'cat_transformer', CategoricalTransformer( categorical_features ) ),
                                           ( 'one_hot_encoder', OneHotEncoder( sparse = False )                ) ] )

#Defining the steps in the numerical pipeline
numerical_pipeline   = Pipeline( steps = [ ( 'num_transformer', NumericalTransformer( numerical_features ) ),
                                           ( 'imputer',         SimpleImputer(strategy = 'median')         ),
                                           ( 'std_scaler',      StandardScaler( )                          ) ] )

#Combining numerical and categorical piepline into one full big pipeline horizontally
#using FeatureUnion
full_pipeline = FeatureUnion( transformer_list = [ ( 'colors_pipeline',      colors_pipeline      ),
                                                   ( 'language_pipeline',    language_pipeline    ),
                                                   ( 'location_pipleline',   location_pipleline   ),
                                                   ( 'textual_pipleline',    textual_pipleline    ),
                                                   ( 'datetime_pipleline',   datetime_pipleline   ),
                                                   ( 'categorical_pipeline', categorical_pipeline ),
                                                   ( 'numerical_pipeline',   numerical_pipeline   ) ] )
