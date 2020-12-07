from time import strptime
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer, RobustScaler, Normalizer
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.exceptions import NotFittedError

# TODO: Maybe remove this and use SimpleImputer?
from utils.missing_values_filler import MissingValuesFiller
mvf = MissingValuesFiller()

ACCOUNT_AGE_SINCE_DATE = dt.datetime(2021, 1, 1) # Jan 1st, 2021 is the date we measure account "age" from
DEFAULT_NUM_LANGUAGES_TO_FEATUREIZE = 9
DEFAULT_NUM_USR_TZONES_TO_FEATUREIZE = 10
DEFAULT_NUM_UTC_TO_FEATUREIZE = 10
UTC_FOR_NA_VALUES = 10000000 # Define high value to give unique category to nan values

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



# Custom Transformer that modifies colour columns
class ColorsTransformer( BaseEstimator, TransformerMixin ):
    def __init__( self, feature_names ):
        self._in_feature_names = feature_names
        self._out_feature_names = []

    def split_colour_column(self, X, column_name):
        def parse_color(two_char_code):
            if not two_char_code or (two_char_code[0] == 'n'):
                return 0
            return int(two_char_code, 16) / 255

        colours_rgb = X[column_name]
        X[column_name+'_r'] = colours_rgb.apply(lambda colour: parse_color(str(colour)[0:2]) )
        X[column_name+'_g'] = colours_rgb.apply(lambda colour: parse_color(str(colour)[2:4]) )
        X[column_name+'_b'] = colours_rgb.apply(lambda colour: parse_color(str(colour)[4:6]) )
        # And features_names
        self._out_feature_names += [column_name+'_r']
        self._out_feature_names += [column_name+'_g']
        self._out_feature_names += [column_name+'_b']
        return X

    def fit( self, X, y = None ):
        return self

    def transform( self, X, y = None ):
        # Force copy so we don't change X inplace
        X = X.copy()

        # Reset _out_feature_names
        self._out_feature_names = []

        for column_name in self._in_feature_names:
            X = self.split_colour_column(X, column_name)
            #X.drop(column_name, axis=1, inplace=True)
        X = X[self._out_feature_names].values
        return X


# Languages Transformer that modifies colour columns
class LanguagesTransformer( BaseEstimator, TransformerMixin ):
    def __init__( self, feature_names, num_languages_to_featureize = DEFAULT_NUM_LANGUAGES_TO_FEATUREIZE ):
        self._in_feature_names = feature_names
        self._out_feature_names = []
        self._num_languages_to_featureize = DEFAULT_NUM_LANGUAGES_TO_FEATUREIZE

    def fit( self, X, y = None ):
        # Force copy so we don't change X inplace
        X = X.copy()

        # Drop any country code (e.g. "en-gb" -> "en")
        X['User Language'] = X['User Language'].apply(lambda strVal: strVal[0:2].lower() )
        # Find the top n languages
        self._top_n_languages = X['User Language'].value_counts(normalize=True).keys().values[0:self._num_languages_to_featureize]
        return self

    def transform( self, X, y = None ):
        # Force copy so we don't change X inplace
        X = X.copy()

        # Drop any country code (e.g. "en-gb" -> "en")
        X['User Language'] = X['User Language'].apply(lambda strVal: strVal[0:2].lower() )
         # If User Language is other than one of the top n languages, replace it with "other"
        X['User Language'] = X['User Language'].apply(lambda strVal: strVal[0:2] if ( strVal[0:2] in self._top_n_languages ) else "other" )
        
        # Fill _out_feature_names (same as _feature_names here?)
        self._out_feature_names = self._in_feature_names

        return X[self._out_feature_names].values


# Custom Transformer that creates advanced location featrues
class LocationAdvancedTransformer( BaseEstimator, TransformerMixin ):
    def __init__(self, feature_names,
                 num_tzones_to_featureize = DEFAULT_NUM_USR_TZONES_TO_FEATUREIZE,
                 num_utc_to_featureize = DEFAULT_NUM_UTC_TO_FEATUREIZE):
        self._in_feature_names = feature_names
        self._out_feature_names = []
        self._num_tzones_to_featureize = num_tzones_to_featureize
        self._num_utc_to_featureize = num_utc_to_featureize

    def fit( self, X, y = None ):
        # Force copy so we don't change X inplace
        X = X.copy()

        # User time zone
        # User time zone to lower case
        X['User Time Zone'] = X['User Time Zone'].apply(lambda str_val: str(str_val).lower())
        # Find the top time zones
        self._top_n_usr_tzones = X['User Time Zone'].value_counts(normalize=True).keys().values[0:self._num_tzones_to_featureize]

        # UTC
        # TODO nan need to be a category other than values present in the data!
        X['UTC Offset'] = X['UTC Offset'].fillna(UTC_FOR_NA_VALUES)
        # Floor UTC so we group 1/2 hour offsets
        X['UTC Offset'] = np.floor((X['UTC Offset']/60/60)).astype(int)
        X['UTC Offset'] = X['UTC Offset'].apply(lambda str_val: str(str_val).lower())
        # Find the top UTC zones
        self._top_n_utc_zones = X['UTC Offset'].value_counts(normalize=True).keys().values[0:self._num_utc_to_featureize]

        return self

    def transform( self, X, y = None ):
        # Force copy so we don't change X inplace
        X = X.copy()

        # User time zone
        # User time zone to lower case
        X['User Time Zone'] = X['User Time Zone'].apply(lambda str_val: str(str_val).lower())
        # If User time zone is other than one of the top n tzones, replace it with "other"
        X['User Time Zone'] = X['User Time Zone'].apply(lambda str_val: str_val if (str_val in self._top_n_usr_tzones) else "other")

        # UTC
        # TODO nan need to be a category other than values present in the data!
        X['UTC Offset'] = X['UTC Offset'].fillna(UTC_FOR_NA_VALUES)
        # Floor UTC so we group 1/2 hour offsets
        X['UTC Offset'] = np.floor((X['UTC Offset']/60/60)).astype(int)
        X['UTC Offset'] = X['UTC Offset'].apply(lambda str_val: str(str_val).lower())
        # If UTC Offset is other than one of the top n utc zones, replace it with "other"
        X['UTC Offset'] = X['UTC Offset'].apply(lambda str_val: str_val if (str_val in self._top_n_utc_zones) else "other")

        # Fill _out_feature_names (same as _feature_names here?)
        self._out_feature_names = self._in_feature_names

        return X[self._out_feature_names].values


# Custom Transformer that modifies textual columns
class TextTransformer( BaseEstimator, TransformerMixin ):
    def __init__( self, feature_names ):
        self._in_feature_names = feature_names
        self._out_feature_names = []

    def fit( self, X, y = None ):
        return self

    def transform( self, X, y = None ):
        # Force copy so we don't change X inplace
        X = X.copy()

        # Reset _out_feature_names
        self._out_feature_names = []

        for feature_name in self._in_feature_names:
            X['Has '+feature_name] = X[feature_name].apply(lambda urlVal: 0 if str(urlVal).lower() == 'nan' else 1)
            X['Has '+feature_name].fillna(0, inplace=True)
            #X.drop(feature_name, axis=1, inplace=True)
            # Fill out_features_names
            self._out_feature_names += ['Has '+feature_name]

        return X[self._out_feature_names].values


# Custom Transformer that modifies our date time columns
class DateTimeTransformer( BaseEstimator, TransformerMixin ):
    def __init__( self, feature_names ):
        self._in_feature_names = feature_names
        self._out_feature_names = []

    def fit( self, X, y = None ):
        return self

    def transform( self, X, y = None ):
        # Force copy so we don't change X inplace
        X = X.copy()

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
        #X.drop('Profile Creation Timestamp', axis=1, inplace=True)

        # Fill out_feature_names
        self._out_feature_names = ['Account Age Days']

        return X[self._out_feature_names].values


# Custom transformer for all binary features
class BinaryTransformer( BaseEstimator, TransformerMixin ):
    def __init__( self, feature_names ):
        self._in_feature_names = feature_names
        self._out_feature_names = []

    def fit( self, X, y = None ):
        return self

    def transform( self, X, y = None ):
        # Force copy so we don't change X inplace
        X = X.copy()

        X['Is Profile View Size Customized?'].fillna(False, inplace=True)
        X['Is Profile View Size Customized?'] = X['Is Profile View Size Customized?'].astype(bool)

        X['Profile Cover Image Status'].fillna('Not set', inplace=True)
        X['Profile Cover Image Status'] = X['Profile Cover Image Status'].apply(lambda strVal: (str(strVal).lower() == 'set') )

        # X['Has Location'] = self.df['Location'].apply(lambda location: False if pd.isnull(location) else True )
        # TODO: Use geocoding results to add more info?
        #X.drop('Location', axis=1, inplace=True)
        X['Has Location'] = X['Location'].apply(lambda location: False if pd.isnull(location) else True )

        # This column uses "Enabled" and "Disabled" (and "??") rather than "True" and "False"
        X['Location Public Visibility'] = X['Location Public Visibility'].apply(lambda strVal: str(strVal).lower() == 'enabled' ).astype(bool).astype(int)

        # Fill _out_feature_names
        self._out_feature_names = ['Is Profile View Size Customized?',
                                   'Profile Cover Image Status',
                                   'Location Public Visibility',
                                   'Has Location']

        return X[self._out_feature_names].values


# Custom Transformer that modifies our categorical columns
class CategoricalTransformer( BaseEstimator, TransformerMixin ):
    def __init__( self, feature_names ):
        self._in_feature_names = feature_names
        self._out_feature_names = []

    def fit( self, X, y = None ):
        # On Fit, we also want to try to impute values of Profile Category based on Num of Status Updates
        # TODO use fit/transform of mvf.fill_missing_values(...). Needs refactoring
        #X['Profile Category'] = X['Profile Category'].replace(r'^\s*$', 'unknown', regex=True)
        #X = mvf.fill_missing_values(X, 'Profile Category', 'unknown', 'Num of Status Updates', 1)
        return self

    def transform( self, X, y = None ):
        # Force copy so we don't change X inplace
        X = X.copy()
        
        X['Profile Verification Status'] = X['Profile Verification Status'].apply(lambda strVal: str(strVal).lower() )

        # TODO use fit/transform of mvf.fill_missing_values(...). Needs refactoring
        #X['Profile Category'] = X['Profile Category'].replace(r'^\s*$', 'unknown', regex=True)

        # Fill _out_feature_names (same as _feature_names here?)
        self._out_feature_names = self._in_feature_names

        return X[self._out_feature_names].values


# Custom Transformer that modifies our numerical columns
class NumericalTransformer( BaseEstimator, TransformerMixin ):
    def __init__( self, feature_names ):
        self._in_feature_names = feature_names
        self._out_feature_names = []

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
        # TODO if clipping only it is ok to do here, but dropping does not work
        # since X and y won't match anymore
        #X = self.handle_outliers( X, 'Avg Daily Profile Clicks', avg_daily_profile_clicks_lower_limit, avg_daily_profile_clicks_upper_limit )
        #X = self.handle_outliers( X, 'Num of Status Updates',    num_status_updates_lower_limit,       num_status_updates_upper_limit )
        #X = self.handle_outliers( X, 'Num of Followers',         num_followers_lower_limit,            num_followers_upper_limit )
        #X = self.handle_outliers( X, 'Num of People Following',  num_people_following_lower_limit,     num_people_following_upper_limit )
        #X = self.handle_outliers( X, 'Num of Direct Messages',   num_direct_messages_lower_limit,      num_direct_messages_upper_limit )
        return self

    def transform( self, X, y = None ):

        # Don't need to do anything, since the SimpleImputer will handle imputation of missing values
        X['Avg Daily Profile Visit Duration in seconds'] = X['Avg Daily Profile Visit Duration in seconds']
        X['Avg Daily Profile Clicks'] = X['Avg Daily Profile Clicks']
        X['Num of Status Updates']    = X['Num of Status Updates']
        X['Num of Followers']         = X['Num of Followers']
        X['Num of People Following']  = X['Num of People Following']
        X['Num of Direct Messages']   = X['Num of Direct Messages']

        # Fill _out_feature_names (same as _feature_names here?)
        self._out_feature_names = self._in_feature_names

        return X[self._out_feature_names].values


# HAL9001DataTransformer: Wrapper to call all the transfomers
class HAL9001DataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, copy=True):
        self.has_been_fit = False
        self.copy = copy

        # Init all Specific data tansfomers for each goupe of features and build the pipelines
        
        # Color features and pipeline
        color_features = ['Profile Text Color',
                          'Profile Page Color',
                          'Profile Theme Color']
        colors_pipeline = Pipeline(steps = [('colors_transformer', ColorsTransformer(color_features))])

        # Language features and pipeline
        language_features = ['User Language']
        language_pipeline = Pipeline(steps = [('languages_transformer', LanguagesTransformer(language_features)),
                                              ('lang_one_hot_encoder', OneHotEncoder(sparse = False))])

        # Location advanced features and pipeline
        location_adv_features = ['UTC Offset',
                                 'User Time Zone']
        location_adv_transformer = Pipeline(steps = [('location_advanced_transformer', LocationAdvancedTransformer(location_adv_features)),
                                                     ('loc_adv_one_hot_encoder', OneHotEncoder(sparse = False))
                                                    ])

        # Textual features and pipeline
        textual_features = ['Personal URL']
        textual_pipleline = Pipeline(steps = [('text_transformer', TextTransformer(textual_features))])

        # Date features and pipeline
        datetime_features = ['Profile Creation Timestamp']
        datetime_pipleline = Pipeline(steps = [('datetime_transformer', DateTimeTransformer(datetime_features))#,
                                              #('scaler', StandardScaler())
                                              ])

        # Binary features and pipeline
        binary_features = ['Profile Cover Image Status',
                           'Is Profile View Size Customized?',
                           'Location',
                           'Location Public Visibility']
        binary_pipeline = Pipeline(steps = [('binary_transformer', BinaryTransformer(binary_features))])

        # Categorical features and pipeline
        categorical_features = ['Profile Verification Status',
                                'Profile Category']
        categorical_pipeline = Pipeline(steps = [ ('cat_transformer', CategoricalTransformer(categorical_features)),
                                                 #('label_encoder', LabelEncoder())
                                                  ('cat_one_hot_encoder', OneHotEncoder(sparse = False))
                                                ])

        # Numerical features to pass down the numerical pipeline
        numerical_features = ['Num of Followers', 
                              'Num of People Following', 
                              'Num of Status Updates',
                              'Num of Direct Messages', 
                              'Avg Daily Profile Visit Duration in seconds', 
                              'Avg Daily Profile Clicks']
        numerical_pipeline = Pipeline(steps = [('num_transformer', NumericalTransformer(numerical_features)),
                                                # TODO see IterativeImputer()?
                                                #('imputer', IterativeImputer(initial_strategy = 'median'))#,
                                                ('imputer', SimpleImputer(strategy = 'median'))#,
                                                #('num_scaler', StandardScaler())
                                              ])

        # Combining numerical and categorical piepline into one full big pipeline horizontally
        # using FeatureUnion
        self.all_features_transformer = FeatureUnion(
                                    transformer_list = [
                                                        ('binary_pipeline', binary_pipeline),
                                                        ('numerical_pipeline', numerical_pipeline),
                                                        ('colors_pipeline', colors_pipeline),
                                                        ('textual_pipleline', textual_pipleline),
                                                        ('categorical_pipeline', categorical_pipeline),
                                                        ('language_pipeline', language_pipeline),
                                                        ('datetime_pipleline', datetime_pipleline),
                                                        #('location_adv_transformer', location_adv_transformer),
                                                       ])
        self.color_transformer = ColorsTransformer(color_features)


    def fit(self, X, y = None):
        # Force copy so we don't change X inplace
        if self.copy:
            X = X.copy()
        self.all_features_transformer.fit(X)
        self.has_been_fit = True
        return self

    def transform(self, X, y = None):
        if self.has_been_fit:
            # Force copy so we don't change X inplace
            if self.copy:
                X = X.copy()

            transf_X = self.all_features_transformer.transform(X)
            return transf_X
        else:
            msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
                   "appropriate arguments before using this estimator.")
            raise NotFittedError(msg % {'name': type(self).__name__})
        





