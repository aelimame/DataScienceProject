import pandas as pd
import datetime as dt
import numpy as np

from time import strptime

from utils.missing_values_filler import MissingValuesFiller

# Hyper-parameters
NUM_LANGUAGES_TO_FEATUREIZE = 9
ACCOUNT_AGE_SINCE_DATE = dt.datetime(2021, 1, 1) # Jan 1st, 2021 is the date we measure account "age" from

COLUMNS_TO_DROP = ['User Name', 'Profile Image', 'User Time Zone']
COLOUR_COLUMNS = ['Profile Text Color', 'Profile Page Color', 'Profile Theme Color']
BOOLEAN_COLUMNS = ['Is Profile View Size Customized?', 'Location Public Visibility']

class HAL9001DataTransformer: # TODO Inherit from BaseEstimator and TransformerMixin?
    def __init__(self, verbose = False):
        self.verbose = verbose
        self.mvf = MissingValuesFiller()

    # TODO make it work like fit_transform and transform of sklearn to use pipelines...
    def fit_transform(self, input_df):
        # TODO for now, same as passing df in the constructor
        self.df = input_df.copy(deep=True)
        self.df = self.run_all()
        self.df.dropna(inplace=True)

        # TODO Hard coded for now. Remove outiliers using IQR or other techniques
        if 'Num of Profile Likes' in self.df:
            return self.df[self.df['Num of Profile Likes'] < 200000]

        return self.df

    # TODO make it work like fit_transform and transform of sklearn transformers...
    def fit(self, input_df):
        return NotImplementedError

    # TODO make it work like fit_transform and transform of sklearn transformers...
    def transform(self, input_df):
        return NotImplementedError


    def run_all(self):
        self.df = self.clean()
        self.df = self.engineer()
        return self.df

    def clean(self):
        self.remove_columns()
        # More cleaning?
        return self.df

    def split_colour_column(self, column_name):
        def parse_color(two_char_code):
            if not two_char_code or (two_char_code[0] == 'n'):
                return 0
            return int(two_char_code, 16) / 255

        colours_rgb = self.df[column_name]
        self.df[column_name+'_r'] = colours_rgb.apply(lambda colour: parse_color(str(colour)[0:2]) )
        self.df[column_name+'_g'] = colours_rgb.apply(lambda colour: parse_color(str(colour)[2:4]) )
        self.df[column_name+'_b'] = colours_rgb.apply(lambda colour: parse_color(str(colour)[4:6]) )
        return self.df

    def engineer_colour_columns(self):
        for column_name in COLOUR_COLUMNS:
            self.df = self.split_colour_column(column_name)
            self.df.drop(column_name, axis=1, inplace=True)
        return self.df

    def engineer_personal_url(self):
        self.df['Has Personal URL'] = self.df['Personal URL'].apply(lambda urlVal: 0 if str(urlVal).lower() == 'nan' else 1 )
        self.df.drop('Personal URL', axis=1, inplace=True)
        return self.df

    def engineer_utc_offset(self):
        # We are going to bin into one-hour timezones, and one-hot
        # Maybe try this to fill missing values?
        # self.df = self.mvf.fill_missing_values(self.df, 'UTC Offset', NaN, 'Num of Profile Likes', 5)

        # Using a random number between -8 UTC and + 10 UTC because that encompases most of the
        # land mass of the Earth.
        self.df['UTC Offset'] = self.df['UTC Offset'].fillna( np.random.randint( -8, 11 )*60*60 )
        # floor so we group 1/2 hour offsets
        self.df['UTC Offset'] = np.floor((self.df['UTC Offset']/60/60)).astype(int)

        one_hot_categories = pd.get_dummies(self.df['UTC Offset'], prefix='UTC Offset')
        self.df.drop('UTC Offset', axis=1, inplace=True)
        self.df = self.df.join(one_hot_categories)

		# Ensure all the important columns are there
        for i in np.arange(-12, 13):
        	if 'UTC Offset_'+str(i) not in self.df:
        		self.df['UTC Offset_'+str(i)] = 0

        return self.df

    def engineer_location(self):
        # For now, just set a flag if the location is empty
        self.df['Has Location'] = self.df['Location'].apply(lambda location: False if pd.isnull(location) else True )
        # TODO: Use geocoding results to add more info?
        self.df.drop('Location', axis=1, inplace=True)
        return self.df

    def engineer_profile_category(self):
        self.df['Profile Category'] = self.df['Profile Category'].replace(r'^\s*$', 'unknown', regex=True)
        # TODO: Remove debugging prints, when the issue is solved.
        # print("======================================")
        # print("Value counts input:\n{}\n".format(self.df['Profile Category'].value_counts()))
        # print("Row count input:\n{}\n".format(self.df.shape[0]))
        # print("Col count input:\n{}\n".format(self.df.shape[1]))
        # print("======================================")
        if 'Num of Status Updates' in self.df:
            self.df = self.mvf.fill_missing_values(self.df, 'Profile Category', 'unknown', 'Num of Status Updates', 1)
        # print("======================================")
        # print("Value counts output:\n{}\n".format(self.df['Profile Category'].value_counts()))
        # print("Row count output:\n{}\n".format(self.df.shape[0]))
        # print("Col count output:\n{}\n".format(self.df.shape[1]))
        one_hot_categories = pd.get_dummies(self.df['Profile Category'], prefix='Category')
        self.df.drop('Profile Category', axis=1, inplace=True)
        self.df = self.df.join(one_hot_categories)
        # print("Df columns:\n{}\n:".format(self.df.columns))
        # print("======================================")
        return self.df

    def engineer_profile_verification_status(self):
        ## TODO: Uncomment and test performance changes
        # self.df['Profile Verification Status'] = self.df['Profile Verification Status'].replace('Pending', 'Not verified', regex=True)
        # self.df['Profile Verification Status'] = self.df['Profile Verification Status'].apply(lambda val: True if val == 'Verified' else False)
        # Got: Validation rmsle: 1.7569663216091798 (was 1.7460608920116656 before)
        one_hot_categories = pd.get_dummies(self.df['Profile Verification Status'], prefix='Verification_Status')
        self.df.drop('Profile Verification Status', axis=1, inplace=True)
        self.df = self.df.join(one_hot_categories)
        return self.df

    def engineer_profile_cover_image_status(self):
        self.df['Profile Cover Image Status'].fillna('Not set', inplace=True)
        self.df['Profile Cover Image Status'] = self.df['Profile Cover Image Status'].apply(lambda strVal: 1 if (str(strVal).lower() == 'set') else 0 )
        return self.df

    def engineer_user_language(self, n):
        # Drop any country code (e.g. "en-gb" -> "en")
        self.df['User Language'] = self.df['User Language'].apply(lambda strVal: strVal[0:2].lower() )
        # Find the top n languages
        top_n_languages = self.df['User Language'].value_counts(normalize=True).keys().values[0:n]
        # If User Language is other than one of the top n languages, replace it with "other"
        self.df['User Language'] = self.df['User Language'].apply(lambda strVal: strVal[0:2] if ( strVal[0:2] in top_n_languages ) else "other" )
        # One-hot vector for each of top N languages
        one_hot_top_n_languages = pd.get_dummies(self.df['User Language'], prefix='Language')
        self.df.drop('User Language', axis=1, inplace=True)
        self.df = self.df.join(one_hot_top_n_languages)
        return self.df

    def engineer_avg_daily_visit_seconds(self):
        # Remove outliers?
        # TODO: What's better for fillNA here?
        col_name = 'Avg Daily Profile Visit Duration in seconds'
        self.df[col_name] = self.df[col_name].fillna(self.df[col_name].mean())
        return self.df

    def engineer_avg_daily_profile_clicks(self):
        # Remove outliers?
        # TODO: What's better for fillNA here?
        col_name = 'Avg Daily Profile Clicks'
        self.df[col_name] = self.df[col_name].fillna(self.df[col_name].mean())
        return self.df

    def engineer_profile_creation_timestamp(self):
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

        self.df['Account Age Days'] = self.df['Profile Creation Timestamp'].apply(lambda x: days_since_fixed_date(x) )
        self.df.drop('Profile Creation Timestamp', axis=1, inplace=True)
        return self.df

    def engineer(self):
        self.df = self.engineer_location()
        self.df = self.engineer_utc_offset()
        self.df = self.engineer_boolean_columns()
        self.df = self.engineer_colour_columns()
        self.df = self.engineer_personal_url()
        self.df = self.engineer_profile_creation_timestamp()
        self.df = self.engineer_profile_cover_image_status()
        self.df = self.engineer_profile_verification_status()
        self.df = self.engineer_profile_category()
        self.df = self.engineer_user_language(NUM_LANGUAGES_TO_FEATUREIZE)
        self.df = self.engineer_avg_daily_visit_seconds()
        self.df = self.engineer_avg_daily_profile_clicks()
        # More engineering ?
        return self.df


    def standardize_location_public_visibility(self):
        # This column uses "Enabled" and "Disabled" (and "??") rather than "True" and "False"
        self.df['Location Public Visibility'] = self.df['Location Public Visibility'].apply(lambda strVal: str(strVal).lower() == 'enabled' )
        return self.df

    def engineer_boolean_columns(self):
        self.df = self.standardize_location_public_visibility()
        for column_name in BOOLEAN_COLUMNS:
            self.df[column_name].fillna(False, inplace=True)
            self.df[column_name] = self.df[column_name].astype(bool).astype(int)
        return self.df

    def remove_columns(self):
        self.df.drop(COLUMNS_TO_DROP, axis=1, inplace=True)
        return self.df



