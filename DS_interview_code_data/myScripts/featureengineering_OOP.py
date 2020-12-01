import pandas as pd
import numpy as np
import datetime

from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.base import TransformerMixin, BaseEstimator, clone

#Class Objects:
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = preprocessing.LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = preprocessing.LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

class FeatureFunctionTransformer(BaseEstimator, TransformerMixin):
    """ A DataFrame transformer providing imputation or function application
    
    Parameters
    ----------
    impute : Boolean, default False
        
    func : function that acts on an array of the form [n_elements, 1]
        if impute is True, functions must return a float number, otherwise 
        an array of the form [n_elements, 1]
    
    """
    
    def __init__(self):
        self.series = pd.Series()
        
    def fit(self, X, y=None, **fit_params):
        """ Do nothing function

        Parameters
        ----------
        X : pandas DataFrame
        y : default None


        Returns
        ----------
        self  
        """
        return self

    def transform(self, X, **transformparams):
        """ Transforms a DataFrame
        
        Parameters
        ----------
        X : DataFrame
            
        Returns
        ----------
        trans : pandas DataFrame
            Transformation of X 
        """
        analysis_df = X.copy()
        
        all_col = list(analysis_df.columns)
        cat_col = ['country', 'sex', 'browser', 'source']
        datetime_col = ['signup_time', 'purchase_time']
        analysis_df = analysis_df.fillna('NA')
        
        def change_datetodate(dfcolumns):
            dfcolumns = pd.to_datetime(dfcolumns, format = '%m/%d/%Y %H:%M')
            return dfcolumns
        
        # Change To Date
        for item in datetime_col:
            analysis_df[item]= pd.to_datetime(analysis_df[item], format = '%m/%d/%Y %H:%M')
        
        
        def calculate_time_latency(df):
            '''Calculates the difference between sign up and purchase times'''
            df['time_latency'] = (df.purchase_time - df.signup_time).dt.total_seconds()/60/60
            return df
        
        # TimeLatancy
        latency_df = calculate_time_latency(analysis_df)
        
        def apply_label_encoding(df):
            return MultiColumnLabelEncoder(columns = df.columns).fit_transform(df)
        
        def calculate_ratio_fraud(analysis_df, sel_var):
            '''
            Args: 
                analysis_df: Dataframe with transaction level details
                sel_var: variable of interest for ratio calculation (country, device, )

            Output:
                Dataframe that merges the ratio of fraudulent transaction specific to selected variable to analysis_df
            '''
            tmp = analysis_df.groupby([sel_var, 'class']).user_id.nunique()\
            .unstack(level = 1)\
            .reset_index()\
            .rename(columns = {0:'Not Fraud', 1: 'Fraud'}).fillna(0.0)
            tmp['ratio_fraud_' + sel_var] = tmp['Fraud']/(tmp['Fraud'] + tmp['Not Fraud'])
            tmp['num_trans_' + sel_var] = tmp['Fraud'] + tmp['Not Fraud']
            return analysis_df[['user_id', sel_var]]\
                    .merge(tmp[[sel_var, 'ratio_fraud_' + sel_var, 'num_trans_' + sel_var]], on = sel_var)
        
        # Helper Variable
        c = ["device_id", "country", "age", "sex","source", "browser"]
        
        feature_df = calculate_time_latency(analysis_df)
        
        for item in c:
            fraud_by_X = calculate_ratio_fraud(analysis_df, item)
            feature_df = pd.merge(feature_df, fraud_by_X, how='outer', on='user_id', suffixes=[None, "_dup"])
            
        V = list(feature_df.columns)
        
        for item in V:
            if item.endswith("_dup"):
                V.remove(item)
        feature_df = feature_df[V]
        
        df_cat = apply_label_encoding(feature_df[['country', 'sex', 'browser', 'source']])
        df = pd.concat([feature_df.drop(['country', 'sex', 'browser', 'source'], axis = 1), df_cat], axis = 1).set_index(['user_id', 'device_id'])
        df = df[['ratio_fraud_device_id', 'num_trans_device_id', 'ratio_fraud_country',
       'num_trans_country', 'ratio_fraud_sex', 'num_trans_sex', 'age',
       'ratio_fraud_age', 'num_trans_age', 'ratio_fraud_browser',
       'num_trans_browser', 'ratio_fraud_source', 'num_trans_source',
       'purchase_value', 'class', 'time_latency', 'country', 'sex', 'browser',
       'source']]
        return df
 
