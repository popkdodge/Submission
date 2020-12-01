import pandas as pd
import numpy as np
## Merge two dataframes in range
def merge_on_range(A, B):
    '''
    Args: 
        Dataframe A consists of the ip_addresses associated with each transaction
        Dataframe B consists of lower and upper bounds of ip_addresses corresponding to each country
        
    Output:
        Dataframe that uniquely maps each ip_address to a 
        specific country and returns a null if a match is not found
    '''
    a = A.ip_address.values
    bh = B.upper_bound_ip_address.astype(float).values
    bl = B.lower_bound_ip_address.astype(float).values

    i, j = np.where((a[:, None] >= bl) & (a[:, None] <= bh))

    return pd.DataFrame(
        np.column_stack([A.values[i], B.values[j]]),
        columns=A.columns.append(B.columns)
    ).drop(['lower_bound_ip_address', 'upper_bound_ip_address'], axis = 1)

def join_datasets(fraud_datapath, IPtoCountry_datapath, output_path):
    '''
    Args: 
        fraud_datapath: path to the csv file containing dataset with all transactions
        IPtoCountry_datapath: path to the xlsx file with lower and upper bounds of ip_addresses for each country
        output_path: path for the output of mapping function
    Output:
        Dataframe that uniquely maps each ip_address to a specific country
        Dataframe is also saved as a csv file in the output_path
    '''
    fraud_df = pd.read_csv(fraud_datapath).drop('Unnamed: 0', axis = 1)
    ip_mapping_df = pd.read_excel(IPtoCountry_datapath)
    df_joined = merge_on_range(fraud_df, ip_mapping_df)

    ## Some IPs do not fall within any range
    country_na = fraud_df[~fraud_df.user_id.isin(df_joined.user_id)]

    ## Bring them in to the analysis dataset with missing country information
    analysis_df = pd.concat([df_joined, country_na], sort = False).fillna('NA')
    analysis_df.to_csv(output_path)
    
    return analysis_df