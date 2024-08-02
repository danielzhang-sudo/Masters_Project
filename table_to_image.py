import re
import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.image as mim # type: ignore

def string_to_list(string):
    """
    Converts a string containing a list of numbers separated by spaces into a list of numbers.
    
    Args:
        string (str): The input string containing the list of numbers.
        
    Returns:
        list: A list of numbers extracted from the input string.
    """
    # Remove square brackets and split the string by spaces
    numbers_str = re.sub(r'[\[\]]', '', string).split()
    
    # Convert each string to a float and append to a list
    numbers_list = [np.float32(num) for num in numbers_str]
    
    return numbers_list

if __name__=='__main__':

    pd.set_option('display.max_columns', None)
    df_copy = pd.read_csv('encoded_dataset.csv')

    df_copy['emp_title_embedded'] = df_copy['emp_title_embedded'].apply(string_to_list)
    df_copy['title_embedded'] = df_copy['title_embedded'].apply(string_to_list)
    df_copy.drop(columns=['emp_title', 'title'], inplace=True)

    df_copy = pd.get_dummies(df_copy, columns=['term'], dtype=int)
    # df_copy = df_copy[['loan_amnt', 'int_rate', 'installment', 'grade', 'sub_grade', 'emp_length', 'annual_inc', 'issue_d', 'dti', 'earliest_cr_line', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'mort_acc', 'pub_rec_bankruptcies', 'term_ 36 months', 'term_ 60 months', 'home_ownership_ANY', 'home_ownership_MORTGAGE', 'home_ownership_NONE', 'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT', 'verification_status_Not Verified', 'verification_status_Source Verified', 'verification_status_Verified', 'purpose_car', 'purpose_credit_card', 'purpose_debt_consolidation', 'purpose_educational', 'purpose_home_improvement', 'purpose_house', 'purpose_major_purchase', 'purpose_medical', 'purpose_moving', 'purpose_other', 'purpose_renewable_energy', 'purpose_small_business', 'purpose_vacation', 'purpose_wedding', 'initial_list_status_f', 'initial_list_status_w', 'application_type_DIRECT_PAY', 'application_type_INDIVIDUAL', 'application_type_JOINT', 'loan_status_Charged Off', 'loan_status_Fully Paid', 'emp_title_embedded', 'title_embedded']]

    df_z_scaled = df_copy

    df_z_scaled['loan_status'] = df_z_scaled[['loan_status_Charged Off', 'loan_status_Fully Paid']].idxmax(1)
    df_z_scaled = df_z_scaled.replace({'loan_status_Charged Off':0, 'loan_status_Fully Paid':1})
    # df_z_scaled = df_z_scaled.replace({'loan_status_Charged Off':'Charged Off', 'loan_status_Fully Paid':1})
    df_z_scaled = df_z_scaled.drop(columns=['loan_status_Charged Off', 'loan_status_Fully Paid'])

    df_z_scaled['term'] = df_z_scaled[['term_ 36 months', 'term_ 60 months']].idxmax(1)
    df_z_scaled = df_z_scaled.replace({'term_ 36 months':0, 'term_ 60 months':1})
    df_z_scaled = df_z_scaled.drop(columns=['term_ 36 months', 'term_ 60 months'])

    df_z_scaled['initial_list_status'] = df_z_scaled[['initial_list_status_f', 'initial_list_status_w']].idxmax(1)
    df_z_scaled = df_z_scaled.replace({'initial_list_status_f':0, 'initial_list_status_w':1})
    df_z_scaled = df_z_scaled.drop(columns=['initial_list_status_f', 'initial_list_status_w'])

    df_z_scaled = df_z_scaled[['loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade', 'emp_length', 'annual_inc', 'issue_d', 'dti', 'earliest_cr_line', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'initial_list_status', 'mort_acc', 'pub_rec_bankruptcies', 'home_ownership_ANY', 'home_ownership_MORTGAGE', 'home_ownership_NONE', 'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT', 'verification_status_Not Verified', 'verification_status_Source Verified', 'verification_status_Verified', 'purpose_car', 'purpose_credit_card', 'purpose_debt_consolidation', 'purpose_educational', 'purpose_home_improvement', 'purpose_house', 'purpose_major_purchase', 'purpose_medical', 'purpose_moving', 'purpose_other', 'purpose_renewable_energy', 'purpose_small_business', 'purpose_vacation', 'purpose_wedding', 'application_type_DIRECT_PAY', 'application_type_INDIVIDUAL', 'application_type_JOINT','emp_title_embedded', 'title_embedded',  'loan_status']]

    # apply normalization techniques 
    columns = ['loan_amnt', 'int_rate', 'installment', 'grade', 'sub_grade', 'emp_length', 'annual_inc', 'issue_d', 'dti', 'earliest_cr_line', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'mort_acc', 'pub_rec_bankruptcies']
    for column in columns: 
        df_z_scaled[column] = (df_z_scaled[column] -
                            df_z_scaled[column].mean()) / df_z_scaled[column].std()

    for i in range(len(df_z_scaled)):
        row_atts = []
        row_pred = []
        for col in df_z_scaled.columns:
            val = df_z_scaled[col].loc[i]
            if col == 'loan_status':
                row_pred.append(val)
            else:
                if isinstance(val, float):
                    row_atts.append(val)
                elif isinstance(val, np.int32) or isinstance(val, np.int64):
                    row_atts.append(val)
                elif isinstance(val, list):
                    row_atts += val

        for u in range(len(row_atts)):
            row_atts[u] = np.float32(row_atts[u])
        
        mim.imsave(f'images/{i}.png', np.array(row_atts).reshape(31,51))
        # mim.imsave(f'images/{counter}.png', np.array(row_atts).reshape(31,51), cmap='grey')