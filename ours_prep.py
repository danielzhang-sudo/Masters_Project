import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import BertModel, BertTokenizer
import torch
import time
import re
import numpy as np
import matplotlib.image as mim

def date_ordinal(sorted_dates):
    ordinal_encoding = {}

    current_ordinal = 0
    current_date = sorted_dates[0]

    for date in sorted_dates[:]:
        while current_date.strftime('%b-%Y') != date.strftime('%b-%Y'):
            current_ordinal += 1
            if current_date.month == 12:
                current_date = current_date.replace(year=(current_date.year + 1), 
                                                    month=(current_date.month % 12 + 1))
            else:
                current_date = current_date.replace(month=(current_date.month % 12) + 1)
        ordinal_encoding[date.strftime('%b-%Y')] = current_ordinal
        
    return ordinal_encoding

def encode_column_text(df_copy, column_name, model, tokenizer):
    # Encode the sentence
    sentences = list(df_copy[column_name])

    input_idss = []
    for sentence in sentences:
        input_idss.append(tokenizer.encode(sentence, 
                                           add_special_tokens=True, 
                                           return_tensors='pt'))

    embedded_sentences = []

    # Get the BERT embedding
    with torch.no_grad():
        for input_ids in input_idss:
            outputs = model(input_ids)
            embedded_sentences.append(outputs[0][:, 0, :].numpy())

    sentences_encoded = []

    for i in range(len(embedded_sentences)):
        sentences_encoded.append(embedded_sentences[i][0])

    df_copy[column_name+'_embedded'] = sentences_encoded

    return df_copy

def save_image(df_scaled, path):
    for i in list(df_scaled.index):
        row_atts = []
        row_pred = []
        for col in df_scaled.columns:
            val = df_scaled[col].loc[i]
            if col == 'loan_status':
                if val == 'Charged Off':
                        row_pred.append(0)
                else:
                    row_pred.append(1)
            else:
                if isinstance(val, list):
                    row_atts += val
                else:
                    row_atts.append(val)

        for u in range(len(row_atts)):
            row_atts[u] = np.float32(row_atts[u])
        
        # mim.imsave(f'images/{i}.png', np.array(row_atts).reshape(31,51))
        mim.imsave(f'{path}/{i}.png', np.array(row_atts).reshape(36,44))

def ours_prp():
    whole_df = pd.read_csv('lending_club_loan_two.csv')
    # drop address
    whole_df.drop('address', axis=1, inplace=True)
    # one hot encoding
    whole_df = pd.get_dummies(whole_df, columns=['home_ownership', 'verification_status', 
                                                'purpose', 'application_type'], dtype=int)
    # ordinal encoding
    whole_df['grade'] = pd.Categorical(list(whole_df['grade'])).codes
    whole_df['sub_grade'] = pd.Categorical(list(whole_df['sub_grade'])).codes
    # fill missing values
    whole_df['emp_length'] = whole_df['emp_length'].fillna(value='nan')
    whole_df['emp_title'] = whole_df['emp_title'].fillna(value='Not available')
    whole_df['title'] = whole_df['title'].fillna(value='Not available')
    whole_df['mort_acc'] = whole_df['mort_acc'].fillna(value=-1)
    whole_df['pub_rec_bankruptcies'] = whole_df['pub_rec_bankruptcies'].fillna(value=-1)
    # Define the mapping of emp_length values to their ordinal encoding
    mapping = {'nan': -1, '< 1 year': 11, '1 year': 10, 
            '2 years': 9, '3 years': 8, '4 years': 7, 
            '5 years': 6, '6 years': 5, '7 years': 4, 
            '8 years': 3, '9 years': 2, '10+ years': 1}
    # Apply the mapping to the 'emp_length' column
    whole_df['emp_length'] = whole_df['emp_length'].map(mapping)
    # drop nan rows
    whole_df = whole_df[whole_df['revol_util'].notna()]
    # create copy
    df_copy = whole_df.copy()
    # Parse the date string and create separate year and month columns
    df_copy['issue_d'] = pd.to_datetime(whole_df['issue_d'], format='%b-%Y')
    df_copy['issue_month'] = df_copy['issue_d'].dt.month - 1
    df_copy['issue_year'] = df_copy['issue_d'].dt.year - df_copy['issue_d'].dt.year.min()
    # Parse the date string and create separate year and month columns
    df_copy['earliest_cr_line'] = pd.to_datetime(whole_df['earliest_cr_line'], 
                                                format='%b-%Y')
    df_copy['cr_month'] = df_copy['earliest_cr_line'].dt.month - 1
    df_copy['cr_year'] = df_copy['earliest_cr_line'].dt.year - \
                                df_copy['earliest_cr_line'].dt.year.min()
    # drop old date formats
    df_copy.drop(['issue_d', 'earliest_cr_line'], axis=1, inplace=True)
    # Define the mapping of emp_length values to their ordinal encoding
    list_status_mapping = {'f': 0, 'w': 1}
    # Apply the mapping to the 'emp_length' column
    df_copy['initial_list_status'] = df_copy['initial_list_status'].map(list_status_mapping)
    term_mapping = {' 36 months':0, ' 60 months':1}
    df_copy['term'] = df_copy['term'].map(term_mapping)
    #add padding
    df_copy['padding'] = 0

    # Load pre-trained BERT model and tokenizer
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # print('text encoders loaded')

    df_copy = encode_column_text(df_copy, 'emp_title', model, tokenizer)
    df_copy.to_csv('emp_title_embedded.csv', index=False)
    # print('emp_title encoded')

    df_copy = encode_column_text(df_copy, 'title', model, tokenizer)
    # print('title encoded')
    df_copy.to_csv('encoded_dataset.csv', index=False)
    df_copy.drop(['emp_title','title'], axis=1, inplace=True)

    # df_z_scaled = df_copy.copy()

    df_copy = df_copy[['padding', 'loan_amnt', 'term', 'int_rate', 'installment', 
        'grade', 'sub_grade', 'emp_length', 'annual_inc', 'issue_month','issue_year', 
        'dti', 'cr_month', 'cr_year', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 
        'total_acc', 'initial_list_status', 'mort_acc', 'pub_rec_bankruptcies',
        'home_ownership_ANY', 'home_ownership_MORTGAGE', 'home_ownership_NONE',
        'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT',
        'verification_status_Not Verified','verification_status_Source Verified', 
        'verification_status_Verified', 'purpose_car', 'purpose_credit_card', 
        'purpose_debt_consolidation', 'purpose_educational', 'purpose_home_improvement', 
        'purpose_house', 'purpose_major_purchase', 'purpose_medical', 'purpose_moving',
        'purpose_other', 'purpose_renewable_energy', 'purpose_small_business',
        'purpose_vacation', 'purpose_wedding', 'application_type_DIRECT_PAY',
        'application_type_INDIVIDUAL', 'application_type_JOINT', 'emp_title_embedded',
        'title_embedded', 'loan_status']]

    subset_20 = df_copy.sample(frac=0.2)
    subset_500 = df_copy.sample(frac=500/len(subset_20))
    subset_100 = df_copy.sample(frac=100/len(subset_500))

    df_scaled = df_copy.copy()
    df_minmax = df_copy.copy()

    # z-score normalization
    cols = ['loan_amnt', 'int_rate', 'installment', 'grade', 'sub_grade', 'emp_length', 'annual_inc', 'issue_month', 'issue_year', 'dti', 'cr_month', 'cr_year', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'mort_acc', 'pub_rec_bankruptcies']
    for column in cols: 
        df_scaled[column] = (df_scaled[column] -
                            df_scaled[column].mean()) / df_scaled[column].std()

    #minmax normalization
    for column in cols:
        df_minmax[column] = (df_minmax[column] - \
                            df_minmax[column].min()) / (df_minmax[column].max() - \
                                                        df_minmax[column].min())
        
    subset_10_index = df_scaled['loan_status'].sample(frac=0.1).index

    save_image(df_scaled, 'CREDIT/credit/ours')
    save_image(subset_100, 'CREDIT/credit/ours_100')
    save_image(subset_500, 'CREDIT/credit/ours_500')
    save_image(subset_20, 'CREDIT/credit/ours_20')

    save_image(df_scaled.loc[subset_10_index], 'CREDIT/credit/zscore_norm_viridis')
    save_image(df_scaled.loc[subset_10_index], 'CREDIT/credit/zscore_norm', 'binary')
    save_image(df_minmax.loc[subset_10_index], 'CREDIT/credit/minmax_norm', 'viridis')
    save_image(df_minmax.loc[subset_10_index], 'CREDIT/credit/minmax_norm', 'binary')

    df_copy[['loan_status']].to_csv('CREDIT/credit/preprocessed_table.csv', index=True)
    subset_100.to_csv('CREDIT/credit/subset_100.csv', index=True)
    subset_500.to_csv('CREDIT/credit/subset_500.csv', index=True)
    subset_20.to_csv('CREDIT/credit/subset_20.csv', index=True)
    df_minmax.loc[subset_10_index].to_csv('ablation.csv', index=True)

if __name__=='__main__':
    ours_prp()