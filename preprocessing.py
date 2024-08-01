import pandas as pd
from datetime import datetime
from transformers import BertModel, BertTokenizer
import torch

def create_caption(row):
    return f"""A person requesting a loan of {row['loan_amnt']}, with an interest rate of {row['int_rate']}, with monthly installments of {row['installment']}. The grade of the loan is {row['grade']}, the subgrade is {row['sub_grade']}. The employment is {row['emp_title']} with a length in years of {row['emp_length']} with an annual income of {row['annual_inc']}. The issue date of the loan is {row['issue_d']}. The debt-to-income ratio of the customer is {row['dti']}, with the earliest credit line opened in {row['earliest_cr_line']}. The customer has {row['open_acc']} credit lines opened, with {row['pub_rec']} derogatory public records. The revolving balance is {row['revol_bal']}, with a revolving utilization rate for this line of {row['revol_util']}. The customer has currently opened {row['total_acc']} credit lines. The customer has {row['mort_acc']} mortgage accounts and {row['pub_rec_bankruptcies']}. The term of the loan is {row['term']}, and his home ownership status is {row['home_ownership']}. The verification status of the income is {row['verification_status']}. The purpose of the loan is {row['purpose']}, its initial listing status is {row['initial_list_status']} and the application type is {row['application_type']}. The customer comments regarding the loan application are the following: {row['title']}. The loan status is {row['loan_status']}"""

def date_ordinal(sorted_dates):
    ordinal_encoding = {}

    current_ordinal = 0
    current_date = sorted_dates[0]

    for date in sorted_dates[:]:
        while current_date.strftime('%b-%Y') != date.strftime('%b-%Y'):
            current_ordinal += 1
            if current_date.month == 12:
                current_date = current_date.replace(year=(current_date.year + 1), month=(current_date.month % 12 + 1))
            else:
                current_date = current_date.replace(month=(current_date.month % 12) + 1)
        ordinal_encoding[date.strftime('%b-%Y')] = current_ordinal
        
    return ordinal_encoding

def encode_column_text(df_copy, column_name):
    # Encode the sentence
    sentences = list(df_copy[column_name])

    input_idss = []
    for sentence in sentences:
        input_idss.append(tokenizer.encode(sentence, add_special_tokens=True, return_tensors='pt'))

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

if __name__=='__main__':
    print('start preprocessing')

    df_copy = pd.read_csv('lending_club_loan_two.csv')
    pd.set_option('display.max_columns', None)
    print('data loaded')

    # df_copy = df.copy()
    df_copy.drop('address', axis=1, inplace=True)

    # get image caption
    df_copy['caption'] = df_copy.apply(create_caption, axis=1)

    # one hot encoding
    df_copy = pd.get_dummies(df_copy, columns=['home_ownership', 'verification_status', 'purpose', 'initial_list_status', 'application_type'], dtype=int)
    print('one-hot encoded')

    # ordinal encoding
    df_copy['grade'] = pd.Categorical(list(df_copy['grade'])).codes
    df_copy['sub_grade'] = pd.Categorical(list(df_copy['sub_grade'])).codes
    print('ordinal encoded')

    # fill missing values
    df_copy['emp_length'] = df_copy['emp_length'].fillna(value='nan')
    df_copy['emp_title'] = df_copy['emp_title'].fillna(value='Not available')
    df_copy['title'] = df_copy['title'].fillna(value='Not available')
    print('missing values filled')

    # Define the mapping of emp_length values to their ordinal encoding
    mapping = {'nan': -1, '< 1 year': 11, '1 year': 10, '2 years': 9, '3 years': 8, '4 years': 7, '5 years': 6, '6 years': 5, '7 years': 4, '8 years': 3, '9 years': 2, '10+ years': 1}

    # Apply the mapping to the 'emp_length' column
    df_copy['emp_length'] = df_copy['emp_length'].map(mapping)

    iss_d = date_ordinal(sorted([datetime.strptime(date, '%b-%Y') for date in list(df_copy['issue_d'].unique())]))
    ear_cr_line = date_ordinal(sorted([datetime.strptime(date, '%b-%Y') for date in list(df_copy['earliest_cr_line'].unique())]))
    df_copy = df_copy.replace({'issue_d':iss_d, 'earliest_cr_line':ear_cr_line})
    print('date encoded')

    # fill nan values
    df_copy['mort_acc'] = df_copy['mort_acc'].fillna(value=-1)
    df_copy['pub_rec_bankruptcies'] = df_copy['pub_rec_bankruptcies'].fillna(value=-1)

    # drop nan rows
    df_copy = df_copy[df_copy['revol_util'].notna()]
    print('nan values')
    """
    # Load pre-trained BERT model and tokenizer
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print('text encoders loaded')

    df_copy = encode_column_text(df_copy, 'emp_title')
    df_copy.to_csv('emp_title_embedded.csv', index=False)
    print('emp_title encoded')

    df_copy = encode_column_text(df_copy, 'title')
    print('title encoded')
    """

    # df_copy.to_csv('encoded_dataset.csv', index=False)
    df_copy[['caption', 'loan_status']].to_csv('captions_classes_dataset.csv', index=False)
    
    print('data saved')