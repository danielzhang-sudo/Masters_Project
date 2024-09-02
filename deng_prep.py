import pandas as pd
import matplotlib.pyplot as plt
import re


def clean_missing(df):
    # fill missing values
    df['emp_length'] = df['emp_length'].fillna(value='nan')
    df['emp_title'] = df['emp_title'].fillna(value='Not available')
    df['title'] = df['title'].fillna(value='Not available')
    df['mort_acc'] = df['mort_acc'].fillna(value=-1)
    df['pub_rec_bankruptcies'] = df['pub_rec_bankruptcies'].fillna(value=-1)
    print('missing values filled')

    # drop nan rows
    df = df[df['revol_util'].notna()]
    print('nan values')

    return df

def table2image_2(df, path):
    coll = list(df.columns)
    coll.remove('loan_status')

    def clean_text(text):
        if isinstance(text, str):
            return re.sub(r'([$%&])', r'\\\1', text)
        return text

    for i, (index, row) in enumerate(df.iloc[:, df.columns != 'loan_status'].iterrows()):
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(50, 10))  # Adjust size as needed
        ax.axis('tight')
        ax.axis('off')

        cleaned_values = [clean_text(val) for val in row.values]

        # Create a table from the current row                               
        table = ax.table(cellText=[cleaned_values],
                        colLabels=coll,
                        cellLoc='center', 
                        loc='center')
        
        plt.savefig(f'delete_test/{path}/{index}.png', bbox_inches='tight', dpi=200)
        plt.close(fig)  # Close the figure to avoid displaying it in interactive environments

def deng_method():
    whole_df = pd.read_csv('lending_club_loan_two.csv')
    subset_100_index = pd.read_csv('CREDIT/credit/subset_100.csv', index_col=0).index
    subset_500_index = pd.read_csv('CREDIT/credit/subset_500.csv', index_col=0).index
    subset_20_index = pd.read_csv('CREDIT/credit/subset_20.csv', index_col=0).index

    subset_100 = whole_df.loc[subset_100_index]
    subset_500 = whole_df.loc[subset_500_index]
    subset_20 = whole_df.loc[subset_20_index]

    subset_100 = clean_missing(subset_100)
    subset_500 = clean_missing(subset_500)
    subset_20 = clean_missing(subset_20)

    table2image_2(subset_100, 'CREDIT/credit/deng_100')
    table2image_2(subset_500, 'CREDIT/credit/deng_500')
    table2image_2(subset_20, 'CREDIT/credit/deng_20')

if __name__=='__main__':
    deng_method()