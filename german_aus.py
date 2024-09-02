import pandas as pd
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import matplotlib.image as mim
import numpy as np


def replace_attribute_codes(df):
    # Dictionary to map attribute codes to their meanings
    attribute_meanings = {
        'A11': '< 0 DM',
        'A12': '0 <= ... < 200 DM',
        'A13': '>= 200 DM / salary assignments for at least 1 year',
        'A14': 'no checking account',
        'A30': 'no credits taken/ all credits paid back duly',
        'A31': 'all credits at this bank paid back duly',
        'A32': 'existing credits paid back duly till now',
        'A33': 'delay in paying off in the past',
        'A34': 'critical account/ other credits existing (not at this bank)',
        'A40': 'car (new)',
        'A41': 'car (used)',
        'A42': 'furniture/equipment',
        'A43': 'radio/television',
        'A44': 'domestic appliances',
        'A45': 'repairs',
        'A46': 'education',
        'A47': 'vacation',
        'A48': 'retraining',
        'A49': 'business',
        'A410': 'others',
        'A61': '< 100 DM',
        'A62': '100 <= ... < 500 DM',
        'A63': '500 <= ... < 1000 DM',
        'A64': '>= 1000 DM',
        'A65': 'unknown/ no savings account',
        'A71': 'unemployed',
        'A72': '< 1 year',
        'A73': '1 <= ... < 4 years',
        'A74': '4 <= ... < 7 years',
        'A75': '>= 7 years',
        'A91': 'male : divorced/separated',
        'A92': 'female : divorced/separated/married',
        'A93': 'male : single',
        'A94': 'male : married/widowed',
        'A95': 'female : single',
        'A101': 'none',
        'A102': 'co-applicant',
        'A103': 'guarantor',
        'A121': 'real estate',
        'A122': 'building society savings agreement/ life insurance',
        'A123': 'car or other, not in attribute 6',
        'A124': 'unknown / no property',
        'A141': 'bank',
        'A142': 'stores',
        'A143': 'none',
        'A151': 'rent',
        'A152': 'own',
        'A153': 'for free',
        'A171': 'unemployed/ unskilled - non-resident',
        'A172': 'unskilled - resident',
        'A173': 'skilled employee / official',
        'A174': 'management/ self-employed/ highly qualified employee/ officer',
        'A191': 'none',
        'A192': 'yes, registered under the customers name',
        'A201': 'yes',
        'A202': 'no'
    }
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_replaced = df.copy()
    
    # Iterate through columns and replace values
    for column in df_replaced.columns:
        if df_replaced[column].dtype == 'object':  # Only process string columns
            df_replaced[column] = df_replaced[column].map(
                attribute_meanings).fillna(df_replaced[column])
    
    return df_replaced


def table_to_image(df, path, x, y):
    for i in list(df.index):
        row_atts = []
        row_pred = []
        for col in df.columns:
            val = df[col].loc[i]
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
        
        mim.imsave(f'{path}/{i}.png', np.array(row_atts).reshape(y,x))

def table2image(df, path):
    coll = list(df.columns)

    for i, (index, row) in enumerate(df.iterrows()):
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(12, 0.01))  # Adjust size as needed
        ax.axis('tight')
        ax.axis('off')

        # Create a table from the current row
        table = ax.table(cellText=[row.values],
                        colLabels=coll,
                        cellLoc='center', 
                        loc='center')
        # plt.imshow()
        # break
        plt.savefig(f'{path}/{index}.png', bbox_inches='tight', dpi=100)
        plt.close(fig)  # Close the figure to avoid displaying it in interactive environments


statlog_german_credit_data = fetch_ucirepo(id=144)
X_german = statlog_german_credit_data.data.features
y_german = statlog_german_credit_data.data.targets

german = pd.DataFrame(X_german)
german_col = list(statlog_german_credit_data.variables.description)[:20]
german.columns = german_col

new_german = replace_attribute_codes(german)

new_german['Status of existing checking account'] = pd.Categorical(
    list(new_german['Status of existing checking account'])).codes
new_german['Credit history'] = pd.Categorical(
    list(new_german['Credit history'])).codes
new_german['Savings account/bonds'] = pd.Categorical(
    list(new_german['Savings account/bonds'])).codes
new_german['Present employment since'] = pd.Categorical(
    list(new_german['Present employment since'])).codes

colll = ['Purpose', 'Personal status and sex', 
         'Other debtors / guarantors', 'Property', 
         'Other installment plans', 'Housing', 
         'Job', 'Telephone', 'foreign worker']

cols = [col for col in new_german.columns if col not in colll]
for column in cols: 
    new_german[column] = (new_german[column] -
                        new_german[column].mean()) / new_german[column].std()

new_german = pd.get_dummies(new_german, columns=['Purpose', 
                    'Personal status and sex', 'Other debtors / guarantors', 
                    'Property', 'Other installment plans', 'Housing', 
                    'Job', 'Telephone', 'foreign worker'], dtype=int)

new_german['padding_1'] = 0
new_german['padding_2'] = 0

new_german = new_german[['padding_1', 'padding_2', 
       'Status of existing checking account', 'Duration', 'Credit history',
       'Credit amount', 'Savings account/bonds', 'Present employment since',
       'Installment rate in percentage of disposable income',
       'Present residence since', 'Age',
       'Number of existing credits at this bank',
       'Number of people being liable to provide maintenance for',
       'Purpose_business', 'Purpose_car (new)', 'Purpose_car (used)',
       'Purpose_domestic appliances', 'Purpose_education',
       'Purpose_furniture/equipment', 'Purpose_others',
       'Purpose_radio/television', 'Purpose_repairs', 'Purpose_retraining',
       'Personal status and sex_female : divorced/separated/married',
       'Personal status and sex_male : divorced/separated',
       'Personal status and sex_male : married/widowed',
       'Personal status and sex_male : single',
       'Other debtors / guarantors_co-applicant',
       'Other debtors / guarantors_guarantor',
       'Other debtors / guarantors_none',
       'Property_building society savings agreement/ life insurance',
       'Property_car or other, not in attribute 6', 'Property_real estate',
       'Property_unknown / no property', 'Other installment plans_bank',
       'Other installment plans_none', 'Other installment plans_stores',
       'Housing_for free', 'Housing_own', 'Housing_rent',
       'Job_management/ self-employed/ highly qualified employee/ officer',
       'Job_skilled employee / official',
       'Job_unemployed/ unskilled - non-resident', 'Job_unskilled - resident',
       'Telephone_none', 'Telephone_yes, registered under the customers name',
       'foreign worker_no', 'foreign worker_yes']]

table_to_image(new_german, 'CREDIT/german/german_ours', 8, 6)
table_german = replace_attribute_codes(german)
table2image(table_german)

# australian dataset
statlog_australian_credit_approval = fetch_ucirepo(id=143)
X_aus = statlog_australian_credit_approval.data.features 
y_aus = statlog_australian_credit_approval.data.targets

aus_2 = pd.DataFrame(X_aus)
df_aus = aus_2.copy()

aus_2 = pd.get_dummies(aus_2, columns=['A4','A5','A6','A12'], dtype=int)

norm_col = ['A2', 'A3', 'A7', 'A10', 'A13', 'A14']

for column in norm_col: 
    aus_2[column] = (aus_2[column] -
                        aus_2[column].mean()) / aus_2[column].std()
    
aus_2['padding_1'] = 0
aus_2['padding_2'] = 0

aus_2 = aus_2[['padding_1', 'padding_2', 'A1', 'A2', 'A3', 'A7', 
               'A8', 'A9', 'A10', 'A11', 'A13', 'A14', 'A4_1',
               'A4_2', 'A4_3', 'A5_1', 'A5_2', 'A5_3', 'A5_4', 
               'A5_5', 'A5_6', 'A5_7', 'A5_8', 'A5_9', 'A5_10', 
               'A5_11', 'A5_12', 'A5_13', 'A5_14', 'A6_1',
               'A6_2', 'A6_3', 'A6_4', 'A6_5', 'A6_7', 'A6_8', 
               'A6_9', 'A12_1', 'A12_2', 'A12_3']]

table_to_image(aus_2, 'CREDIT/aus/aus_ours', 8, 5)
table2image(df_aus, 'CREDIT/aus_aus_deng')

y_german.to_csv('CREDIT/german/german_classes.csv', index=True)
y_aus.to_csv('CREDIT/aus/aus_classes.csv', index=True)