import random
import json
import pandas as pd

def get_class_label(clas):
    return 0 if str(clas) == 'Charged Off' else 1


if __name__=='__main__':
    all_entries = []

    df = pd.read_csv('captions_classes_dataset.csv')

    for i, row in df.iterrows():
        filename = f'{i}.png'
        label = get_class_label(row['loan_status'])
        classname = row['loan_status']
        all_entries.append([filename, label, classname])

    random.shuffle(all_entries)

    total = len(all_entries)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)

    train_data = all_entries[:train_size]
    val_data = all_entries[train_size:train_size + val_size]
    test_data = all_entries[train_size + val_size:]

    output_dict = {
        'train':train_data,
        'val':val_data,
        'test':test_data
    }

    with open('split_Credit.json', 'w') as f:
        json.dump(output_dict, f, indent=4)