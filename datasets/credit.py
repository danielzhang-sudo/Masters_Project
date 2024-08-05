import os
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader


template = ['a photo of a customer\'s financial credit risk indicator that {} the loan.']

def read_split(filepath, path_prefix):
    def _convert(items):
        out = []
        for impath, label, classname in items:
            impath = os.path.join(path_prefix, impath)
            item = Datum(
                impath=impath,
                label=int(label),
                classname=classname
            )
            out.append(item)
        return out

    def get_class_label(clas):
        return 0 if str(clas) == 'Charged Off' else 1
    
    print(f'Reading split from {filepath}')
    df = pd.read_csv(filepath)
    items = []
    
    for idx, row in df.iterrows():
        impath = f"{idx}.jpg"
        label = get_class_label(row['loan_status'])
        items.append((impath, label, row['loan_status']))

    train_items, temp_items = train_test_split(items, test_size=0.4, random_state=42)
    val_items, test_items = train_test_split(temp_items, test_size=0.5, random_state=42)
    
    train = _convert(train_items)
    val = _convert(val_items)
    test = _convert(test_items)

    return train, val, test

class Credit(DatasetBase):

    dataset_dir = 'CREDIT'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self.split_path = os.path.join(self.dataset_dir, 'split_Credit.json')
        
        self.template = template

        train, val, test = read_split(self.split_path, self.image_dir)
        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)