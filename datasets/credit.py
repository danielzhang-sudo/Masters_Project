import os
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader


# template = ['a photo representing information about a customer\'s financial credit risk indicators that {} the loan.']
template = ['a photo of a {} loan.']

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
    df = pd.read_csv(filepath, index_col=0)
    items = []
    
    for idx, row in df.iterrows():
        impath = f"{idx}.png"
        label = get_class_label(row['loan_status'])
        items.append((impath, label, row['loan_status']))

    temp_items, test_items = train_test_split(items, test_size=0.2, random_state=42)
    train_items, val_items = train_test_split(temp_items, test_size=0.3, random_state=42)
    
    train = _convert(train_items)
    val = _convert(val_items)
    test = _convert(test_items)

    return train, val, test

class Credit(DatasetBase):
    dataset_dir = 'CREDIT/credit'
    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'ours')
        self.split_path = os.path.join(self.dataset_dir, 'preprocessed_table.csv')
        
        self.template = template

        train, val, test = read_split(self.split_path, self.image_dir)
        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)


class CreditOld(DatasetBase):
    dataset_dir = 'CREDIT/credit'
    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        # self.image_dir = os.path.join(self.dataset_dir, 'ours_old')
        self.image_dir = '/scratch/uceexdz/CLIP-LoRA/CREDIT/credit/ours_old'
        self.split_path = os.path.join(self.dataset_dir, 'preprocessed_old.csv')
        
        self.template = template

        train, val, test = read_split(self.split_path, self.image_dir)
        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)


class Credit100(DatasetBase):
    dataset_dir = 'CREDIT/credit'
    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'ours_100/images')
        self.split_path = os.path.join(self.dataset_dir, 'subset_100.csv')
        # self.image_dir = '/scratch/uceexdz/CLIP-LoRA/CREDIT_OLD/images/images_100'
        # self.image_dir = os.path.join(self.dataset_dir, 'images_100')
        # self.split_path = '/scratch/uceexdz/CLIP-LoRA/CREDIT_OLD/subset_100.csv'
        
        self.template = template

        train, val, test = read_split(self.split_path, self.image_dir)
        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)


class Credit500(DatasetBase):
    dataset_dir = 'CREDIT/credit'
    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'ours_500/images')
        self.split_path = os.path.join(self.dataset_dir, 'subset_500.csv')
        # self.image_dir = '/scratch/uceexdz/CLIP-LoRA/CREDIT_OLD/images/images'
        # self.image_dir = os.path.join(self.dataset_dir, 'images')
        # self.split_path = '/scratch/uceexdz/CLIP-LoRA/CREDIT_OLD/subset_500.csv'
        
        self.template = template

        train, val, test = read_split(self.split_path, self.image_dir)
        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)


class Credit20(DatasetBase):
    dataset_dir = 'CREDIT/credit'
    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'ours_20/images')
        self.split_path = os.path.join(self.dataset_dir, 'subset_20.csv')
        
        self.template = template

        train, val, test = read_split(self.split_path, self.image_dir)
        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)


class CreditZS_Viridis(DatasetBase):
    dataset_dir = '/scratch/uceexdz/CLIP-LoRA/'
    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'zscore_norm_viridis/')
        self.split_path = os.path.join(self.dataset_dir, 'ablation.csv')
        
        self.template = template

        train, val, test = read_split(self.split_path, self.image_dir)
        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)

class CreditZS_Binary(DatasetBase):
    dataset_dir = '/scratch/uceexdz/CLIP-LoRA/'
    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'zscore_norm_binary/')
        self.split_path = os.path.join(self.dataset_dir, 'ablation.csv')
        
        self.template = template

        train, val, test = read_split(self.split_path, self.image_dir)
        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)

class CreditMM_Viridis(DatasetBase):
    dataset_dir = '/scratch/uceexdz/CLIP-LoRA/'
    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'minmax_norm_viridis/')
        self.split_path = os.path.join(self.dataset_dir, 'ablation.csv')
        
        self.template = template

        train, val, test = read_split(self.split_path, self.image_dir)
        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)

class CreditMM_Binary(DatasetBase):
    dataset_dir = '/scratch/uceexdz/CLIP-LoRA/'
    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'minmax_norm_binary/')
        self.split_path = os.path.join(self.dataset_dir, 'ablation.csv')
        
        self.template = template

        train, val, test = read_split(self.split_path, self.image_dir)
        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)