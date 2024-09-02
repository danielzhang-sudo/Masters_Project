from .credit import Credit, CreditOld, Credit100, Credit500, Credit20, CreditZS_Viridis, CreditZS_Binary, CreditMM_Viridis, CreditMM_Binary
from .deng import Deng100, Deng500, Deng20
from .aus import AusOurs, AusDeng
from .german import GermanOurs, GermanDeng

dataset_list = {
                "credit":Credit,
                "credit_old":CreditOld,
                "credit100":Credit100,
                "credit500":Credit500,
                "credit20":Credit20,
                "zsv":CreditZS_Viridis,
                "zsb":CreditZS_Binary,
                "mmv":CreditMM_Viridis,
                "mmb":CreditMM_Binary,
                "deng100":Deng100,
                "deng500":Deng500,
                "deng20":Deng20,
                "aus_ours":AusOurs,
                "aus_deng":AusDeng,
                "german_ours":GermanOurs,
                "german_deng":GermanDeng,
                }


def build_dataset(dataset, root_path, shots, preprocess):
    return dataset_list[dataset](root_path, shots)