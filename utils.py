from tqdm import tqdm # type: ignore
import torch # type: ignore
import clip
from loralib.utils import save_lora, save_weights
import matplotlib.pyplot as plt # type: ignore
# from sklearn.metrics import classification_report # type: ignore
from sklearn.metrics import cohen_kappa_score, roc_curve, auc, confusion_matrix, precision_recall_fscore_support # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report # type: ignore

class EarlyStopper():
    def __init__(self, patience=10, min_delta=0, list_lora_layers=None, args=None):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_val_loss = float('inf')
        self.args = args
        self.list_lora_layers = list_lora_layers
        # self.lora_weigths, self.save_path = save_lora(args, list_lora_layers, save_true=False)

    def early_stop(self, val_loss, epoch):
        if val_loss < self.min_val_loss:
            print('Val loss less')
            self.min_val_loss = val_loss
            self.counter = 0
        elif val_loss >= (self.min_val_loss + self.min_delta):
            self.counter += 1
            print(f'Val loss greater: {epoch}')
            if self.counter >= self.patience:
                # if self.args.save_path != None:
                #     _, _ = save_lora(self.args, self.list_lora_layers, save_true=True)
                return True
        return False

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    
    return acc

def cls_metrics(output, target, topk=1, num_classes=2):
    # Get predictions
    pred = output.topk(topk, 1, True, True)[1].t()

    # Calculate metrics for each class
    metrics = {
        'accuracy': 0.0,
        'precision': torch.zeros(num_classes),
        'recall': torch.zeros(num_classes),
        'f1_score': torch.zeros(num_classes),
        'specificity': torch.zeros(num_classes)
    }

    for i in range(num_classes):
        true_positives = ((pred == i) & (target == i)).sum().float()
        true_negatives = ((pred != i) & (target != i)).sum().float()
        false_positives = ((pred == i) & (target != i)).sum().float()
        false_negatives = ((pred != i) & (target == i)).sum().float()

        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        specificity = true_negatives / (true_negatives + false_positives) if true_negatives + false_positives > 0 else 0

        metrics['precision'][i] = precision
        metrics['recall'][i] = recall
        metrics['specificity'][i] = specificity
        metrics['f1_score'][i] = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    # Calculate overall accuracy
    metrics['accuracy'] = (pred == target).float().mean().item() * 100

    return metrics

def extended_cls_metrics(output, target, topk=1, num_classes=2):
    # Get predictions
    pred = output.topk(topk, 1, True, True)[1].t()
    
    # Convert to numpy arrays for sklearn functions
    y_true = target.cpu().numpy()
    y_pred = pred.cpu().numpy().ravel()
    y_score = output.cpu().numpy()[:, 1]  # Assuming binary classification, use scores for positive class

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate metrics for each class
    metrics = {
        'accuracy': 0.0,
        'precision': torch.zeros(num_classes),
        'recall': torch.zeros(num_classes),
        'f1_score': torch.zeros(num_classes),
        'specificity': torch.zeros(num_classes),
        'false_negative_rate': torch.zeros(num_classes),
        'true_negative_rate': torch.zeros(num_classes),
        'true_positive_rate': torch.zeros(num_classes),
        'positive_predictive_value': torch.zeros(num_classes)
    }

    for i in range(num_classes):
        true_positives = ((pred == i) & (target == i)).sum().float()
        true_negatives = ((pred != i) & (target != i)).sum().float()
        false_positives = ((pred == i) & (target != i)).sum().float()
        false_negatives = ((pred != i) & (target == i)).sum().float()

        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        specificity = true_negatives / (true_negatives + false_positives) if true_negatives + false_positives > 0 else 0

        metrics['precision'][i] = precision
        metrics['recall'][i] = recall
        metrics['specificity'][i] = specificity
        metrics['f1_score'][i] = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        
        # Additional metrics
        metrics['false_negative_rate'][i] = false_negatives / (false_negatives + true_positives) if false_negatives + true_positives > 0 else 0
        metrics['true_negative_rate'][i] = specificity  # True Negative Rate is the same as Specificity
        metrics['true_positive_rate'][i] = recall  # True Positive Rate is the same as Recall
        metrics['positive_predictive_value'][i] = precision  # Positive Predictive Value is the same as Precision

    # Calculate overall accuracy
    metrics['accuracy'] = (pred == target).float().mean().item() * 100

    # Calculate Cohen's Kappa
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

    # Calculate micro and macro average scores
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')

    metrics['precision_macro'] = precision_macro
    metrics['recall_macro'] = recall_macro
    metrics['f1_macro'] = f1_macro
    metrics['precision_micro'] = precision_micro
    metrics['recall_micro'] = recall_micro
    metrics['f1_micro'] = f1_micro

    # Calculate ROC curve and ROC AUC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    metrics['roc_auc'] = roc_auc

    # # Plot ROC curve
    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend(loc="lower right")
    # plt.show()

    return metrics, cm

def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
        clip_weights = torch.stack(clip_weights, dim=1).cuda()
        
    return clip_weights


def pre_load_features(clip_model, loader):
    features, labels = [], []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            images, target = images.cuda(), target.cuda()
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features.cpu())
            labels.append(target.cpu())
        features, labels = torch.cat(features), torch.cat(labels)
    
    return features, labels

