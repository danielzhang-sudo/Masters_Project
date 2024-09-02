import torch # type: ignore
import torch.nn.functional as F # type: ignore
import matplotlib.pyplot as plt # type: ignore

from utils import *
from utils import EarlyStopper

from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, save_weights, load_lora
from loralib import layers as lora_layers

def evaluate_lora(args, clip_model, loader, dataset):
    clip_model.eval()
    with torch.no_grad():
        template = dataset.template[0] 
        texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
        text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.
    tot_samples = 0
    loss_epoch = 0.
    metrics_sum = {
        'accuracy': 0.0,
        'precision': torch.zeros(len(dataset.classnames)),
        'recall': torch.zeros(len(dataset.classnames)),
        'f1_score': torch.zeros(len(dataset.classnames)),
        'specificity': torch.zeros(len(dataset.classnames))
    }

    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            cosine_similarity = image_features @ text_features.t()
            loss = F.cross_entropy(cosine_similarity, target)

            acc += cls_acc(cosine_similarity, target) * target.shape[0]
            batch_metrics = cls_metrics(cosine_similarity, target, num_classes=len(dataset.classnames))
            
            for key in metrics_sum:
                if key == 'accuracy':
                    metrics_sum[key] += batch_metrics[key] * len(target)
                else:
                    metrics_sum[key] += batch_metrics[key] * len(target)

            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]
    
        acc /= tot_samples
        # Average the metrics
        for key in metrics_sum:
            if key == 'accuracy':
                metrics_sum[key] /= tot_samples
            else:
                metrics_sum[key] /= tot_samples
        
        loss_epoch /= tot_samples

    return acc, loss_epoch, metrics_sum


def run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader):
    
    VALIDATION = True
    
    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    textual_features = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = pre_load_features(clip_model, val_loader)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(clip_model, test_loader)
    
    test_features = test_features.cuda()
    test_labels = test_labels.cuda()
 
    # Zero-shot CLIP
    clip_logits = logit_scale * test_features @ textual_features
    zs_acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(zs_acc))

    # metrics = cls_metrics(clip_logits, test_labels)
    # print(f"\n**** Zero-shot CLIP's test metrics: \n{metrics}. ****\n")

    ext_metrics, confusion_matrix = extended_cls_metrics(clip_logits, test_labels)
    print(f"\n {args.dataset} \n**** Zero-shot CLIP's test extended metrics: \n{ext_metrics}. ****\n")

    with open('out.txt', 'a') as f:
        f.write(f"\n{args.dataset}\n**** Zero-shot CLIP's test extended metrics: \n{ext_metrics}. ****\n")

    if not args.finetune:
        print('no fineune')
        return
    
    test_features = test_features.cpu()
    test_labels = test_labels.cpu()
    
    
    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda() 
    
    if args.eval_only:
        load_lora(args, list_lora_layers)
        acc_test, loss_test, metrics_test = evaluate_lora(args, clip_model, test_loader, dataset)
        print("**** Test accuracy: {:.2f}, loss: {:.2f}. ****\n".format(acc_test, loss_test))
        return

    mark_only_lora_as_trainable(clip_model)
    total_iters = args.n_iters * args.shots
    
    optimizer = torch.optim.AdamW(get_lora_parameters(clip_model), weight_decay=1e-2, betas=(0.9, 0.999), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, eta_min=1e-6)
    early_stopper = EarlyStopper(patience=5, min_delta=0, list_lora_layers=list_lora_layers, args=args)
    
    best_acc_val, best_acc_test = 0., 0.
    best_epoch_val = 0
    
    # training LoRA
    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    finish = False

    train_acc_list = []
    val_acc_list = []
    loss_train_list = []
    loss_val_list = []

    while count_iters < total_iters:
        print('epoch:', count_iters)
        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.
        loss_epoch_val = 0.
        metrics_sum = {
            'accuracy': 0.0,
            'precision': torch.zeros(len(dataset.classnames)),
            'recall': torch.zeros(len(dataset.classnames)),
            'f1_score': torch.zeros(len(dataset.classnames)),
            'specificity': torch.zeros(len(dataset.classnames))
        }
        
        if args.encoder == 'vision': 
            text_features = textual_features.t().half()
        for i, (images, target) in enumerate(tqdm(train_loader)):
            
            template = dataset.template[0]
            texts = [template.format(classname.replace('_', ' ')) for classname in dataset.classnames]
            images, target = images.cuda(), target.cuda()
            if args.encoder == 'text' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    texts = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(texts)
                text_features = class_embeddings/class_embeddings.norm(dim=-1, keepdim=True)
                
            if args.encoder == 'vision' or args.encoder == 'both':
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model.encode_image(images)
            image_features = image_features/image_features.norm(dim=-1, keepdim=True)
            
            cosine_similarity = logit_scale * image_features @ text_features.t()
            loss = F.cross_entropy(cosine_similarity, target)

            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            batch_metrics = cls_metrics(cosine_similarity, target, num_classes=len(dataset.classnames))
            for key in metrics_sum:
                if key == 'accuracy':
                    metrics_sum[key] += batch_metrics[key] * target.shape[0]
                else:
                    metrics_sum[key] += batch_metrics[key] * target.shape[0]

            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()
            
            count_iters += 1
            
            if count_iters == total_iters:
                break
            
        if count_iters < total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            # Average the metrics
            for key in metrics_sum:
                if key == 'accuracy':
                    metrics_sum[key] /= tot_samples
                else:
                    metrics_sum[key] /= tot_samples
            
            # current_lr = scheduler.get_last_lr()[0]
            print('Acc: {:.4f}, Loss: {:.4f}'.format(acc_train, loss_epoch))
            print(metrics_sum)
            train_acc_list.append(acc_train)
            loss_train_list.append(loss_epoch)

        
        # Eval
        if VALIDATION:
            clip_model.eval()
            acc_val, loss_epoch_val, metrics_val = evaluate_lora(args, clip_model, val_loader, dataset)
            print("**** Val accuracy: {:.2f}, val loss: {:.6f}. ****\n".format(acc_val, loss_epoch_val))
            val_acc_list.append(acc_val)
            loss_val_list.append(loss_epoch_val)
        
        # Scheduler step
        # scheduler.step(loss_epoch_val)

        if early_stopper.early_stop(loss_epoch_val, count_iters-1):
            break
    
    acc_test, loss_test, metrics_test = evaluate_lora(args, clip_model, test_loader, dataset)
    print("**** Final test accuracy: {:.2f}, test loss: {:.2f}. ****\n".format(acc_test, loss_test))
    
    if args.save_path != None:
        _, _ = save_lora(args, list_lora_layers, True)
        # save_weights(data, path)

    # Create the accuracy plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_acc_list, 'b-', label='Training Accuracy')
    plt.plot(val_acc_list, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/accuracy_plot.png')
    plt.close()

    # Create the loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(loss_train_list, 'b-', label='Training Loss')
    plt.plot(loss_val_list, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/loss_plot.png')
    plt.close()
    return
            
    
            
