import torch # type: ignore
import torchvision.transforms as transforms # type: ignore
import clip
from datasets import build_dataset
from datasets.utils import build_data_loader

from utils import *
from run_utils import *
from lora import run_lora


def main():

    # Load config file
    args = get_arguments()
    
    set_random_seed(args.seed)
    
    # CLIP
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()
    logit_scale = 100

    # Prepare dataset
    print("Preparing dataset.")
        
    dataset = build_dataset(args.dataset, args.root_path, args.shots, preprocess)
    
    val_loader = build_data_loader(data_source=dataset.val, batch_size=256, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=256, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8)
    
    train_loader = None
    if not args.eval_only:
        train_tranform = transforms.Compose([
            transforms.Resize(size=(224,224), interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        
        train_loader = build_data_loader(data_source=dataset.train_x, batch_size=args.batch_size, tfm=train_tranform, is_train=True, shuffle=True, num_workers=8)

    run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader)

if __name__ == '__main__':
    main()