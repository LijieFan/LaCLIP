import os.path

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as datasets

import json
import argparse

from tokenizer import SimpleTokenizer
from open_clip import create_model_and_transforms
from training.file_utils import pt_load
from eval_zeroshot_imagenet import validate_zeroshot

def main(args):
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        '',
        precision='amp',
        device='cuda',
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=None,
        force_image_size=224,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        aug_cfg={},
        output_dict=True,
    )
    checkpoint = pt_load(args.ckpt_path, map_location='cpu')
    sd = checkpoint["state_dict"]
    model.load_state_dict(sd)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    cudnn.benchmark = True

    with open('imagenet_labels.json') as f:
        labels = json.load(f)

    # Data loading code
    print("... creating dataset")
    tokenizer = SimpleTokenizer()


    val_dataset = datasets.ImageFolder(os.path.join(args.imagenet_root, 'val'), transform=preprocess_val)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    templates = json.load(open('imagenet_templates.json'))
    acc = validate_zeroshot(val_loader, templates, labels, model, tokenizer)
    print(f'ImageNet zero-shot accuracy: {acc}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ImageNet zero-shot evaluations', add_help=False)
    parser.add_argument('--imagenet-root', default='data/imagenet', type=str, help='path to imagenet dataset')
    parser.add_argument('--ckpt-path', default='checkpoints/cc12m_laclip.ckpt', type=str, help='model to test')
    parser.add_argument('--batch-size', default=256, type=int, help='batch_size')
    parser.add_argument('--model', default='ViT-B-32', type=str, help='model architecture')
    parser.add_argument('-j', '--workers', default=10, type=int)
    args = parser.parse_args()
    main(args)
