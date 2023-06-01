import os.path

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import json
import argparse
from collections import OrderedDict

import models
from tokenizer import SimpleTokenizer

def get_model(model):
    if isinstance(model, torch.nn.DataParallel) \
      or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model

def main(args):
    ckpt_path = args.ckpt_path
    ckpt = torch.load(ckpt_path, map_location='cpu')

    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    print("creating model: {}".format(args.model))
    print(f"loading checkpoint '{args.ckpt_path}")

    model = getattr(models, args.model)(rand_embed=False)

    model.cuda()
    model.load_state_dict(state_dict, strict=True)

    cudnn.benchmark = True

    with open('imagenet_labels.json') as f:
        labels = json.load(f)

    # Data loading code
    print("... creating dataset")
    tokenizer = SimpleTokenizer()
    val_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            lambda x: x.convert('RGB'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


    val_dataset = datasets.ImageFolder(os.path.join(args.imagenet_root, 'val'), transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    templates = [
        "itap of a {}.",
        "a bad photo of the {}.",
        "a origami {}.",
        "a photo of the large {}.",
        "a {} in a video game.",
        "art of the {}.",
        "a photo of the small {}."
    ]

    acc = validate_zeroshot(val_loader, templates, labels, model, tokenizer)
    print(f'ImageNet zero-shot accuracy: {acc}')


def validate_zeroshot(val_loader, templates, labels, model, tokenizer):
    # switch to evaluate mode
    model.eval()
    total_top1 = 0
    total_images = 0

    print('... getting classifier')
    with torch.no_grad():
        text_features = []
        for label in labels:
            if isinstance(label, list):
                texts = [t.format(l) for t in templates for l in label]
            else:
                texts = [t.format(label) for t in templates]
            texts = tokenizer(texts).cuda(non_blocking=True)
            texts = texts.view(-1, 77).contiguous()
            class_embeddings = get_model(model).encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=0)

        for images, target in val_loader:
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # encode images
            image_features = get_model(model).encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logits_per_image = image_features @ text_features.t()

            # measure accuracy and record loss
            pred = logits_per_image.argmax(dim=1)
            correct = pred.eq(target).sum()
            total_top1 += correct.item()
            total_images += images.size(0)
    return 100 * total_top1 / total_images



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ImageNet zero-shot evaluations', add_help=False)
    parser.add_argument('--imagenet-root', default='data/imagenet', type=str, help='path to imagenet dataset')
    parser.add_argument('--ckpt-path', default='checkpoints/cc12m_laclip.ckpt', type=str, help='model to test')
    parser.add_argument('--batch-size', default=256, type=int, help='batch_size')
    parser.add_argument('--model', default='CLIP_VITB16', type=str, help='model architecture')
    parser.add_argument('-j', '--workers', default=10, type=int)
    args = parser.parse_args()
    main(args)
