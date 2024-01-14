import logging
import os
import random
from dataclasses import dataclass

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import csv
from tqdm import tqdm


class CsvDatasetAugCap(Dataset):
    def __init__(self, input_filename, transforms, tokenizer=None, root=None, augmented_caption_filelist=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        self.images = []
        self.captions = []
        self.root = root
        assert input_filename.endswith('.csv')
        assert augmented_caption_filelist is not None, 'augmented_caption_filelist is None, use csvdataset instead'

        num_augcap = len(augmented_caption_filelist)
        augmented_captions = []
        file_length = []
        for f in augmented_caption_filelist:
            with open(f, 'r') as file:
                cur_captions = file.readlines()
                file_length.append(len(cur_captions))
                augmented_captions.append(cur_captions)
        assert len(augmented_captions) == num_augcap, 'number of augmented captions is not equal to num_augcap'

        for i in range(num_augcap):
            assert file_length[i] == file_length[0], 'number of captions in each file is not the same'
        num_samples = file_length[0]

        with open(input_filename, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            row_index = 0
            for row in tqdm(csv_reader):
                image = row[0]
                prompt = row[1]
                if image.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(self.root, image)
                    self.images.append(image_path)

                    if row_index < num_samples:
                        self.captions.append([prompt])
                        for augcap_idx in range(num_augcap):
                            self.captions[row_index].append(augmented_captions[augcap_idx][row_index].replace('\n',''))
                        assert len(self.captions[row_index]) == num_augcap + 1, 'number of captions is not equal to num_augcap + 1'
                    row_index += 1
            assert row_index % num_samples == 0, 'number of samples in csv is not equal to num_samples in new caption'

        self.num_samples = num_samples
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        caption_list = self.captions[idx%self.num_samples]
        caption = random.choice(caption_list)
        if len(caption.split(' ')) < 2:
            caption = caption_list[0]
        texts = caption
        texts = self.tokenizer(str(texts))
        return images, texts


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, tokenizer=None, root=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        self.images = []
        self.captions = []
        self.root = root
        assert input_filename.endswith('.csv')
        with open(input_filename, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in tqdm(csv_reader):
                image = row[0]
                prompt = row[1]
                if image.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(self.root, image)
                    self.images.append(image_path)
                    self.captions.append(prompt)
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenizer(str(self.captions[idx]))
        return images, texts


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None

    def set_epoch(self, epoch):
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_csv_dataset(args, preprocess_fn, is_train, tokenizer=None, aug_text=False):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    if args.aug_text:
        augmented_caption_filelist = args.augmented_caption_filelist
        dataset = CsvDatasetAugCap(
            input_filename,
            preprocess_fn,
            root=args.root,
            tokenizer=tokenizer,
            augmented_caption_filelist=augmented_caption_filelist,
        )

    else:
        dataset = CsvDataset(
            input_filename,
            preprocess_fn,
            root=args.root,
            tokenizer=tokenizer
        )

    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_data(args, preprocess_fns, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {"train": get_csv_dataset(args, preprocess_train, is_train=True, tokenizer=tokenizer)}

    return data
