# [NeurIPS 2023] Improving CLIP Training with Language Rewrites
This repo contains text data, code and pre-trained models for paper [Improving CLIP Training with Language Rewrites](https://arxiv.org/abs/2305.20088). 
If you find the data, models or code useful, please consider citing our paper:
```bib
@inproceedings{fan2023improving,
  title={Improving CLIP Training with Language Rewrites},
  author={Fan, Lijie and Krishnan, Dilip and Isola, Phillip and Katabi, Dina and Tian, Yonglong},
  booktitle={NeurIPS},
  year={2023}
}
```

## Overview: 
We propose Language augmented CLIP (LaCLIP). LaCLIP enhances CLIP training by rewriting text descriptions with large language models. 
Key steps:

- **Meta-Input-Output Generation:**
we explored different strategies for generating meta-input-output pairs that can be used as examples in the prompt context for LLaMA in-context learning, namely *ChatGPT, Bard, MSCOCO* and *Human*.
Examples of generating such pairs with ChatGPT:
<p align="center"><img src="asset/chatgpt.png" alt="chatgpt" width="500"/></p>

- **In-Context Learning with LLaMA:**
Utilizing the constructed context input as a prompt, LLaMA exhibits its ability to perform text completion and generate rewritten versions of the corresponding text samples. This process is conducted for each text sample present in the pre-training image-text dataset.
Example of LLaMA rewriting a text sample:
<p align="center"><img src="asset/ICL.png" alt="ICL" width="500"/></p>

## Pre-trained Models
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">Dataset</th>
<th valign="center">Method</th>
<th valign="center">Zero-Shot</th>
<th valign="center">Checkpoint</th>

<!-- TABLE BODY -->
<tr>
<td align="center">CC3M</td>
<td align="center">CLIP</td>
<td align="center">15.8</td>
<td align="center"><a href="https://www.dropbox.com/s/5jsthdm85r2nfpz/cc3m_clip.pt?dl=0">ViT-B/16</a></td>
</tr>
<tr>
<td align="center">CC3M</td>
<td align="center">LaCLIP</td>
<td align="center">21.5</td>
<td align="center"><a href="https://www.dropbox.com/s/k2e1tgsfmo0afme/cc3m_laclip.pt?dl=0">ViT-B/16</a></td>
</tr>
<tr>
<td align="center">CC12M</td>
<td align="center">CLIP</td>
<td align="center">40.2</td>
<td align="center"><a href="https://www.dropbox.com/s/wwfq3txw4tk1yzj/cc12m_clip.pt?dl=0">ViT-B/16</a></td>
</tr>
<tr>
<td align="center">CC12M</td>
<td align="center">LaCLIP</td>
<td align="center">48.4</td>
<td align="center"><a href="https://www.dropbox.com/s/lle8x0tdxssfz11/cc12m_laclip.pt?dl=0">ViT-B/16</a></td>
</tr>
<tr>
<td align="center">RedCaps</td>
<td align="center">CLIP</td>
<td align="center">42.9</td>
<td align="center"><a href="https://www.dropbox.com/s/qvrvkwsy6j26suv/redcaps_clip.pt?dl=0">ViT-B/16</a></td>
</tr>
<tr>
<td align="center">RedCaps</td>
<td align="center">LaCLIP</td>
<td align="center">46.2</td>
<td align="center"><a href="https://www.dropbox.com/s/wpedkikz46gfzmg/redcaps_laclip.pt?dl=0">ViT-B/16</a></td>
</tr>
<tr>
<td align="center">LAION-400M</td>
<td align="center">CLIP</td>
<td align="center">62.0</td>
<td align="center"><a href="https://www.dropbox.com/s/zskxrso4pc4pe3j/laion400m_clip.pt?dl=0">ViT-B/32</a></td>
</tr>
<tr>
<td align="center">LAION-400M</td>
<td align="center">LaCLIP</td>
<td align="center">64.4</td>
<td align="center"><a href="https://www.dropbox.com/s/ahj8ys8uufndy9y/laion400m_laclip.pt?dl=0">ViT-B/32</a></td>
</tr>
<tr>
<td align="center">LAION-400M</td>
<td align="center">CLIP</td>
<td align="center">67.0</td>
<td align="center"><a href="https://www.dropbox.com/scl/fi/e235mjd5lkhu8h5nk4ovn/laion400m_clip_vitb16.pt?rlkey=s6guy23m8eyl7zzhfmyn1xt21&dl=0">ViT-B/16</a></td>
</tr>
<tr>
<td align="center">LAION-400M</td>
<td align="center">LaCLIP</td>
<td align="center">69.3</td>
<td align="center"><a href="https://www.dropbox.com/scl/fi/il3o958e2hvun2ei774ao/laion400m_laclip_vitb16.pt?rlkey=0domivxgaimqrfyuruak0h96b&dl=0">ViT-B/16</a></td>
</tr>
<tr>
<td align="center">LAION-400M</td>
<td align="center">CLIP</td>
<td align="center">71.8</td>
<td align="center"><a href="https://www.dropbox.com/scl/fi/o61athl9ijt2pufoplpsh/laion400m_clip_vitl14.pt?rlkey=ri0fzco6b5yfs7aryxvxc9rvw&dl=0">ViT-L/14</a></td>
</tr>
<tr>
<td align="center">LAION-400M</td>
<td align="center">LaCLIP</td>
<td align="center">74.5</td>
<td align="center"><a href="https://www.dropbox.com/scl/fi/v9y0vtgfjisctl0qyufdg/laion400m_laclip_vitl14.pt?rlkey=pdyzxpjeji1mw1xztn5gptqgo&dl=0">ViT-L/14</a></td>
</tr>
</tbody></table>


## Code Overview
- Code for generating rewrites of text samples
- 4 versions of augmented text on 3 datasets (CC3M, CC12M, RedCaps)
- Pre-trained models with LaCLIP and vanilla CLIP
- Zero-shot evaluation code on ImageNet
- Code for training LaCLIP
#### Dependencies
- PyTorch 1.11.0
- torchvision 0.12.0
- timm 0.5.4
- [open_clip](https://github.com/mlfoundations/open_clip/tree/main) (optional, for LAION-400M models)
- [LLaMA](https://github.com/facebookresearch/llama/tree/llama_v1) (for generating rewrites)

## Pre-computed Augmented Texts
- **Original** is the original caption associated with each image.
- **ChatGPT/Bard/MSCOCO/Human** is the text generated by LLaMA ICL with the ChatGPT/Bard/MSCOCO/Human Meta-Input-Output pairs as in-context learning examples.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">Dataset</th>
<th valign="center">Original</th>
<th valign="center">ChatGPT</th>
<th valign="center">Bard</th>
<th valign="center">MSCOCO</th>
<th valign="center">Human</th>

<!-- TABLE BODY -->
<tr>
<td align="center">CC3M</td>
<td align="center"><a href="https://www.dropbox.com/s/wajxrpfotcgkt7s/cc3m_original.csv?dl=0">Link</a></td>
<td align="center"><a href="https://www.dropbox.com/s/6x26v34g2iuoiss/cc3m_chatgpt.txt?dl=0">Link</a></td>
<td align="center"><a href="https://www.dropbox.com/s/u6o9mv7ukpo7epv/cc3m_bard.txt?dl=0">Link</a></td>
<td align="center"><a href="https://www.dropbox.com/s/upf6e3usak3ubnn/cc3m_mscoco.txt?dl=0">Link</a></td>
<td align="center"><a href="https://www.dropbox.com/s/6csrd83yb6tz2va/cc3m_human.txt?dl=0">Link</a></td>
</tr>
<tr>
<td align="center">CC12M</td>
<td align="center"><a href="https://www.dropbox.com/s/hndkbf5kxd2m0wi/cc12m_original.csv?dl=0">Link</a></td>
<td align="center"><a href="https://www.dropbox.com/s/znofj6j374mfvz9/cc12m_chatgpt.txt?dl=0">Link</a></td>
<td align="center"><a href="https://www.dropbox.com/s/kbv9z30s2glcaos/cc12m_bard.txt?dl=0">Link</a></td>
<td align="center"><a href="https://www.dropbox.com/s/ilfwnd46pt7doz8/cc12m_mscoco.txt?dl=0">Link</a></td>
<td align="center"><a href="https://www.dropbox.com/s/d4wue2loku20l3a/cc12m_human.txt?dl=0">Link</a></td>
</tr>
<tr>
<td align="center">RedCaps</td>
<td align="center"><a href="https://www.dropbox.com/s/viovukgvyc9uodv/redcaps_original.txt?dl=0">Link</a></td>
<td align="center"><a href="https://www.dropbox.com/s/x1by6mkffndg2ru/redcaps_chatgpt.txt?dl=0">Link</a></td>
<td align="center"><a href="https://www.dropbox.com/s/upaq5dw5xszl2fw/redcaps_bard.txt?dl=0">Link</a></td>
<td align="center"><a href="https://www.dropbox.com/s/xr9m8h8bwg9cn66/redcaps_mscoco.txt?dl=0">Link</a></td>
<td align="center"><a href="https://www.dropbox.com/s/94obx9y2d3w72u3/redcaps_human.txt?dl=0">Link</a></td>
</tr>
</tbody></table>


## Rewrite for other datasets
In order to rewrite for other datasets of your own interest, we put the code for rewriting in the `rewrite` folder.
Please refer to Meta [LLaMA page](https://github.com/facebookresearch/llama/tree/llama_v1) for detailed instructions for model access and environment setup. 
The rewritten text could be generated by running the following command:
```
export LLAMA_FOLDER=/PATH/TO/LLAMA/WEIGHTS
export PYTHONPATH=/PATH/TO/LLAMA/
export model='7b'
torchrun --nproc_per_node 1 --master_port 12388 \
    llama_rewrite.py --ckpt_dir ${LLAMA_FOLDER}/${model} --tokenizer_path ${LLAMA_FOLDER}/${model}/tokenizer.model \
    --max_batch_size 100 --max_seq_len 400 --prompt_filename text/source.txt --output_filename text/target.txt --sample_mode chatgpt --temperature 0.9
```
#### Main Arguments
- `--prompt_filename`: text file to be rewritten, each line is one sentence
- `--output_filename`: output path
- `--sample_mode`: sample mode for in-context learning (`chatgpt`, `bard`, `mscoco`, or `human`)
- `--temperature`: temperature for sampling, higher temperature leads to more diverse text

## Zero-shot Evaluation on ImageNet
To perform zero-shot evaluation on ImageNet, use the following command:

For CC3M, CC12M and RedCaps models:
```
python eval_zeroshot_imagenet.py --imagenet-root [PATH_TO_IMAGENET] --ckpt-path [PATH_TO_CHECKPOINT] --model CLIP_VITB16 --batch-size 128 --workers 8
```
For LAION-400M models:
```
python eval_zeroshot_imagenet_laion.py --imagenet-root [PATH_TO_IMAGENET] --ckpt-path [PATH_TO_CHECKPOINT] --model [ViT-B-32, ViT-B-16 or ViT-L-14] --batch-size 128 --workers 8
```
add `--quickgelu` for ViT-L-14 models.

## Training LaCLIP
To train LaCLIP, use the following command:
```
torchrun --nproc_per_node=GPU_PER_NODE --nnodes=NUM_NODE --node_rank=NODE_RANK \
  --master_addr=MASTER_NODE --master_port=PORT \
  train.py \
    --train-data PATH/TO/TRAINING/CSV \
    --root PATH/TO/TRAINING/IMAGE/ROOT \
    --imagenet-root PATH/TO/IMAGENET/ROOT \
    --aug-text --augmented_caption_filelist PATH/TO/AUGMENTED/CAPTION/FILES  \
    --output-dir PATH/TO/OUTPUT \
    --model CLIP_VITB16 \
    --lr 1e-3 --wd 0.5  --warmup-epochs 1 --batch-size 256 --epochs 35
```
#### Main Arguments
- `--train-data`: csv file for training data, each line is one image-text pair, with the relative image path and original caption separated by a comma
- `--root`: root dir for images
- `--imagenet-root`: root dir for ImageNet, used for zero-shot evaluation
- `--aug-text`: whether to use augmented text
- `--augmented_caption_filelist`: text files for augmented text, each line is one sentence, the order of the sentences should be the same as the order of the images in `--train-data`. Seperate the augmented text files with a space for multiple augmented text files.
- `--output-dir`: saving dir for logs and checkpoints
- `--model`: CLIP backbone architecture

#### Additional Notes
- Make sure the sample order in the `--augmented_caption_filelist` is the same as the order in `--train-data`.
- Please refer to Table A3 in the paper for the hyperparameters used for each dataset.
- To train vanilla CLIP, comment out the `--aug-text` and `--augmented_caption_filelist` arguments.
