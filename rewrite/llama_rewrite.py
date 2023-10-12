from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import pickle
from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from tqdm import tqdm
import random
import math
from utils import chatgpt_source_caption_list, chatgpt_target_caption_list
from utils import bard_source_caption_list, bard_target_caption_list
from utils import human_source_caption_list, human_target_caption_list
from utils import coco_caption_list

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.9,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
    prompt_filename: str = 'text/source.txt',
    output_filename: str = 'text/target.txt',
    sample_mode: str = 'chatgpt',
):
    print('current sample mode is: ', sample_mode)
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    # read text from file, line by line to a list
    new_prompt_filename = output_filename
    # change parent directory to output
    with open(prompt_filename, 'r') as f:
        original_prompts = f.readlines()

    new_prompts = []
    num_batches = math.ceil(len(original_prompts) / max_batch_size)

    for batch_idx in tqdm(range(num_batches)):
        prompts = []
        current_batch = original_prompts[batch_idx * max_batch_size: (batch_idx + 1) * max_batch_size]

        for prompt_idx, original_prompt in enumerate(current_batch):
            chosen_source_caption_list = []
            chosen_target_caption_list = []
            if sample_mode == 'chatgpt':
                num_caps = len(chatgpt_source_caption_list)
                chosen_idx = random.sample(range(num_caps), 3)
                for idx in chosen_idx:
                    chosen_source_caption_list.append(chatgpt_source_caption_list[idx])
                    chosen_target_caption_list.append(chatgpt_target_caption_list[idx])
            elif sample_mode == 'bard':
                num_caps = len(bard_source_caption_list)
                chosen_idx = random.sample(range(num_caps), 3)
                for idx in chosen_idx:
                    chosen_source_caption_list.append(bard_source_caption_list[idx])
                    chosen_target_caption_list.append(bard_target_caption_list[idx])
            elif sample_mode == 'coco':
                num_caps = len(coco_caption_list)
                chosen_idx = random.sample(range(num_caps), 3)
                for idx in chosen_idx:
                    coco_chosen_idx = random.sample(range(len(coco_caption_list[idx])), 2)
                    chosen_source_caption_list.append(coco_caption_list[idx][coco_chosen_idx[0]])
                    chosen_target_caption_list.append(coco_caption_list[idx][coco_chosen_idx[1]])
            elif sample_mode == 'human':
                num_caps = len(human_source_caption_list)
                chosen_idx = random.sample(range(num_caps), 3)
                for idx in chosen_idx:
                    chosen_source_caption_list.append(human_source_caption_list[idx])
                    chosen_target_caption_list.append(human_target_caption_list[idx])
            else:
                raise ValueError('sample mode not supported')

            current_prompt = """write image captions differently,

            {} => {}

            {} => {}

            {} => {}

            {} =>""".format(
                chosen_source_caption_list[0], chosen_target_caption_list[0],
                chosen_source_caption_list[1], chosen_target_caption_list[1],
                chosen_source_caption_list[2], chosen_target_caption_list[2],
                original_prompt.replace('\n', ''))
            prompt_tokens = generator.tokenizer.encode(current_prompt, bos=True, eos=False)
            if len(prompt_tokens) <= max_seq_len-5:
                prompts.append(current_prompt)
            else:
                cut_len = max_seq_len - 10
                prompt_tokens = prompt_tokens[:cut_len]
                current_prompt = generator.tokenizer.decode(prompt_tokens) + ' =>'
                prompts.append(current_prompt)

        results = generator.generate(
            prompts, max_gen_len=77, temperature=temperature, top_p=top_p
        )

        for result in results:
            prompt_line = result.split('\n')[8].strip()
            new_prompt = prompt_line.split('=>')[1].strip()
            new_prompts.append(new_prompt)

        if local_rank == 0:
            with open(new_prompt_filename, 'w') as f:
                f.writelines([p.strip().replace('\n', ' ') + '\n' for p in new_prompts])


if __name__ == "__main__":
    fire.Fire(main)