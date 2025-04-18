import hashlib
from pathlib import Path
from typing import List, Tuple
import os
import ftfy
import regex as re
import html
import torch
from accelerate.logging import get_logger
from safetensors.torch import save_file, load_file
from torch.utils.data import Dataset
from torchvision import transforms
import decord
import numpy as np
import json
decord.bridge.set_bridge("torch")
LOG_NAME = None
LOG_LEVEL = None

logger = get_logger(LOG_NAME, LOG_LEVEL)
from datasets import load_dataset



def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text

def load_prompts(prompt_path: Path) -> List[str]:
    with open(prompt_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines() if len(line.strip()) > 0]

def load_videos(video_path: Path) -> List[Path]:
    with open(video_path, "r", encoding="utf-8") as file:
        return [
            video_path.parent / line.strip() for line in file.readlines() if len(line.strip()) > 0
        ]

def preprocess_video_with_resize(
    video_path: Path | str,
    max_num_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Loads and resizes a single video.

    The function processes the video through these steps:
      1. If video frame count > max_num_frames, downsample frames evenly
      2. If video dimensions don't match (height, width), resize frames

    Args:
        video_path: Path to the video file.
        max_num_frames: Maximum number of frames to keep.
        height: Target height for resizing.
        width: Target width for resizing.

    Returns:
        A torch.Tensor with shape [F, C, H, W] where:
          F = number of frames
          C = number of channels (3 for RGB)
          H = height
          W = width
    """
    if isinstance(video_path, str):
        video_path = Path(video_path)
    video_reader = decord.VideoReader(uri=video_path.as_posix(), width=width, height=height)
    video_num_frames = len(video_reader)
    if video_num_frames < max_num_frames:
        # Get all frames first
        frames = video_reader.get_batch(list(range(video_num_frames)))
        # Repeat the last frame until we reach max_num_frames
        last_frame = frames[-1:]
        num_repeats = max_num_frames - video_num_frames
        repeated_frames = last_frame.repeat(num_repeats, 1, 1, 1)
        frames = torch.cat([frames, repeated_frames], dim=0)
        return frames.float().permute(0, 3, 1, 2).contiguous()
    else:
        indices = list(range(0, video_num_frames, video_num_frames // max_num_frames))
        frames = video_reader.get_batch(indices)
        frames = frames[:max_num_frames].float()
        frames = frames.permute(0, 3, 1, 2).contiguous()
        return frames

class T2VDataset(Dataset):

    def __init__(self,
                 root_dir: str,
                 max_num_frames: int,
                 height: int,
                 width: int,
                 scale: int,
                 use_keyframe: bool = False,
                 tokenizer=None,
                 ):
        super().__init__()

        self.data_root = root_dir

        self.dataset = load_dataset('json', data_files=self.data_root)["train"]

        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width

        self.scale = scale

        self.use_keyframe = use_keyframe
        self.tokenizer = tokenizer

        self.__frame_transform = transforms.Compose(
            [transforms.Lambda(lambda x : x / 255.0 * 2.0 -1.0)]
        )
        self.resize = transforms.Resize((int(self.height // self.scale), int(self.width // self.scale)), interpolation=transforms.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if isinstance(index, list):
            return  index
        input_video = self.dataset[index]["input_video"]
        prompt = self.dataset[index]["caption"]
        condition = self.dataset[index]["condition"]

        input_video = self.preprocess(input_video)
        input_video = self.frames_process(input_video)

        prompt, prompt_mask = self.tokenize_prompt_t5(prompt)

        condition = self.preprocess(condition)
        if self.use_keyframe:
            keyframe = condition[0].unsqueeze(0)
            keyframe = self.frames_process(keyframe)
        condition = self.resize(condition)
        condition = self.frames_process(condition)

        # exit()
        return {
            "prompt": prompt,
            "prompt_mask": prompt_mask,
            "input_video": input_video,
            "condition": condition,
            "keyframe": keyframe if self.use_keyframe else None,
        }

    def tokenize_prompt_t5(self, text):
        text = [text] if isinstance(text, str) else text
        text = [prompt_clean(u) for u in text]
    
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=226,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return text_inputs.input_ids, text_inputs.attention_mask


    def preprocess(self, video_path: Path) -> torch.Tensor:
        """
        这里暂时先考虑在空间维度上进行低分辨率操作，时间维度暂不考虑
        """
        high_video = preprocess_video_with_resize(video_path, self.max_num_frames, self.height, self.width)    # [81,3,480,832]
        
        return high_video

    def frames_process(self, frames: torch.Tensor):

        frames = self.video_transform(frames)
        frames = frames.unsqueeze(0)
        frames = frames.permute(0, 2, 1, 3, 4).contiguous()
        return frames

    def video_transform(self, frames: torch.Tensor):
        return torch.stack([self.__frame_transform(f) for f in frames], dim=0)





