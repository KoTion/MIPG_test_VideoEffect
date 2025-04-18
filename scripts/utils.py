import torch
from torchvision import transforms

def encode_video(video, vae):

    latents_mean = (
        torch.tensor(vae.config.latents_mean)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(vae.device, vae.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        vae.device, vae.dtype
    )

    video = video.to(vae.device, dtype=vae.dtype)
    latent_dist = vae.encode(video).latent_dist
    latent = (latent_dist.sample() - latents_mean) * latents_std
    return latent


def frames_process(frames):
    frame_transform = transforms.Compose(
        [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)]
    )
    frames = torch.stack([frame_transform(f) for f in frames], dim=0)
    frames = frames.unsqueeze(0)
    frames = frames.permute(0, 2, 1, 3, 4).contiguous()
    return frames

def collate_fn(samples):
    ret = {"prompt_ids": [], "input_videos": [], "conditions": [], "keyframes": [], "prompt_mask": []}
    for sample in samples:
        input_video = sample["input_video"]
        condition = sample["condition"]
        keyframe = sample["keyframe"]
        prompt = sample["prompt"]
        prompt_mask = sample["prompt_mask"]

        ret["input_videos"].append(input_video)
        ret["conditions"].append(condition)
        ret["keyframes"].append(keyframe)
        ret["prompt_ids"].append(prompt)
        ret["prompt_mask"].append(prompt_mask)

    ret["input_videos"] = torch.stack(ret['input_videos']).squeeze(1)
    ret["conditions"] = torch.stack(ret["conditions"]).squeeze(1)
    ret["keyframes"] = torch.stack(ret["keyframes"]).squeeze(1)
    ret["prompt_ids"] = torch.stack(ret["prompt_ids"]).squeeze(1)
    ret["prompt_mask"] = torch.stack(ret["prompt_mask"]).squeeze(1)

    return ret
