from pathlib import Path

import clip
import fire
import torch

from results_reproduce import zero_shot
from results_reproduce import train_adapter


def load_adapter_model(checkpoint_dir: str, device):
    checkpoint_dir = Path(checkpoint_dir)

    model_meta = torch.load(checkpoint_dir / 'model_meta.ckpt')
    clip_model, preprocess = clip.load(model_meta['visual_encoder_name'], device)

    model = train_adapter.ClipAdapter(clip_model, model_meta['dataset_name'], model_meta['visual_encoder_name'], model_meta['output_dim'])
    del model.clip_model
    model_checkpoint = torch.load(checkpoint_dir / 'model.ckpt')
    model = model.to(device)
    model.load_state_dict(model_checkpoint)
    model.clip_model = clip_model
    train_adapter.convert_models_to_fp32(model)
    
    return model, preprocess


def run(checkpoint_dir: str, dataset_name: str, batch_size: int = 32, num_workers: int = 2, device: str = 'cuda', random_state: int = 42):
    print(f'{dataset_name=}, {batch_size=}, {num_workers=}, {checkpoint_dir=}, {device=}')
    zero_shot.set_random_state(random_state)
    adapter_model, preprocess = load_adapter_model(checkpoint_dir, device)
    dataset = zero_shot.get_dataset(dataset_name, preprocess)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    classes, templates = zero_shot.load_promts(dataset_name)
    zeroshot_weights = zero_shot.zeroshot_classifier(adapter_model, classes, templates)

    top1, top5 = zero_shot.compute_accuracy(adapter_model, zeroshot_weights, loader)
    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")


if __name__ == '__main__':
    fire.Fire(run)
