import itertools
from pathlib import Path
from copy import copy

import torch
import clip
import fire
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from results_reproduce import zero_shot


class ClipAdapter(nn.Module):
    def __init__(self, clip_model, dataset_name, visual_encoder_name, output_dim=None):
        super().__init__()

        self.clip_model = clip_model
        self.dataset_name = dataset_name
        self.visual_encoder_name = visual_encoder_name
        self.classnames, self.templates = zero_shot.load_promts(dataset_name)
        
        zeroshot_weights = zero_shot.zeroshot_classifier(clip_model, self.classnames, self.templates)
        self.zeroshot_weights = nn.parameter.Parameter(zeroshot_weights, requires_grad=False) 

        embed_dim = clip_model.text_projection.size(dim=1)
        self.output_dim = output_dim or embed_dim
        self.vision_adapter = nn.Linear(embed_dim, self.output_dim)
        self.text_adapter = nn.Linear(embed_dim, self.output_dim)

    def encode_image(self, image):
        image = self.clip_model.encode_image(image)
        image = self.vision_adapter(image)
        return image

    def encode_text(self, text):
        text = self.clip_model.encode_text(text)
        text = self.text_adapter(text)
        return text

    def forward(self, image, label):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            text_features = self.zeroshot_weights[:, label].t()

        image_features = self.vision_adapter(image_features)
        text_features = self.text_adapter(text_features)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text


# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for param in model.parameters():
        param.data = param.data.float()
        if param.requires_grad and param.grad is not None:
            param.grad.data = param.grad.data.float() 


def train_epoch(loader, model, loss, optimizer, device, summary_writer, tb_loss_step):
    model.train()
    model = model.to(device)
    convert_models_to_fp32(model)
    epoch_loss = 0.

    for images, labels in tqdm(loader):
        model.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        
        logits_per_image, logits_per_text = model(images, labels)
        dummy_labels = torch.arange(len(labels), device=device)
        image_loss = loss(logits_per_image, dummy_labels)
        text_loss = loss(logits_per_text, dummy_labels)
        agg_loss = (image_loss + text_loss) / 2

        summary_writer.add_scalar('loss-train-image', image_loss.item(), tb_loss_step)
        summary_writer.add_scalar('loss-train-text', text_loss.item(), tb_loss_step)
        summary_writer.add_scalar('loss-train-agg', agg_loss.item(), tb_loss_step)
        tb_loss_step += 1
        epoch_loss += agg_loss.item()

        agg_loss.backward()
        # convert_models_to_fp32(model)
        optimizer.step()
        # clip.model.convert_weights(model)

    return model, epoch_loss, optimizer, tb_loss_step


def eval_model(loader, model: ClipAdapter):
    zeroshot_weights = zero_shot.zeroshot_classifier(model, model.classnames, model.templates)
    return zero_shot.compute_accuracy(model, zeroshot_weights, loader)


def save_epoch_model(model, optimizer, tb_loss_step, epoch_num, checkpoints_dir: Path):
    epoch_dir = checkpoints_dir / f'epoch_{epoch_num}'
    epoch_dir.mkdir(parents=True, exist_ok=True)

    def save_data(data, data_name):
        with open(epoch_dir / f'{data_name}.ckpt', 'wb') as f:
            torch.save(data, f)

    model_state_dict = copy(model.state_dict())
    clip_model_keys = [param_name for param_name in model_state_dict.keys() if param_name.startswith('clip_model')]
    for clip_model_key in clip_model_keys:
        model_state_dict.pop(clip_model_key)

    model_meta = {'visual_encoder_name': model.visual_encoder_name, 'dataset_name': model.dataset_name, 'output_dim': model.output_dim}
    save_data(model_state_dict, 'model')
    save_data(model_meta, 'model_meta')
    save_data(optimizer.state_dict(), 'optimizer')
    save_data(tb_loss_step, 'tb_loss_step')


def train_model(train_loader, val_loader, model, loss, optimizer, device, epochs_num, checkpoints_dir: Path):
    log_dir = checkpoints_dir / 'tb_runs'
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_writer = SummaryWriter(log_dir)
    tb_loss_step = 0

    for epoch_num in range(1, epochs_num + 1):
        print(f'Running epoch {epoch_num}...')
        model, epoch_loss, optimizer, tb_loss_step = train_epoch(train_loader, model, loss, optimizer, device, summary_writer, tb_loss_step)
        print(f'Evaluating model on train...')
        eval_top1, eval_top5 = eval_model(train_loader, model)
        summary_writer.add_scalar('train-epoch-sum-loss', epoch_loss, epoch_num)
        summary_writer.add_scalar('train-epoch-acc@1', eval_top1, epoch_num)
        summary_writer.add_scalar('train-epoch-acc@5', eval_top5, epoch_num)
        print('train-acc@1:', eval_top1)
        print('train-acc@5:', eval_top5)
        print(f'Evaluating model on validation...')
        eval_top1, eval_top5 = eval_model(val_loader, model)
        summary_writer.add_scalar('val-epoch-sum-loss', epoch_loss, epoch_num)
        summary_writer.add_scalar('val-epoch-acc@1', eval_top1, epoch_num)
        summary_writer.add_scalar('val-epoch-acc@5', eval_top5, epoch_num)
        print('val-acc@1:', eval_top1)
        print('val-acc@5:', eval_top5)
        print(f'Saving checkpoint after {epoch_num} epoch...')
        save_epoch_model(model, optimizer, tb_loss_step, epoch_num, checkpoints_dir)


def train_val_split(dataset, validation_size):
    val_size = int(len(dataset) * validation_size)
    train_size = len(dataset) - val_size
    return torch.utils.data.random_split(dataset, [train_size, val_size])


def run(model_name: str = 'ViT-L/14@336px', dataset_name: str = 'CIFAR100', validation_size: float = 0.25, learning_rate: float = 1e-5,
        batch_size: int = 32, num_workers: int = 2, epochs_num: int = 10, checkpoints_dir: str = 'checkpoints', device: str = 'cuda',
        random_state: int = 42):
    print(f'{model_name=}, {dataset_name=}, {validation_size=}, {learning_rate=}, {batch_size=}, {num_workers=}, {epochs_num=}, {checkpoints_dir=}, {device=}')
    zero_shot.set_random_state(random_state)
    device = torch.device(device)
    checkpoints_dir = Path(checkpoints_dir)

    clip_model, preprocess = clip.load(model_name, device, jit=False)
    dataset = zero_shot.get_dataset(dataset_name, preprocess)
    train_dataset, val_dataset = train_val_split(dataset, validation_size)
    print(f'train-size={len(train_dataset)}, val-size={len(val_dataset)}')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    adapter_model = ClipAdapter(clip_model, dataset_name, model_name)
    loss = nn.CrossEntropyLoss()
    parameters = (adapter_model.vision_adapter.parameters(), adapter_model.text_adapter.parameters())
    optimizer = torch.optim.Adam(itertools.chain(*parameters), lr=learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
    
    train_model(train_loader, val_loader, adapter_model, loss, optimizer, device, epochs_num, checkpoints_dir)
    

if __name__ == '__main__':
    fire.Fire(run)
