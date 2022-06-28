import itertools
from pathlib import Path

import torch
import clip
import fire
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from results_reproduce import zero_shot


class ClipAdapter(nn.Module):
    def __init__(self, clip_model, classnames, templates, output_dim=None):
        super().__init__()

        self.clip_model = clip_model
        self.classnames = classnames
        self.templates = templates
        zeroshot_weights = zero_shot.zeroshot_classifier(clip_model, classnames, templates)
        self.zeroshot_weights = nn.parameter.Parameter(zeroshot_weights, requires_grad=False) 

        embed_dim = self.clip_model.text_projection.size(dim=1)
        output_dim = output_dim or embed_dim
        print(f'Adapter: {embed_dim=}, {output_dim=}')
        self.vision_adapter = nn.Linear(embed_dim, output_dim)
        self.text_adapter = nn.Linear(embed_dim, output_dim)

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
    epoch_dir = checkpoints_dir / f'model_epoch_{epoch_num}'
    epoch_dir.mkdir(parents=True, exist_ok=True)

    def save_data(data, data_name):
        with open(epoch_dir / f'{data_name}.ckpt', 'wb') as f:
            torch.save(data, f)

    save_data(model.state_dict(), 'model')
    save_data(optimizer.state_dict(), 'optimizer')
    save_data(tb_loss_step, 'tb_loss_step')


def train_model(loader, model, loss, optimizer, device, epochs_num, checkpoints_dir: Path):
    log_dir = checkpoints_dir / 'tb_runs'
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_writer = SummaryWriter(log_dir)
    tb_loss_step = 0

    for epoch_num in range(1, epochs_num + 1):
        print(f'Running epoch {epoch_num}...')
        model, epoch_loss, optimizer, tb_loss_step = train_epoch(loader, model, loss, optimizer, device, summary_writer, tb_loss_step)
        eval_top1, eval_top5 = eval_model(loader, model)
        summary_writer.add_scalar('epoch-sum-loss', epoch_loss, epoch_num)
        summary_writer.add_scalar('epoch-acc@1', eval_top1, epoch_num)
        summary_writer.add_scalar('epoch-acc@5', eval_top5, epoch_num)
        print('acc@1:', eval_top1)
        print('acc@5:', eval_top5)
        print(f'Saving checkpoint after {epoch_num} epoch...')
        save_epoch_model(model, optimizer, tb_loss_step, epoch_num, checkpoints_dir)


def run(model_name: str = 'ViT-L/14@336px', dataset_name: str = 'CIFAR100', learning_rate: float = 1e-5, batch_size: int = 32,
        num_workers: int = 2, epochs_num: int = 10, checkpoints_dir: str = 'checkpoints', device: str = 'cuda'):
    print(f'{model_name=}, {dataset_name=}, {learning_rate=}, {batch_size=}, {num_workers=}, {device=}')
    device = torch.device(device)
    checkpoints_dir = Path(checkpoints_dir)

    clip_model, preprocess = clip.load(model_name, device, jit=False)
    dataset = zero_shot.get_dataset(dataset_name, preprocess)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    classnames, templates = zero_shot.load_promts(dataset_name)

    adapter_model = ClipAdapter(clip_model, classnames, templates)
    loss = nn.CrossEntropyLoss()
    parameters = (adapter_model.vision_adapter.parameters(), adapter_model.text_adapter.parameters())
    optimizer = torch.optim.Adam(itertools.chain(*parameters), lr=learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
    
    train_model(loader, adapter_model, loss, optimizer, device, epochs_num, checkpoints_dir)
    

if __name__ == '__main__':
    fire.Fire(run)
