from pathlib import Path
import logging
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from termcolor import colored
from hydra.core.hydra_config import HydraConfig
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.data import data_loader
from utils.common import train_one_epoch, AverageMeter, count_parameters, logging_tensorboard
from termcolor import colored

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    
    runtime_output = HydraConfig.get().runtime.output_dir
    output_dir = Path(runtime_output)
    log.info(f"===== Configuration =====\n{colored(OmegaConf.to_yaml(cfg), 'green')}")
    log.info(f"Output Directory: {output_dir}")
    
    tensorboard_dir = output_dir / "tensorboard"
    checkpoint_dir = output_dir / "weights"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"TensorBoard path: {tensorboard_dir}")
    log.info(f"Checkpoint path: {checkpoint_dir}")

    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Load data and model
    train_loader, test_loader = data_loader(cfg)
    model = hydra.utils.instantiate(cfg.model).to(cfg.device)
    loss_fn = hydra.utils.instantiate(cfg.loss)
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    
    if isinstance(optimizer, torch.optim.SGD):
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150, 225], gamma=0.1)
    elif isinstance(optimizer, torch.optim.AdamW):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)
    else:
        # Keep the learning rate constant if no scheduler is specified
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.training.epochs + 1, gamma=1.0)
    
    log.info(f"Model Parameters: {count_parameters(model):,}")
    log.info("===== Training Start =====")
    AvgMeter_train = AverageMeter()
    AvgMeter_val = AverageMeter()
    best_loss = float('inf')
    
    for epoch in range(1, cfg.training.epochs + 1):
        
        AvgMeter_train.reset()
        AvgMeter_val.reset()
        
        result = train_one_epoch(model = model, 
                        data_loader= {'train': train_loader, 'test': test_loader},
                        optimizer = optimizer,
                        scheduler = scheduler,
                        loss_fn = loss_fn,
                        epoch = epoch,
                        metric = {'train': AvgMeter_train, 'val': AvgMeter_val},
                        cfg = cfg)
        
        logging_tensorboard(writer, result, epoch, optimizer)
        
        log.info(colored(f"Epoch [{epoch}/{cfg.training.epochs}] - "
                        f"Train Loss: {result['train_loss']:.4f}, Test Loss: {result['val_loss']:.4f}, "
                        f"LR: {optimizer.param_groups[0]['lr']:.6f}", "cyan"))

        # save checkpoint
        if result['val_loss'] < best_loss:
            best_loss = result['val_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, checkpoint_dir / "ckpt.pt")
            log.info(colored(f"Saved Best Model with Test Loss: {best_loss:.4f} at Epoch {epoch}", "green"))
        
    writer.close()
    log.info("===== Training End =====")

if __name__ == "__main__":
    
    train()
