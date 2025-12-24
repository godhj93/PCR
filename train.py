from pathlib import Path
import logging
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from termcolor import colored
from hydra.core.hydra_config import HydraConfig
import torch.nn as nn
# PyTorch, TensorBoard 예시를 위해 import
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.data import data_loader
from utils.common import train_one_epoch, AverageMeter, count_parameters
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
    log.info(f"Model Parameters: {count_parameters(model):,}")
    
    # optimizer = torch.optim.SGD(model.parameters(), lr=cfg.optimizer.lr)
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.loss)
    
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
                        loss_fn = loss_fn,
                        epoch = epoch,
                        metric = {'train': AvgMeter_train, 'val': AvgMeter_val},
                        cfg = cfg)
        
        train_loss = result['train_loss']
        val_loss = result['val_loss']
        
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", val_loss, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
        log.info(colored(f"Epoch [{epoch}/{cfg.training.epochs}] - Train Loss: {train_loss:.4f}, Test Loss: {val_loss:.4f}", "cyan"))

        # save checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, checkpoint_dir / "ckpt.pt")
            log.info(f"Saved Best Model with Test Loss: {best_loss:.4f} at Epoch {epoch}")
        
    writer.close()
    log.info("===== Training End =====")

if __name__ == "__main__":
    
    train()
