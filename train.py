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
from utils.utils import train_one_epoch

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
    
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.optimizer.lr)
    loss_fn = nn.MSELoss()
    
    log.info("===== Training Start =====")
    pbar = tqdm(range(1, cfg.training.epochs + 1), desc="Training Epochs")
    
    best_loss = float('inf')
    
    for epoch in pbar:
        # 가상의 loss 계산
        
        loss, acc = train_one_epoch(model = model, 
                        train_loader = train_loader,
                        optimizer = optimizer,
                        loss_fn = loss_fn,
                        epoch = epoch,
                        cfg = cfg)
        
        # TensorBoard에 loss 기록
        writer.add_scalar("Loss/train", loss['train_loss'], epoch)
        writer.add_scalar("Loss/test", loss['test_loss'], epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("Accuracy/test", acc, epoch)
        log.info(f"Epoch [{epoch}/{cfg.training.epochs}] - Train Loss: {loss['train_loss']:.4f}, Test Loss: {loss['test_loss']:.4f}, Test Acc: {acc:.4f}")

        # save checkpoint
        if loss['test_loss'] < best_loss:
            best_loss = loss['test_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, checkpoint_dir / "best.pt")
            log.info(f"Saved Best Model with Test Loss: {best_loss:.4f} at Epoch {epoch}")
        
    writer.close()
    log.info("===== Training End =====")


if __name__ == "__main__":
    
    train()
