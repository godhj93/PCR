from pathlib import Path
import logging
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from termcolor import colored
from hydra.core.hydra_config import HydraConfig

# PyTorch, TensorBoard 예시를 위해 import
import torch
from torch.utils.tensorboard import SummaryWriter

log = logging.getLogger(__name__)

# 모델 및 데이터 로더 등은 별도로 정의되어 있다고 가정합니다.
# def get_model(cfg): ...
# def get_dataloader(cfg): ...

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    """
    Hydra의 자동 생성 디렉토리 내에 TensorBoard 로그와 모델 가중치를 저장하는 학습 함수
    """
    # 1. Hydra가 자동으로 생성하고 변경해준 현재 작업 디렉토리 경로를 가져옵니다.
    #    이제부터 모든 경로는 이곳을 기준으로 생성됩니다.
    runtime_output = HydraConfig.get().runtime.output_dir
    output_dir = Path(runtime_output)
    log.info(f"===== Configuration =====\n{colored(OmegaConf.to_yaml(cfg), 'green')}")
    log.info(f"Output Directory: {output_dir}")
    # 2. TensorBoard 및 가중치를 저장할 하위 디렉토리 생성
    tensorboard_dir = output_dir / "tensorboard"
    checkpoint_dir = output_dir / "checkpoints"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"TensorBoard path: {tensorboard_dir}")
    log.info(f"Checkpoint path: {checkpoint_dir}")

    # 3. TensorBoard SummaryWriter 초기화
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # --- 가상의 학습 루프 ---
    model = torch.nn.Linear(10, 2) # 예시 모델
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.optimizer.lr)

    log.info("===== Training Start =====")
    pbar = tqdm(range(1, cfg.training.epochs + 1), desc="Training Epochs")
    for epoch in pbar:
        # 가상의 loss 계산
        loss = 1.0 / epoch 

        # TensorBoard에 loss 기록
        writer.add_scalar("Loss/train", loss, epoch)
        pbar.set_description(f"Epoch {epoch}, Loss: {loss:.4f}")
        
    
    writer.close()
    log.info("===== Training End =====")


if __name__ == "__main__":
    train()
