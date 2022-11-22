from argparse import Namespace
from typing import List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from pathlib import Path
import sys
modulePath = str(Path(__file__).resolve().parents[1])
print(f'modulePath: {modulePath}')
sys.path.append(modulePath)

from image_to_latex.data import Im2Latex
from image_to_latex.lit_models import LitResNetTransformer


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    datamodule = Im2Latex(**cfg.data)
    datamodule.setup()
    
    # 将 outputs/yy-mm-dd/wandb/latest-run/files/image-to-latex/epoch=XXX/loss=YYY/cer=ZZZ.ckpt
    # 转移到 outputs/ 下进行
    checkpoints = list((Path(modulePath) / 'outputs').rglob('*.ckpt'))
    checkpoints.extend(list((Path(modulePath) / 'outputs').rglob('*.pt')))
    if len(checkpoints) > 0:
        lit_model = LitResNetTransformer.load_from_checkpoint(checkpoints[0])
    else:
        lit_model = LitResNetTransformer(**cfg.lit_model)

    callbacks: List[Callback] = []
    if cfg.callbacks.model_checkpoint:
        callbacks.append(ModelCheckpoint(**cfg.callbacks.model_checkpoint))
    if cfg.callbacks.early_stopping:
        callbacks.append(EarlyStopping(**cfg.callbacks.early_stopping))

    logger: Optional[WandbLogger] = None
    if cfg.logger:
        logger = WandbLogger(**cfg.logger)

    trainer = Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)

    if trainer.logger:
        trainer.logger.log_hyperparams(Namespace(**cfg))

    trainer.tune(lit_model, datamodule=datamodule)
    trainer.fit(lit_model, datamodule=datamodule)
    trainer.test(lit_model, datamodule=datamodule)


if __name__ == "__main__":
    main()
