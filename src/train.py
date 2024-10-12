"""
Copyright (C) 2024 Yukara Ikemiya
"""

import sys
sys.dont_write_bytecode = True

# DDP
from accelerate import Accelerator, DistributedDataParallelKwargs, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration

import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from utils.torch_common import get_world_size, count_parameters, set_seed
from trainer import Trainer


@hydra.main(version_base=None, config_path='../configs/', config_name="default.yaml")
def main(cfg: DictConfig):

    # Update config if ckpt_dir is specified (training resumption)
    if cfg.trainer.ckpt_dir is not None:
        overrides = HydraConfig.get().overrides.task
        overrides = [e for e in overrides if isinstance(e, str)]
        override_conf = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_conf)

        # Load checkpoint configuration
        cfg_ckpt = OmegaConf.load(f'{cfg.trainer.ckpt_dir}/config.yaml')
        cfg = OmegaConf.merge(cfg_ckpt, override_conf)

    # HuggingFace Accelerate for distributed training

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    dl_config = DataLoaderConfiguration(split_batches=True)
    p_config = ProjectConfiguration(project_dir=cfg.trainer.output_dir)
    accel = Accelerator(
        mixed_precision=cfg.trainer.amp,
        dataloader_config=dl_config,
        project_config=p_config,
        kwargs_handlers=[ddp_kwargs],
        log_with='wandb'
    )

    accel.init_trackers(cfg.trainer.logger.project_name, config=OmegaConf.to_container(cfg),
                        init_kwargs={"wandb": {"name": cfg.trainer.logger.run_name, "dir": cfg.trainer.output_dir}})

    if accel.is_main_process:
        print("->->-> DDP Initialized.")
        print(f"->->-> World size (Number of GPUs): {get_world_size()}")

    set_seed(cfg.trainer.seed)

    # Dataset

    batch_size = cfg.trainer.batch_size
    num_workers = cfg.trainer.num_workers
    train_dataset = hydra.utils.instantiate(cfg.data.train)
    test_dataset = hydra.utils.instantiate(cfg.data.test)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))

    # Model

    generator = hydra.utils.instantiate(cfg.model.generator.model)
    discriminator = hydra.utils.instantiate(cfg.model.discriminator.model)

    # Loss modules

    mrstft_loss = hydra.utils.instantiate(cfg.model.loss.mrstft)
    melmae_loss = hydra.utils.instantiate(cfg.model.loss.melmae)

    # Optimizer

    opt_G = hydra.utils.instantiate(cfg.optimizer.G.optimizer)(params=generator.parameters())
    sche_G = hydra.utils.instantiate(cfg.optimizer.G.scheduler)(optimizer=opt_G)
    opt_D = hydra.utils.instantiate(cfg.optimizer.D.optimizer)(params=discriminator.parameters())
    sche_D = hydra.utils.instantiate(cfg.optimizer.D.scheduler)(optimizer=opt_D)

    # Log

    generator.train()
    discriminator.train()
    num_params_g = count_parameters(generator.generator) / 1e6
    num_params_d = count_parameters(discriminator) / 1e6
    if accel.is_main_process:
        print("=== Parameters ===")
        print(f"\tGenerator:\t{num_params_g:.2f} [million]")
        print(f"\tDiscriminator:\t{num_params_d:.2f} [million]")
        print("=== Dataset ===")
        print(f"\tBatch size: {cfg.trainer.batch_size}")
        print("\tTrain data:")
        print(f"\t\tFiles:\t{len(train_dataset.filenames)}")
        print(f"\t\tChunks:\t{len(train_dataset)}")
        print(f"\t\tBatches:\t{len(train_dataset)//cfg.trainer.batch_size}")
        print("\tTest data:")
        print(f"\t\tFiles:\t{len(test_dataset.filenames)}")
        print(f"\t\tChunks:\t{len(test_dataset)}")
        print(f"\t\tBatches:\t{len(test_dataset)//cfg.trainer.batch_size}")

    # Prepare for DDP

    (train_dataloader, test_dataloader, generator, discriminator,
     mrstft_loss, melmae_loss, opt_G, sche_G, opt_D, sche_D) = accel.prepare(
        train_dataloader, test_dataloader, generator, discriminator,
        mrstft_loss, melmae_loss, opt_G, sche_G, opt_D, sche_D)

    # Start training

    trainer = Trainer(
        generator, discriminator, {'mrstft': mrstft_loss, 'melmae': melmae_loss},
        opt_G, sche_G, opt_D, sche_D, train_dataloader, test_dataloader,
        accel, cfg, ckpt_dir=cfg.trainer.ckpt_dir
    )

    trainer.start_training()


if __name__ == '__main__':
    main()
