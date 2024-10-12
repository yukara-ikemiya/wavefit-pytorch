"""
Copyright (C) 2024 Yukara Ikemiya
"""

import os

import torch
import torchaudio
import wandb

from model import WaveFit, Discriminator
from utils.logging import MetricsLogger
from utils.torch_common import sort_dict, print_once
from utils.viz import spectrogram_image


class Trainer:
    def __init__(
        self,
        G,                  # Generator (WaveFit)
        D,                  # Discriminator
        loss_modules,       # Loss modules
        opt_G, sche_G,      # Optimizer/scheduler of G
        opt_D, sche_D,      # Optimizer/scheduler of D
        train_dataloader,
        test_dataloader,
        accel,              # Accelerator object
        cfg,                # Configurations
        ckpt_dir=None
    ):
        self.G: WaveFit = accel.unwrap_model(G)
        self.D: Discriminator = accel.unwrap_model(D)
        self.loss_modules = loss_modules
        self.opt_G = opt_G
        self.sche_G = sche_G
        self.opt_D = opt_D
        self.sche_D = sche_D
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.accel = accel
        self.cfg = cfg
        self.cfg_t = cfg.trainer
        self.loss_lambdas = cfg.model.loss.lambdas
        self.EPS = 1e-8

        self.logger = MetricsLogger()           # Logger for WandB
        self.logger_print = MetricsLogger()     # Logger for printing
        self.logger_test = MetricsLogger()      # Logger for test

        self.states = {'global_step': 0, 'best_metrics': float('inf'), 'latest_metrics': float('inf')}

        # time measurement
        self.s_event = torch.cuda.Event(enable_timing=True)
        self.e_event = torch.cuda.Event(enable_timing=True)

        # resume training
        if ckpt_dir is not None:
            self.__load_ckpt(ckpt_dir)

    def start_training(self):
        """
        Start training with infinite loops
        """
        self.G.train()
        self.D.train()
        self.s_event.record()

        print_once("\n[ Started training ]\n")

        while True:
            for batch in self.train_dataloader:
                # Update
                metrics = self.run_step(batch)

                # Test (validation)
                if self.__its_time(self.cfg_t.logging.n_step_test):
                    self.__test()

                if self.accel.is_main_process:
                    self.logger.add(metrics)
                    self.logger_print.add(metrics)

                    # Log
                    if self.__its_time(self.cfg_t.logging.n_step_log):
                        self.__log_metrics()

                    # Print
                    if self.__its_time(self.cfg_t.logging.n_step_print):
                        self.__print_metrics()

                    # Save checkpoint
                    if self.__its_time(self.cfg_t.logging.n_step_ckpt):
                        self.__save_ckpt()

                    # Sample
                    if self.__its_time(self.cfg_t.logging.n_step_sample):
                        self.__sampling()

                self.states['global_step'] += 1

    def run_step(self, batch, train: bool = True):
        """ One training step """

        audios, _ = batch

        # Prepare inputs

        mel_spec, r_noise, pstft_spec = self.G.mel.get_shaped_noise(audios)

        # Generator update

        if train:
            self.opt_G.zero_grad()

        preds = self.G(r_noise, torch.log(mel_spec.clamp(min=self.EPS)), pstft_spec)

        losses = {}
        for idx, pred in enumerate(preds):
            losses_i = {}
            losses_i.update(self.loss_modules['mrstft'](pred, audios))
            losses_i.update(self.loss_modules['melmae'](pred, audios))
            losses_i.update(self.D.compute_G_loss(pred, audios))
            for k, v in losses_i.items():
                losses[k] = losses.get(k, 0.) + v / len(preds)
                losses[f'{k}/iter-{idx+1}'] = v.detach()  # for logging

        loss_g = 0.
        for k in self.loss_lambdas.keys():
            loss_g += losses[k] * self.loss_lambdas[k]

        if train:
            self.accel.backward(loss_g)
            if self.accel.sync_gradients:
                self.accel.clip_grad_norm_(self.G.parameters(), self.cfg_t.max_grad_norm)
            self.opt_G.step()
            self.sche_G.step()

        losses['G/loss'] = loss_g.detach()

        # Discriminator update

        if train:
            self.opt_D.zero_grad()

        loss_d_real = self.D.compute_D_loss(audios, mode='real')['D/loss']

        # NOTE: Discriminator loss is also computed for all intermediate predictions (Sec.4.2)
        loss_d_fake = 0.
        for idx, pred in enumerate(preds):
            loss_d_fake_ = self.D.compute_D_loss(pred.detach(), mode='fake')['D/loss']
            loss_d_fake += loss_d_fake_ / len(preds)
            losses[f'D/loss/iter-{idx+1}'] = loss_d_fake_.detach()  # for logging

        losses['D/loss/real'] = loss_d_real.detach()
        losses['D/loss/fake'] = loss_d_fake.detach()
        losses['D/loss'] = loss_d_real + loss_d_fake

        if train:
            self.accel.backward(losses['D/loss'])
            if self.accel.sync_gradients:
                self.accel.clip_grad_norm_(self.D.parameters(), self.cfg_t.max_grad_norm)
            self.opt_D.step()
            self.sche_D.step()

        return {k: v.detach() for k, v in losses.items()}

    @torch.no_grad()
    def __test(self):
        self.G.eval()
        self.D.eval()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        for batch in self.test_dataloader:
            metrics = self.run_step(batch, train=False)
            self.logger_test.add(metrics)

        end.record()
        torch.cuda.synchronize()
        p_time = start.elapsed_time(end) / 1000.  # [sec]

        metrics = self.logger_test.pop()
        # gather from all processes
        metrics_g = {}
        for k, v in metrics.items():
            metrics_g[k] = self.accel.gather(v).mean()

        if self.accel.is_main_process:
            # log and print
            step = self.states['global_step']
            metrics_g = sort_dict(metrics_g)
            self.accel.log({f"test/{k}": v for k, v in metrics_g.items()}, step=step)
            metrics_g = {k: v.item() for k, v in metrics_g.items() if 'iter' not in k}
            s = f"[Test] ({p_time:.1e} [sec]): " + ' / '.join([f"[{k}] - {v:.3e}" for k, v in metrics_g.items()])
            print(s)

        # update states
        m_for_ckpt = self.cfg_t.logging.metrics_for_best_ckpt
        m_latest = sum([metrics_g[k] * self.loss_lambdas[k] for k in m_for_ckpt])
        self.states['latest_metrics'] = m_latest
        if m_latest < self.states['best_metrics']:
            self.states['best_metrics'] = m_latest

        self.G.train()
        self.D.train()

    @torch.no_grad()
    def __sampling(self):
        self.G.eval()

        # randomly select samples
        dataset = self.test_dataloader.dataset
        n_sample = self.cfg_t.logging.n_samples
        idxs = torch.randint(len(dataset), size=(n_sample,))
        audios = torch.stack([dataset[idx][0] for idx in idxs], dim=0).to(self.accel.device)

        # sample
        mel_spec, r_noise, pstft_spec = self.G.mel.get_shaped_noise(audios)
        preds = self.G(r_noise, torch.log(mel_spec.clamp(min=self.EPS)), pstft_spec, return_only_last=True)[-1]
        mel_spec_pred = self.G.mel.compute_mel(preds.squeeze(1))
        mel_spec_inoise = self.G.mel.compute_mel(r_noise.squeeze(1))

        # save/log audios
        out_dir = self.cfg_t.output_dir + '/sample'
        os.makedirs(out_dir, exist_ok=True)
        columns = ['gt (audio)', 'gt (spec)', 'init_noise (audio)', 'init_noise (spec)', 'pred (audio)', 'pred (spec)']
        table_audio = wandb.Table(columns=columns)
        for idx in range(audios.shape[0]):
            torchaudio.save(f"{out_dir}/item-{idx}_gt.wav", audios[idx].cpu(), sample_rate=dataset.sr, encoding='PCM_F')
            torchaudio.save(f"{out_dir}/item-{idx}_pred.wav", preds[idx].cpu(), sample_rate=dataset.sr, encoding='PCM_F')

            data = [
                wandb.Audio(audios[idx].cpu().numpy().T, sample_rate=dataset.sr),
                wandb.Image(spectrogram_image(mel_spec[idx].cpu().numpy())),
                wandb.Audio(r_noise[idx].cpu().numpy().T, sample_rate=dataset.sr),
                wandb.Image(spectrogram_image(mel_spec_inoise[idx].cpu().numpy())),
                wandb.Audio(preds[idx].cpu().numpy().T, sample_rate=dataset.sr),
                wandb.Image(spectrogram_image(mel_spec_pred[idx].cpu().numpy()))
            ]

            table_audio.add_data(*data)

        self.accel.log({'Samples': table_audio}, step=self.states['global_step'])

        self.G.train()

        print("\t->->-> Sampled.")

    def __save_ckpt(self):
        import shutil
        import json
        from omegaconf import OmegaConf

        out_dir = self.cfg_t.output_dir + '/ckpt'

        # save latest ckpt
        latest_dir = out_dir + '/latest'
        os.makedirs(latest_dir, exist_ok=True)
        ckpts = {'generator': self.G,
                 'discriminator': self.D,
                 'opt_g': self.opt_G,
                 'opt_d': self.opt_D,
                 'sche_g': self.sche_G,
                 'sche_d': self.sche_D}
        for name, m in ckpts.items():
            torch.save(m.state_dict(), f"{latest_dir}/{name}.pth")

        # save states and configuration
        OmegaConf.save(self.cfg, f"{latest_dir}/config.yaml")
        with open(f"{latest_dir}/states.json", mode="wt", encoding="utf-8") as f:
            json.dump(self.states, f, indent=2)

        # save best ckpt
        if self.states['latest_metrics'] == self.states['best_metrics']:
            shutil.copytree(latest_dir, out_dir + '/best', dirs_exist_ok=True)

        print("\t->->-> Saved checkpoints.")

    def __load_ckpt(self, dir: str):
        import json

        print_once(f"\n[Resuming training from the checkpoint directory] -> {dir}")
        ckpts = {'generator': self.G,
                 'discriminator': self.D,
                 'opt_g': self.opt_G,
                 'opt_d': self.opt_D,
                 'sche_g': self.sche_G,
                 'sche_d': self.sche_D}
        for k, v in ckpts.items():
            v.load_state_dict(torch.load(f"{dir}/{k}.pth", weights_only=False))

        with open(f"{dir}/states.json", mode="rt", encoding="utf-8") as f:
            self.states.update(json.load(f))

    def __log_metrics(self, sort_by_key: bool = True):
        metrics = self.logger.pop()
        # learning rate
        metrics['G/lr'] = self.sche_G.get_last_lr()[0]
        metrics['D/lr'] = self.sche_D.get_last_lr()[0]
        if sort_by_key:
            metrics = sort_dict(metrics)

        self.accel.log(metrics, step=self.states['global_step'])

    def __print_metrics(self, sort_by_key: bool = True):
        self.e_event.record()
        torch.cuda.synchronize()
        p_time = self.s_event.elapsed_time(self.e_event) / 1000.  # [sec]

        metrics = self.logger_print.pop()
        # tensor to scalar
        metrics = {k: v.item() for k, v in metrics.items() if 'iter' not in k}
        # learning rate
        metrics['G/lr'] = self.sche_G.get_last_lr()[0]
        metrics['D/lr'] = self.sche_D.get_last_lr()[0]
        if sort_by_key:
            metrics = sort_dict(metrics)

        step = self.states['global_step']
        s = f"Step {step} ({p_time:.1e} [sec]): " + ' / '.join([f"[{k}] - {v:.3e}" for k, v in metrics.items()])
        print(s)

        self.s_event.record()

    def __its_time(self, itv: int):
        return (self.states['global_step'] - 1) % itv == 0
