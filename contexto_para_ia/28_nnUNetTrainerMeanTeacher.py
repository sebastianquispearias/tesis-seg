"""
nnU-Net Mean Teacher SSL trainer (ONLINE Mean Teacher — NOT self-training).

Goal: apply SSL (Mean Teacher) INSIDE nnU-Net's self-configuring pipeline to test
whether it improves the strong supervised nnU-Net. The valid comparison is within
nnU-Net: supervised nnU-Net vs nnU-Net + Mean Teacher.

Phase E2 (design "A" — shared geometry + photometric perturbation):
  - EMA teacher (Phase C) + unlabeled dataloader over Dataset505 (Phase D);
  - total loss = sup_loss + lambda_t * consistency_loss;
  - lambda_t = 0 before semi_start_epoch, linear ramp after. Aligned with the
    U-Net++ Mean Teacher: lambda_t_max = 0.05, ema_decay = 0.99. NO tau (tau
    belongs to pseudo-labeling, not Mean Teacher).
  - nnU-Net augments each unlabeled crop ONCE (shared geometry — crucial, because
    nnU-Net's spatial aug reaches +/-180 deg rotation and cannot be applied
    independently to each view without breaking pixel alignment). The teacher sees
    that crop; the student sees the SAME crop + an added STRONG photometric
    perturbation (noise + brightness/contrast, no spatial change). This is the
    canonical Mean Teacher perturbation (input noise) and keeps the softmax-MSE
    consistency valid because both predictions share geometry.
  - consistency = MSE between student and teacher softmax on the HIGHEST-res
    deep-supervision output.

  Reported as a complementary single-run experiment (supervised nnU-Net vs
  nnU-Net + MT), not as a formal multi-seed result.

Usage (after copying to nnU-Net's variants directory):
    nnUNetv2_train 501 2d 0 -tr nnUNetTrainerMeanTeacher
"""
import os
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import autocast
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class


class nnUNetTrainerMeanTeacher(nnUNetTrainer):
    """
    EMA teacher + unlabeled loader + consistency loss (shared geometry, strong
    photometric perturbation on the student).
    """

    UNLABELED_DATASET = 'Dataset505_VFSS_UNLABELED'

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # --- schedule (env-overridable so the notebook controls it without edits) ---
        # Defaults are a short smoke test; the notebook sets MT_NUM_EPOCHS=1000 etc.
        # for the full single-run comparison.
        self.num_epochs = int(os.environ.get('MT_NUM_EPOCHS', 8))
        self.semi_start_epoch = int(os.environ.get('MT_SEMI_START', 3))
        self.semi_warmup_epochs = int(os.environ.get('MT_WARMUP', 3))
        # --- MT hyperparameters aligned with the U-Net++ pipeline (NO tau) ---
        self.ema_decay = float(os.environ.get('MT_EMA_DECAY', 0.99))
        self.lambda_t_max = float(os.environ.get('MT_LAMBDA', 0.05))
        # --- strong (student) photometric perturbation; geometry stays shared ---
        self.strong_noise_std = 0.10
        self.strong_brightness = 0.10
        self.strong_contrast = 0.10

        self.teacher_network = None
        self.dataloader_unlabeled = None
        self._mt_logged = False
        self.print_to_log_file("=" * 60)
        self.print_to_log_file(">>> CUSTOM TRAINER LOADED: nnUNetTrainerMeanTeacher <<<")
        self.print_to_log_file(f"    Phase: E2 (Mean Teacher active, shared-geometry + strong photometric, {self.num_epochs} epochs)")
        self.print_to_log_file(f"    ema_decay={self.ema_decay} lambda_t_max={self.lambda_t_max} "
                               f"semi_start_epoch={self.semi_start_epoch} warmup={self.semi_warmup_epochs} (NO tau)")
        self.print_to_log_file(f"    strong photometric: noise_std={self.strong_noise_std} "
                               f"brightness={self.strong_brightness} contrast={self.strong_contrast}")
        self.print_to_log_file("=" * 60)

    # --- helpers ---------------------------------------------------------
    @staticmethod
    def _unwrap(net):
        if hasattr(net, 'module'):
            net = net.module
        if hasattr(net, '_orig_mod'):
            net = net._orig_mod
        return net

    @staticmethod
    def _highest_res(out):
        return out[0] if isinstance(out, (list, tuple)) else out

    def _lambda_t(self):
        e = self.current_epoch
        if e < self.semi_start_epoch:
            return 0.0
        if self.semi_warmup_epochs <= 0:
            return self.lambda_t_max
        p = min(1.0, (e - self.semi_start_epoch) / float(self.semi_warmup_epochs))
        return self.lambda_t_max * p

    def _strong_perturb(self, x):
        # Strong PHOTOMETRIC perturbation only (no spatial change) so the student
        # view stays pixel-aligned with the teacher's view -> softmax-MSE is valid.
        # nnU-Net's spatial aug is already applied (shared) upstream in the loader.
        n = x.shape[0]
        contrast = 1.0 + (torch.rand(n, 1, 1, 1, device=x.device) * 2 - 1) * self.strong_contrast
        bright = (torch.rand(n, 1, 1, 1, device=x.device) * 2 - 1) * self.strong_brightness
        noise = torch.randn_like(x) * self.strong_noise_std
        return x * contrast + bright + noise

    # --- teacher lifecycle (Phase C) ------------------------------------
    def initialize(self):
        super().initialize()
        # lambda_t_max == 0 -> pure supervised control (no teacher, no unlabeled
        # loader): the exact same trainer with SSL switched off, for a clean match.
        if self.lambda_t_max > 0.0 and self.teacher_network is None:
            student = self._unwrap(self.network)
            self.teacher_network = deepcopy(student).to(self.device)
            for p in self.teacher_network.parameters():
                p.requires_grad_(False)
            self.teacher_network.eval()
            self.print_to_log_file(
                f"    [Phase C] teacher_network created via deepcopy "
                f"(no grad, eval, ema_decay={self.ema_decay})")

    @torch.no_grad()
    def _update_teacher(self):
        d = self.ema_decay
        student = self._unwrap(self.network)
        for tp, sp in zip(self.teacher_network.parameters(), student.parameters()):
            tp.mul_(d).add_(sp.detach(), alpha=1.0 - d)
        for tb, sb in zip(self.teacher_network.buffers(), student.buffers()):
            tb.copy_(sb)

    # --- unlabeled dataloader (Phase D) ---------------------------------
    def _build_unlabeled_dataloader(self):
        from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

        folder = join(nnUNet_preprocessed, self.UNLABELED_DATASET,
                      self.configuration_manager.data_identifier)
        ds_class = infer_dataset_class(folder)
        identifiers = ds_class.get_identifiers(folder)
        dataset_unlab = ds_class(folder, identifiers, folder_with_segs_from_previous_stage=None)

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        (rotation_for_DA, do_dummy_2d, initial_patch_size,
         mirror_axes) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # nnU-Net's training transforms produce ONE augmented crop (shared geometry).
        # The strong photometric perturbation for the student is added in train_step.
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        dl = nnUNetDataLoader(
            dataset_unlab, self.batch_size, initial_patch_size,
            self.configuration_manager.patch_size, self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling)

        gen = SingleThreadedAugmenter(dl, None)
        _ = next(gen)  # warm up
        self.dataloader_unlabeled = gen
        self.print_to_log_file(
            f"    [Phase D] unlabeled dataloader built over {self.UNLABELED_DATASET} "
            f"({len(identifiers)} cases)")

    def on_train_start(self):
        super().on_train_start()
        if self.lambda_t_max > 0.0 and self.dataloader_unlabeled is None:
            self._build_unlabeled_dataloader()

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._mt_logged = False

    def on_train_end(self):
        gen = self.dataloader_unlabeled
        if gen is not None and hasattr(gen, '_finish'):
            gen._finish()
        super().on_train_end()

    # --- checkpointing: persist the EMA teacher so resume (--c) restores it ---
    def save_checkpoint(self, filename: str) -> None:
        super().save_checkpoint(filename)
        if self.local_rank == 0 and not self.disable_checkpointing and self.teacher_network is not None:
            ckpt = torch.load(filename, map_location='cpu', weights_only=False)
            ckpt['teacher_weights'] = self.teacher_network.state_dict()
            torch.save(ckpt, filename)

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        super().load_checkpoint(filename_or_checkpoint)
        if isinstance(filename_or_checkpoint, str):
            ckpt = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
        else:
            ckpt = filename_or_checkpoint
        if self.teacher_network is not None and 'teacher_weights' in ckpt:
            self.teacher_network.load_state_dict(ckpt['teacher_weights'])
            self.print_to_log_file("    [MT] teacher_network weights restored from checkpoint")

    # --- training step (Mean Teacher) -----------------------------------
    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        lam = self._lambda_t()
        self.optimizer.zero_grad(set_to_none=True)

        # --- supervised loss + backward (frees labeled graph before unlabeled fwd) ---
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            sup_loss = self.loss(output, target)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(sup_loss).backward()
        else:
            sup_loss.backward()

        # --- consistency + backward (shared geometry; student = strong photometric) ---
        cons_val = 0.0
        if lam > 0.0:
            ub = next(self.dataloader_unlabeled)
            u = ub['data'].to(self.device, non_blocking=True)   # shared augmented crop
            student_view = self._strong_perturb(u)              # strong (same geometry)
            with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                s_out = self._highest_res(self.network(student_view))
                with torch.no_grad():
                    t_out = self._highest_res(self.teacher_network(u))   # teacher on clean shared crop
                cons_loss = F.mse_loss(torch.softmax(s_out, 1), torch.softmax(t_out, 1))
                weighted = lam * cons_loss
            if self.grad_scaler is not None:
                self.grad_scaler.scale(weighted).backward()
            else:
                weighted.backward()
            cons_val = float(cons_loss.detach())

        # --- clip + step ---
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        if self.teacher_network is not None:
            self._update_teacher()

        if not self._mt_logged:
            self.print_to_log_file(
                f"    [MT] epoch {self.current_epoch}: lambda_t={lam:.4f} "
                f"sup_loss={float(sup_loss.detach()):.4f} cons_loss={cons_val:.4f}")
            self._mt_logged = True

        total = float(sup_loss.detach()) + lam * cons_val
        return {'loss': total}
