from __future__ import annotations

import argparse
import os
from pathlib import Path
from statistics import mode
import yaml
import json
import random
import time
import numpy as np
import warnings

from tests.test_selective_inference import model

# Suppress warnings
warnings.filterwarnings('ignore', message='Initializing zero-element tensors')
warnings.filterwarnings('ignore', message='os.fork.*JAX')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules.loss')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.autograd.graph')
warnings.filterwarnings('ignore', message='Error detected in.*Backward')
warnings.filterwarnings('ignore', message='for fsspec: HTTPFileSystem assuming index is current')


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler

from alphagenome_pytorch import AlphaGenome, AlphaGenomeConfig, TargetScaler, MultinomialLoss, JunctionsLoss, config
from alphagenome_pytorch.splice_dataset import SpliceDataset
from alphagenome_pytorch.samplers import SpeciesGroupedSampler

def exists(v):
    return v is not None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--resume", type=str, default=None, 
                       help="Path to checkpoint to resume training from")
    args = parser.parse_args()
    return args
    
def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_model(model, optimizer, epoch, path):
    # Handle DataParallel models
    if isinstance(model, nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    torch.save({
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, path)

def load_checkpoint(model, optimizer, checkpoint_path, device='cuda'):
    """Load checkpoint and return the starting epoch."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle DataParallel models
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0)
    print(f"  Resumed from epoch {start_epoch}")
    
    return start_epoch

def get_organism_name(organism_idx, species_mapping):
    """Map organism index to organism name used in model heads"""
    # Invert species_mapping to go from index to name
    for name, idx in species_mapping.items():
        if idx == organism_idx:
            return name
    return f'organism_{organism_idx}'

def train_one_epoch(model, dataloader, optimizer, loss_fns, device, species_mapping, heads_to_train=None, scheduler=None, scaler=None, use_amp=True, freeze_backbone=False):

    if freeze_backbone:

        # First, set entire model to eval mode
        model.eval()
    
        # Disable running_var updates in custom BatchRMSNorm layers
        from alphagenome_pytorch.alphagenome import set_update_running_var
        set_update_running_var(model, False)
        
        # Then, only set trainable heads to train mode
        if heads_to_train and hasattr(model, 'heads'):
            for organism, heads in model.heads.items():
                for head_name, head in heads.items():
                    if head_name in heads_to_train:
                        head.train()
                        # Re-enable running_var updates for trainable heads only (if they have BatchRMSNorm)
                        set_update_running_var(head, True)
    else:
        model.train()
    
    total_loss = 0.0
    total_splice_logits_loss = 0.0
    total_splice_usage_loss = 0.0
    total_splice_juncs_loss = 0.0
    num_batches = 0
    step_in_epoch = 0

    for batch in dataloader:
        # Move inputs to device
        dna = batch['dna'].to(device)
        organism_index = batch['organism_index'].to(device)
        splice_donor_idx = batch['splice_donor_idx'].to(device)
        splice_acceptor_idx = batch['splice_acceptor_idx'].to(device)
        context_indices_map = batch['conditions_mask'].to(device)

        # Targets
        splice_labels = batch['splice_labels'].to(device)  # (batch, seq_len)
        splice_usage_target = batch['splice_usage_target'].to(device)  # (batch, seq_len, num_contexts)

        with autocast(device_type='cuda', enabled=use_amp):
            # Forward pass
            preds = model(
                dna,
                organism_index,
                splice_donor_idx=splice_donor_idx,
                splice_acceptor_idx=splice_acceptor_idx
            )
        
        # Since batches are species-specific, preds should only contain one organism
        # assert len(preds) == 1, f"Expected 1 organism in batch, got {len(preds)}"
        # But the model actually calculates predictions for both species
        # so I will only use predictions for batch species for loss calculation

        # Compute losses for batch organism
        losses = []
        batch_organism = organism_index.unique().tolist()
        for org_name, org_preds in preds.items():
            # Skip organisms not in this batch
            if org_name not in [get_organism_name(org_idx, species_mapping) for org_idx in batch_organism]:
                continue
            # Splice logits loss (5-class classification: none, donor, acceptor, etc.)
            if 'splice_sites_classification' in org_preds:
                splice_logits = org_preds['splice_sites_classification']  # (batch, seq_len, 5)
                splice_logits_flat = splice_logits.reshape(-1, splice_logits.shape[-1])
                splice_labels_flat = splice_labels.reshape(-1)
                splice_logits_loss = loss_fns['splice_sites_classification'](splice_logits_flat, splice_labels_flat)
                losses.append(splice_logits_loss)
                total_splice_logits_loss += splice_logits_loss.item()
            
            # Splice usage loss (per-context usage prediction)
            if 'splice_sites_usage' in org_preds:
                splice_usage = org_preds['splice_sites_usage']  # (batch, seq_len, num_contexts_for_organism)
            
                # Get which SSE columns this organism uses
                sse_columns = context_indices_map[0]  # Get first element since all in batch are same organism
                
                # Since batches are species-specific, all sequences are from this organism
                # Select only the relevant SSE columns for this organism
                org_sse_target = splice_usage_target[:, :, sse_columns]  # (batch, seq_len, num_contexts_for_organism)
                
                # Only compute loss at splice site positions (labels != 0)
                splice_site_mask = splice_labels != 0  # (batch, seq_len)
                
                if splice_site_mask.any():
                    # Select only splice site positions
                    splice_usage_at_sites = splice_usage[splice_site_mask]  # (num_sites, num_contexts)
                    org_sse_target_at_sites = org_sse_target[splice_site_mask]  # (num_sites, num_contexts)
                    
                    # Check for NaN/Inf in targets and replace with zeros
                    if torch.isnan(org_sse_target_at_sites).any() or torch.isinf(org_sse_target_at_sites).any():
                        org_sse_target_at_sites = torch.nan_to_num(org_sse_target_at_sites, nan=0.0, posinf=1.0, neginf=0.0)
                    
                    # Skip siites (i.e. rows) where all targets are zero (no signal)
                    if (org_sse_target_at_sites.sum(dim=1) == 0).any():
                        non_zero_mask = org_sse_target_at_sites.sum(dim=1) != 0
                        splice_usage_at_sites = splice_usage_at_sites[non_zero_mask]
                        org_sse_target_at_sites = org_sse_target_at_sites[non_zero_mask]
                    
                    # Skip if predictions contain NaN
                    if not (torch.isnan(splice_usage_at_sites).any() or torch.isinf(splice_usage_at_sites).any()):
                        # Compute MSE loss
                        splice_usage_loss = loss_fns['splice_sites_usage'](splice_usage_at_sites, org_sse_target_at_sites)
                        
                        if not torch.isnan(splice_usage_loss):
                            losses.append(splice_usage_loss)
                            total_splice_usage_loss += splice_usage_loss.item()

            
            # Splice junction loss (donor-acceptor pairings)
            if 'splice_sites_junctions' in org_preds:
                splice_juncs = org_preds['splice_sites_junctions']  # (batch, num_donors, num_acceptors, num_contexts)
                # Create target junction matrix (placeholder - adjust based on your data)
                # For now, use a simple loss
                splice_juncs_loss = loss_fns['splice_sites_junctions'](splice_juncs, splice_juncs)  # Placeholder
                losses.append(splice_juncs_loss * 0)  # Weight down for now
                total_splice_juncs_loss += splice_juncs_loss.item()

        if len(losses) == 0:
            print("No losses computed for this batch!")
            continue
            

        loss = torch.stack(losses).sum()

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None and step_in_epoch > 0:
            scheduler.step()  # Step the scheduler after each optimizer step
        step_in_epoch += 1
        total_loss += loss.item()
        num_batches += 1    

    avg_loss = total_loss / max(num_batches, 1)    
    avg_splice_juncs_loss = total_splice_juncs_loss / max(num_batches, 1)
    avg_splice_logits_loss = total_splice_logits_loss / max(num_batches, 1)    
    avg_splice_usage_loss = total_splice_usage_loss / max(num_batches, 1)
    
    return avg_loss, avg_splice_logits_loss, avg_splice_usage_loss, avg_splice_juncs_loss

@torch.no_grad()
def validate_one_epoch(model, val_loader, loss_fns, species_mapping, device='cuda', use_amp=True):
    """Run validation and return average loss."""
    model.eval()
    
    total_loss = 0.0
    total_splice_logits_loss = 0.0
    total_splice_usage_loss = 0.0
    total_splice_juncs_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        dna = batch['dna'].to(device)
        organism_index = batch['organism_index'].to(device)
        splice_donor_idx = batch['splice_donor_idx'].to(device)
        splice_acceptor_idx = batch['splice_acceptor_idx'].to(device)
        splice_labels = batch['splice_labels'].to(device)
        splice_usage_target = batch['splice_usage_target'].to(device)
        context_indices_map = batch['conditions_mask'].to(device)

        with autocast(device_type='cuda', enabled=use_amp):
            # Forward pass
            preds = model(
                dna,
                organism_index=organism_index,
                splice_donor_idx=splice_donor_idx,
                splice_acceptor_idx=splice_acceptor_idx
            )

        # Compute losses for batch organisms only
        losses = []
        batch_organism = organism_index.unique().tolist()
        
        for org_name, org_preds in preds.items():
            # Skip organisms not in this batch
            if org_name not in [get_organism_name(org_idx, species_mapping) for org_idx in batch_organism]:
                continue
            # Splice logits loss
            if 'splice_sites_classification' in org_preds:
                splice_logits = org_preds['splice_sites_classification']
                splice_logits_flat = splice_logits.reshape(-1, splice_logits.shape[-1])
                splice_labels_flat = splice_labels.reshape(-1)
                splice_logits_loss = loss_fns['splice_sites_classification'](splice_logits_flat, splice_labels_flat)
                losses.append(splice_logits_loss)
                total_splice_logits_loss += splice_logits_loss.item()
            
            # Splice usage loss
            if 'splice_sites_usage' in org_preds:
                splice_usage = org_preds['splice_sites_usage']  # (batch, seq_len, num_contexts_for_organism)
            
                # Get which SSE columns this organism uses
                sse_columns = context_indices_map[0]  # Get first element since all in batch are same organism
                
                # Since batches are species-specific, all sequences are from this organism
                # Select only the relevant SSE columns for this organism
                org_sse_target = splice_usage_target[:, :, sse_columns]  # (batch, seq_len, num_contexts_for_organism)
                
                # Only compute loss at splice site positions (labels != 0)
                splice_site_mask = splice_labels != 0  # (batch, seq_len)

                if splice_site_mask.any():
                    # Select only splice site positions
                    splice_usage_at_sites = splice_usage[splice_site_mask]  # (num_sites, num_contexts)
                    org_sse_target_at_sites = org_sse_target[splice_site_mask]  # (num_sites, num_contexts)
                    
                    # Check for NaN/Inf in targets and replace with zeros
                    if torch.isnan(org_sse_target_at_sites).any() or torch.isinf(org_sse_target_at_sites).any():
                        org_sse_target_at_sites = torch.nan_to_num(org_sse_target_at_sites, nan=0.0, posinf=1.0, neginf=0.0)
                    
                    # Skip siites (i.e. rows) where all targets are zero (no signal)
                    if (org_sse_target_at_sites.sum(dim=1) == 0).any():
                        non_zero_mask = org_sse_target_at_sites.sum(dim=1) != 0
                        splice_usage_at_sites = splice_usage_at_sites[non_zero_mask]
                        org_sse_target_at_sites = org_sse_target_at_sites[non_zero_mask]
                    
                    # Skip if predictions contain NaN
                    if not (torch.isnan(splice_usage_at_sites).any() or torch.isinf(splice_usage_at_sites).any()):
                        # Compute MSE loss
                        splice_usage_loss = loss_fns['splice_sites_usage'](splice_usage_at_sites, org_sse_target_at_sites)
                        
                        if not torch.isnan(splice_usage_loss):
                            losses.append(splice_usage_loss)
                            total_splice_usage_loss += splice_usage_loss.item()
            
            # Splice junction loss
            if 'splice_sites_junctions' in org_preds:
                splice_juncs = org_preds['splice_sites_junctions']
                splice_juncs_loss = loss_fns['splice_sites_junctions'](splice_juncs, splice_juncs)
                losses.append(splice_juncs_loss * 0)
                total_splice_juncs_loss += splice_juncs_loss.item()

        if len(losses) > 0:
            loss = torch.stack(losses).sum()
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_splice_logits_loss = total_splice_logits_loss / max(num_batches, 1)  
    avg_splice_usage_loss = total_splice_usage_loss / max(num_batches, 1)    
    avg_splice_juncs_loss = total_splice_juncs_loss / max(num_batches, 1)
    
    return avg_loss, avg_splice_logits_loss, avg_splice_usage_loss, avg_splice_juncs_loss

def main():
    # torch.autograd.set_detect_anomaly(True)  # Disable for performance
    start_time = time.time()

    #
    # Config
    #

    args = parse_args()
    config = load_config(args.config_file)
    
    data_dir = config.get('data_dir', '/home/elek/sds/sd17d003/Anamaria/splicevo/data_new/splits_adult_10kb/mouse_human/train/')
    output_dir = config.get('output_dir', '/home/elek/sds/sd17d003/Anamaria/alphagenome_pytorch/outputs/adult_10kb_mouse_human/')
    model_name = config.get('model_name', 'alphagenome_splicing')
    save_path = os.path.join(output_dir, f'{model_name}.pt')

    # Resources
    num_threads = config.get('num_threads', 1)
    num_interop_threads = config.get('num_interop_threads', 1)
    num_workers = config.get('num_workers', 12)
    gpu_mem_fraction = config.get('gpu_mem_fraction', 0.8)
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_interop_threads)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction, device=0)

    cache_dir = './outputs/checkpoints/pretrained/'
    seed = config.get('seed', 1950)
    set_seed(seed)

    # Load species mapping from metadata.json
    metadata_path = os.path.join(data_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Config params
    seq_len = config.get('seq_len', 10240)
    batch_size = config.get('batch_size', 32)
    epochs = config.get('epochs', 20)
    lr = config.get('lr', 1e-5)
    validation_fraction = config.get('validation_fraction', 0.2)
    max_donor_sites = config.get('max_donor_sites', 20)
    max_acceptor_sites = config.get('max_acceptor_sites', 20)

    # Load pretrained weights
    load_pretrained = config.get('load_pretrained', False)
    pretrained_model_version = config.get('pretrained_model_version', 'all_folds')
    freeze_backbone = config.get('freeze_backbone', False)
    heads_to_train = config.get('heads_to_train', ['splice_sites_classification', 'splice_sites_usage'])  # Specify which heads to train
    if heads_to_train == 'all':
        heads_to_train = None  # Train all heads

    # Load pretrained model configuration
    default_cfg = AlphaGenomeConfig()
    model_cfg = config.get('model', {})
    dims = tuple(model_cfg.get('dims', default_cfg.dims))
    basepairs = model_cfg.get('basepairs', default_cfg.basepairs)
    dna_embed_width = model_cfg.get('dna_embed_width', default_cfg.dna_embed_width)
    num_organisms = model_cfg.get('num_organisms', default_cfg.num_organisms)
    transformer_kwargs = model_cfg.get('transformer_kwargs', default_cfg.transformer_kwargs)

    # Heads configurations
    heads_cfg = model_cfg.get('heads_cfg', {
        'human': {
            'num_tracks_1bp': 0,
            'num_tracks_128bp': 0,
            'num_tracks_contacts': 0,
            'num_splicing_contexts': 16
        },
        'mouse': {
            'num_tracks_1bp': 0,
            'num_tracks_128bp': 0,
            'num_tracks_contacts': 0,
            'num_splicing_contexts': 34
        }
    })
    # Print config for verification
    print("Configuration:")
    print(f"  Seed: {seed}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of workers: {num_workers}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Validation fraction: {validation_fraction}")
    print(f"  Data directory: {data_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Model name: {model_name}")
    print(f"  Load pretrained: {load_pretrained}")
    if load_pretrained:
        print(f"  Pretrained model version: {pretrained_model_version}")
        print(f"  Freeze backbone: {freeze_backbone}") 
    print("  Model architecture:")
    print(f"    Dims: {dims}")
    print(f"    Basepairs: {basepairs}")
    print(f"    DNA embed width: {dna_embed_width}")
    print(f"    Num organisms: {num_organisms}")
    print(f"    Transformer kwargs: {transformer_kwargs}")
    print(f"  Heads configuration:")
    for organism, head_cfg in heads_cfg.items():
        print(f"    {organism}:")
        for key, value in head_cfg.items():
            print(f"      {key}: {value}")

   
    # All conditions to predict
    conds = metadata['usage_conditions']
    
    # Alpha Genome conditions
    alphagenome_conds = {}
    alphagenome_conds['human'] = {
        'Brain_1': (219, 586),
        'Brain_5': (219, 586),
        'Brain_10': (219, 586),
        'Brain_11': (219, 586),
        'Brain_12': (219, 586),
        'Brain_13': (219, 586),
        'Brain_14': (219, 586),
        'Brain_15': (219, 586),
        'Cerebellum_1': (277, 278, 644, 645),
        'Cerebellum_5': (277, 278, 644, 645),
        'Cerebellum_10': (276, 643),
        'Cerebellum_11': (276, 643),
        'Cerebellum_12': (276, 643),
        'Cerebellum_13': (276, 643),
        'Cerebellum_14': (276, 643),
        'Cerebellum_15': (276, 643),
        'Heart_1': (217, 218, 287, 288, 289, 290, 584, 585, 654, 655, 656, 657),
        'Heart_5': (217, 218, 287, 288, 289, 290, 584, 585, 654, 655, 656, 657),
        'Heart_10': (217, 218, 287, 288, 289, 290, 584, 585, 654, 655, 656, 657),
        'Heart_11': (217, 218, 287, 288, 289, 290, 584, 585, 654, 655, 656, 657),
        'Heart_12': (217, 218, 287, 288, 289, 290, 584, 585, 654, 655, 656, 657),
        'Heart_13': (217, 218, 287, 288, 289, 290, 584, 585, 654, 655, 656, 657),
        'Heart_14': (217, 218, 287, 288, 289, 290, 584, 585, 654, 655, 656, 657),
        'Heart_15': (217, 218, 287, 288, 289, 290, 584, 585, 654, 655, 656, 657),
        'Kidney_10': (244, 250, 299, 300, 611, 617, 666, 667),
        'Kidney_11': (244, 250, 299, 300, 611, 617, 666, 667),
        'Kidney_12': (244, 250, 299, 300, 611, 617, 666, 667),
        'Kidney_13': (244, 250, 299, 300, 611, 617, 666, 667),
        'Kidney_14': (244, 250, 299, 300, 611, 617, 666, 667),
        'Kidney_15': (244, 250, 299, 300, 611, 617, 666, 667),
        'Liver_10': (230, 231, 232, 597, 598, 599),
        'Liver_11': (230, 231, 232, 597, 598, 599),
        'Liver_12': (230, 231, 232, 597, 598, 599),
        'Liver_13': (230, 231, 232, 597, 598, 599),
        'Liver_14': (230, 231, 232, 597, 598, 599),
        'Liver_15': (230, 231, 232, 597, 598, 599),
        'Ovary_10': (220, 221, 222, 587, 588, 589),
        'Ovary_11': (220, 221, 222, 587, 588, 589),
        'Ovary_12': (220, 221, 222, 587, 588, 589),
        'Ovary_13': (220, 221, 222, 587, 588, 589),
        'Ovary_14': (220, 221, 222, 587, 588, 589),
        'Ovary_15': (220, 221, 222, 587, 588, 589),
        'Testis_10': (209, 210, 211, 576, 577, 578),
        'Testis_11': (209, 210, 211, 576, 577, 578),
        'Testis_12': (209, 210, 211, 576, 577, 578),
        'Testis_13': (209, 210, 211, 576, 577, 578),
        'Testis_14': (209, 210, 211, 576, 577, 578),
        'Testis_15': (209, 210, 211, 576, 577, 578),
    }
    alphagenome_conds['mouse'] = {
        'Brain_1': (49, 62, 64, 139, 152, 154),
        'Brain_5': (49, 62, 64, 139, 152, 154),
        'Brain_10': (63, 65, 153, 155),
        'Brain_11': (63, 65, 153, 155),
        'Brain_12': (63, 65, 153, 155),
        'Brain_13': (63, 65, 153, 155),
        'Brain_14': (63, 65, 153, 155),
        'Brain_15': (63, 65, 153, 155),
        'Cerebellum_1': (68, 158),
        'Cerebellum_5': (69, 159),
        'Cerebellum_10': (69, 159),
        'Cerebellum_11': (69, 159),
        'Cerebellum_12': (69, 159),
        'Cerebellum_13': (69, 159),
        'Cerebellum_14': (69, 159),
        'Cerebellum_15': (69, 159),
        'Heart_1': (47, 48, 137, 138),
        'Heart_5': (47, 48, 137, 138),
        'Heart_10': (47, 48, 137, 138),
        'Heart_11': (47, 48, 137, 138),
        'Heart_12': (47, 48, 137, 138),
        'Heart_13': (47, 48, 137, 138),
        'Heart_14': (47, 48, 137, 138),
        'Heart_15': (47, 48, 137, 138),
        'Kidney_10': (77, 167),
        'Kidney_11': (77, 167),
        'Kidney_12': (77, 167),
        'Kidney_13': (77, 167),
        'Kidney_14': (77, 167),
        'Kidney_15': (77, 167),
        'Liver_10': (74, 75, 164, 165),
        'Liver_11': (74, 75, 164, 165),
        'Liver_12': (74, 75, 164, 165),
        'Liver_13': (74, 75, 164, 165),
        'Liver_14': (74, 75, 164, 165),
        'Liver_15': (74, 75, 164, 165),
        'Ovary_10': (50, 140),
        'Ovary_11': (50, 140),
        'Ovary_12': (50, 140),
        'Ovary_13': (50, 140),
        'Ovary_14': (50, 140),
        'Ovary_15': (50, 140),
        'Testis_10': (45, 135),
        'Testis_11': (45, 135),
        'Testis_12': (45, 135),
        'Testis_13': (45, 135),
        'Testis_14': (45, 135),
        'Testis_15': (45, 135),
    }

    # Maping
    species_mapping = metadata.get('species_mapping', {
        'human': 0,
        'mouse': 1,
        'rat': 2
    })

    #
    # Architecture
    #

    print("\nInitializing model...")
    init_time = time.time()

    # Initialize the model
    model_pretrained = AlphaGenome(dims, basepairs, dna_embed_width, num_organisms, transformer_kwargs)
    model_pretrained.add_reference_heads("human")
    model_pretrained.add_reference_heads('mouse')

    # Check if resuming from checkpoint
    resume_from_checkpoint = args.resume is not None
    start_epoch = 0

    if resume_from_checkpoint:
        # When resuming, load the complete model state from checkpoint
        print("Resuming training from checkpoint...")
        
        # Move model to device before loading checkpoint
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_pretrained = model_pretrained.to(device)
        
        # Create temporary optimizer for loading checkpoint state
        temp_optimizer = torch.optim.AdamW(model_pretrained.parameters(), lr=lr, weight_decay=0.05)
        
        # Load checkpoint
        start_epoch = load_checkpoint(model_pretrained, temp_optimizer, args.resume, device=device)
        
        # Store checkpoint optimizer state to apply later
        checkpoint_optimizer_state = temp_optimizer.state_dict()
        del temp_optimizer
        
    elif load_pretrained:
        # Load pretrained weights
        print(f"Loading pretrained weights from {pretrained_model_version}...")
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True, parents=True)
        cached_model_path = cache_dir / f'pretrained_model_{pretrained_model_version}.pt'
        if cached_model_path.exists():
            print(f"Loading pretrained model from cache: {cached_model_path}")
            checkpoint = torch.load(cached_model_path, map_location='cpu')
            model_pretrained.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("No cached model found. Downloading pretrained weights...")
            model_pretrained.load_from_official_jax_model(pretrained_model_version, strict=False)
            print("Saving to cache for future use...")
            torch.save({'model_state_dict': model_pretrained.state_dict()}, cached_model_path)
            print(f"Model cached to {cached_model_path}")
        
        # Save classification head weights BEFORE clearing heads
        weights_and_biases = dict()
        for organism in ['human', 'mouse']:
            weights_and_biases[organism] = dict()
            w = model_pretrained.heads[organism]['splice_sites_classification'].linear.weight.data.clone()
            b = model_pretrained.heads[organism]['splice_sites_classification'].linear.bias.data.clone()
            weights_and_biases[organism]['classification_head'] = dict()
            weights_and_biases[organism]['classification_head']['weight'] = w
            weights_and_biases[organism]['classification_head']['bias'] = b
            print(f"Output head weights: ")
            print(f"  {organism} splice classification head weight mean: {w.mean().item():.6f}, std: {w.std().item():.6f}")
            print(f"  {organism} splice classification head bias mean: {b.mean().item():.6f}, std: {b.std().item():.6f}")
        print("Pretrained weights loaded successfully")
        print(f"Model initialized in {time.time() - init_time:.2f} seconds.")
        
        # Remove existing heads and add new splicing heads
        model_pretrained.heads = torch.nn.ModuleDict()
        for organism, head_cfg in heads_cfg.items():
            model_pretrained.add_heads(organism=organism, **head_cfg)

        # Copy output head weights from pretrained model
        print("Copying pretrained weights after adding new heads...")
        for org_name in ['human', 'mouse']:
            model_pretrained.heads[org_name]['splice_sites_classification'].linear.weight.data.copy_(weights_and_biases[org_name]['classification_head']['weight'])
            model_pretrained.heads[org_name]['splice_sites_classification'].linear.bias.data.copy_(weights_and_biases[org_name]['classification_head']['bias'])

        # Move model to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_pretrained = model_pretrained.to(device)
        
    else:
        # No pretrained weights and not resuming - train from scratch
        print("\nTraining from scratch (no pretrained weights)")
        
        # Remove existing heads and add new splicing heads
        model_pretrained.heads = torch.nn.ModuleDict()
        for organism, head_cfg in heads_cfg.items():
            model_pretrained.add_heads(organism=organism, **head_cfg)
        
        # Move model to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_pretrained = model_pretrained.to(device)

    print(f"Using device: {device}")

    # Freeze model backbone (applies to all initialization paths if configured)
    if freeze_backbone:
        print("Freezing transformer backbone...")
        for param in model_pretrained.transformer_unet.parameters():
            param.requires_grad = False

        # Freezing embeddings
        print("Freezing embeddings parameters...")
        for param in model_pretrained.organism_embed.parameters():
            param.requires_grad = False
        for param in model_pretrained.outembed_128bp.parameters():
            param.requires_grad = False
        for param in model_pretrained.outembed_1bp.parameters():
            param.requires_grad = False
        for param in model_pretrained.outembed_pair.parameters():
            param.requires_grad = False
        
        # Keep heads trainable
        if heads_to_train is None:
            heads_to_train = []
            for organism, heads in model_pretrained.heads.items():
                for head_name in heads.keys():
                    heads_to_train.append(head_name)

        for organism, heads in model_pretrained.heads.items():
            for head_name, head in heads.items():
                if head_name in heads_to_train:
                    print(f"Keeping head {head_name} for {organism} trainable")
                    for param in head.parameters():
                        param.requires_grad = True
                else:
                    print(f"Freezing head {head_name} for {organism}")
                    for param in head.parameters():
                        param.requires_grad = False
        
        trainable_params = sum(p.numel() for p in model_pretrained.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")

    #
    # Train/validation split
    #

    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    print("\nPreparing data loaders...")
    data_time = time.time()
    
    print(f"Species mapping: {species_mapping}")

    print("Loading dataset...")
    # Load splice dataset
    train_dataset = SpliceDataset(
        data_dir=data_dir,
        target_length=seq_len,
        max_donor_sites=max_donor_sites,
        max_acceptor_sites=max_acceptor_sites,
        species_mapping=species_mapping
    )

    # Split into train and validation (stratified by species)
    print("Splitting dataset into train and validation sets...")
    dataset_size = len(train_dataset)
    val_size = int(dataset_size * validation_fraction)
    train_size = dataset_size - val_size
    
    # Get species labels for stratified split
    species_labels = []
    for idx in range(dataset_size):
        species_id = train_dataset.species.iloc[idx]
        # If it is integer, use directly, otherwise if string (species name), map to index
        if isinstance(species_id, (int, np.integer)):
            organism_idx = int(species_id)
        else:
            organism_idx = train_dataset.species_mapping.get(species_id, 0)
        species_labels.append(organism_idx)
    species_labels = np.array(species_labels)
    
    # Stratified split - ensure each species is represented proportionally
    train_indices = []
    val_indices = []
    
    for organism_idx in np.unique(species_labels):
        # Get all indices for this organism
        org_indices = np.where(species_labels == organism_idx)[0]
        n_org = len(org_indices)
        n_val_org = int(n_org * validation_fraction)
        
        # Shuffle within organism
        np.random.shuffle(org_indices)
        
        # Split
        val_indices.extend(org_indices[:n_val_org])
        train_indices.extend(org_indices[n_val_org:])
    
    # Shuffle the combined indices
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    
    # Create subset datasets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    
    # Print species distribution
    train_species = species_labels[train_indices]
    val_species = species_labels[val_indices]
    print(f"Dataset split: {len(train_indices)} train, {len(val_indices)} validation")
    for organism_idx in np.unique(species_labels):
        organism_name = [k for k, v in species_mapping.items() if v == organism_idx]
        organism_name = organism_name[0] if organism_name else f"organism_{organism_idx}"
        train_count = np.sum(train_species == organism_idx)
        val_count = np.sum(val_species == organism_idx)
        print(f"  {organism_name}: {train_count} train, {val_count} val")
    
    train_sampler = SpeciesGroupedSampler(train_subset, batch_size=batch_size, shuffle=True)
    val_sampler = SpeciesGroupedSampler(val_subset, batch_size=batch_size, shuffle=False)
    
    train_loader = DataLoader(
        train_subset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Data loaders prepared in {time.time() - data_time:.2f} seconds.")

    #
    # Training loop
    #

    train_time = time.time()
    print("\nSetting up training...")

    # Initialize optimizer after freezing (only tracks trainable parameters)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_pretrained.parameters()),
        lr=lr,
        weight_decay=0.01
    )

    # Calculate total steps
    total_steps = epochs * len(train_loader)
    warmup_steps = len(train_loader)  # 1 epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # If resuming, load the optimizer state from checkpoint
    if resume_from_checkpoint:
        optimizer.load_state_dict(checkpoint_optimizer_state)
        print("Loaded optimizer state from checkpoint")

    # Losses for splicing tasks only
    loss_fns = {
        'splice_sites_classification': nn.CrossEntropyLoss(),
        'splice_sites_usage': nn.BCEWithLogitsLoss(),
        'splice_sites_junctions': JunctionsLoss()
    }

    # Set up GradScaler for mixed precision
    scaler = GradScaler('cuda') if torch.cuda.is_available() else None
    use_amp = False #torch.cuda.is_available()

    if resume_from_checkpoint:
        print(f"\nResuming training from epoch {start_epoch + 1} for {epochs - start_epoch} more epochs...")
    else:
        print(f"\nStarting training for {epochs} epochs...")

    print(f"Train size: {train_size}, Validation size: {val_size}")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}")
    best_val_loss = float('inf')
    patience = 5
    epochs_without_improvement = 0

    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()

        # Run training
        avg_train_loss, splice_logits_loss, splice_usage_loss, splice_juncs_loss = train_one_epoch(
            model=model_pretrained,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fns=loss_fns,
            species_mapping=species_mapping,
            device=device,
            heads_to_train=heads_to_train,
            scheduler=scheduler,
            scaler=scaler,
            use_amp=use_amp,
            freeze_backbone=freeze_backbone
        )
        msg = f"Train Loss: {avg_train_loss:.4f} "
        msg += f"(Splice: {splice_logits_loss:.4f}, "
        msg += f"Usage: {splice_usage_loss:.4f}) "

        # Run validation
        avg_val_loss, val_logits_loss, val_usage_loss, val_juncs_loss = validate_one_epoch(
            model=model_pretrained,
            val_loader=val_loader,
            loss_fns=loss_fns,
            species_mapping=species_mapping,
            device=device,
            use_amp=use_amp
        )
        msg += f"Val Loss: {avg_val_loss:.4f} "
        msg += f"(Splice: {val_logits_loss:.4f}, "
        msg += f"Usage: {val_usage_loss:.4f})"

        # Timing
        epoch_time = time.time() - epoch_start_time
        msg = f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) - " + msg
        print(msg)

        # Save checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            save_model(model_pretrained, optimizer, epoch + 1, save_path)
            print(f"  - Saved best model {best_val_loss:.4f}")
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs (no improvement for {patience} epochs)")
                break

    print(f"\nTraining complete! Best validation loss ({best_val_loss:.4f}) achieved.")
    print(f"Total training time: {(time.time() - train_time) / 60:.2f} minutes.")
    print(f"Total script runtime: {(time.time() - start_time) / 60:.2f} minutes.")

if __name__ == "__main__":
    main()
