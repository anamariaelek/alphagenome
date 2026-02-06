from __future__ import annotations

import argparse
import os
import yaml
import json
import random
import time
import numpy as np
import warnings

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

from alphagenome_pytorch import AlphaGenome, AlphaGenomeConfig, TargetScaler, MultinomialLoss, JunctionsLoss
from alphagenome_pytorch.splice_dataset import SpliceDataset
from alphagenome_pytorch.samplers import SpeciesGroupedSampler

from accelerate import Accelerator

def exists(v):
    return v is not None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
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
    torch.save({
        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, path)

def get_organism_name(organism_idx, species_mapping):
    """Map organism index to organism name used in model heads"""
    # Invert species_mapping to go from index to name
    for name, idx in species_mapping.items():
        if idx == organism_idx:
            return name
    return f'organism_{organism_idx}'


def train_one_epoch(model, dataloader, optimizer, loss_fns, device, species_mapping, accelerator: Accelerator | None = None):
    model.train()
    total_loss = 0.0
    total_splice_logits_loss = 0.0
    total_splice_usage_loss = 0.0
    total_splice_juncs_loss = 0.0
    num_batches = 0

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
            if 'splice_logits' in org_preds:
                splice_logits = org_preds['splice_logits']  # (batch, seq_len, 5)
                # Reshape for cross entropy
                splice_logits_flat = splice_logits.reshape(-1, splice_logits.shape[-1])
                splice_labels_flat = splice_labels.reshape(-1)
                splice_logits_loss = loss_fns['splice_logits'](splice_logits_flat, splice_labels_flat)
                losses.append(splice_logits_loss)
                total_splice_logits_loss += splice_logits_loss.item()
            
            # Splice usage loss (per-context usage prediction)
            if 'splice_usage' in org_preds:
                splice_usage = org_preds['splice_usage']  # (batch, seq_len, num_contexts_for_organism)
            
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
                    
                    # Skip if predictions contain NaN
                    if not (torch.isnan(splice_usage_at_sites).any() or torch.isinf(splice_usage_at_sites).any()):
                        # Compute MSE loss
                        splice_usage_loss = loss_fns['splice_usage'](splice_usage_at_sites, org_sse_target_at_sites)
                        
                        if not torch.isnan(splice_usage_loss):
                            losses.append(splice_usage_loss)
                            total_splice_usage_loss += splice_usage_loss.item()

            
            # Splice junction loss (donor-acceptor pairings)
            if 'splice_juncs' in org_preds:
                splice_juncs = org_preds['splice_juncs']  # (batch, num_donors, num_acceptors, num_contexts)
                # Create target junction matrix (placeholder - adjust based on your data)
                # For now, use a simple loss
                splice_juncs_loss = loss_fns['splice_juncs'](splice_juncs, splice_juncs)  # Placeholder
                losses.append(splice_juncs_loss * 0)  # Weight down for now
                total_splice_juncs_loss += splice_juncs_loss.item()

        if len(losses) == 0:
            continue
            
        loss = torch.stack(losses).sum()

        # Backpropagation
        if exists(accelerator):
            accelerator.backward(loss)
        else:
            loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        num_batches += 1    

    avg_loss = total_loss / max(num_batches, 1)    
    avg_splice_juncs_loss = total_splice_juncs_loss / max(num_batches, 1)
    avg_splice_logits_loss = total_splice_logits_loss / max(num_batches, 1)    
    avg_splice_usage_loss = total_splice_usage_loss / max(num_batches, 1)
    
    return avg_loss, avg_splice_logits_loss, avg_splice_usage_loss, avg_splice_juncs_loss

@torch.no_grad()
def validate_one_epoch(model, val_loader, loss_fns, species_mapping, accelerator=None, device='cuda'):
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
            if 'splice_logits' in org_preds:
                splice_logits = org_preds['splice_logits']
                splice_logits_flat = splice_logits.reshape(-1, splice_logits.shape[-1])
                splice_labels_flat = splice_labels.reshape(-1)
                splice_logits_loss = loss_fns['splice_logits'](splice_logits_flat, splice_labels_flat)
                losses.append(splice_logits_loss)
                total_splice_logits_loss += splice_logits_loss.item()
            
            # Splice usage loss
            if 'splice_usage' in org_preds:
                splice_usage = org_preds['splice_usage']  # (batch, seq_len, num_contexts_for_organism)
            
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
                    
                    # Skip if predictions contain NaN
                    if not (torch.isnan(splice_usage_at_sites).any() or torch.isinf(splice_usage_at_sites).any()):
                        # Compute MSE loss
                        splice_usage_loss = loss_fns['splice_usage'](splice_usage_at_sites, org_sse_target_at_sites)
                        
                        if not torch.isnan(splice_usage_loss):
                            losses.append(splice_usage_loss)
                            total_splice_usage_loss += splice_usage_loss.item()
            
            # Splice junction loss
            if 'splice_juncs' in org_preds:
                splice_juncs = org_preds['splice_juncs']
                splice_juncs_loss = loss_fns['splice_juncs'](splice_juncs, splice_juncs)
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
    
    seed = config.get('seed', 1950)
    gpu_mem_fraction = config.get('gpu_mem_fraction', 0.8)
    device = config.get('device', 'cuda')
    torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction, device)

    seq_len = config.get('seq_len', 4096)  # Must be power of 2
    batch_size = config.get('batch_size', 4)
    num_workers = config.get('num_workers', 4)
    epochs = config.get('epochs', 3)
    lr = config.get('lr', 1e-4)
    validation_fraction = config.get('validation_fraction', 0.1)

    max_donor_sites = config.get('max_donor_sites', 20)
    max_acceptor_sites = config.get('max_acceptor_sites', 20)

    # Data paths
    data_dir = config.get('data_dir', '/home/elek/sds/sd17d003/Anamaria/splicevo/data/splits_small_5kb/mouse_human/train/')
    output_dir = config.get('output_dir', './outputs_splicing')
    model_name = config.get('model_name', 'best_model')
    save_path = os.path.join(output_dir, f'{model_name}.pt')

    # Load pretrained weights
    load_pretrained = config.get('load_pretrained', False)
    pretrained_model_version = config.get('pretrained_model_version', 'all_folds')
    freeze_backbone = config.get('freeze_backbone', False)

    # Default config
    default_cfg = AlphaGenomeConfig()
    model_cfg = config.get('model', {})
    dims = tuple(model_cfg.get('dims', default_cfg.dims))
    basepairs = model_cfg.get('basepairs', default_cfg.basepairs)
    dna_embed_width = model_cfg.get('dna_embed_width', default_cfg.dna_embed_width)
    num_organisms = model_cfg.get('num_organisms', default_cfg.num_organisms)
    transformer_kwargs = model_cfg.get('transformer_kwargs', default_cfg.transformer_kwargs)
    
    # Hardcoded heads configuration for splicing-only tasks
    heads_cfg = {
        'human': {
            'num_tracks_1bp': 0,
            'num_tracks_128bp': 0,
            'num_tracks_contacts': 0,
            'num_splicing_contexts': 7 ### Hardcoded for now - adjust based on actual data (e.g. number of cell types or conditions with splicing usage data)
        },
        'mouse': {
            'num_tracks_1bp': 0,
            'num_tracks_128bp': 0,
            'num_tracks_contacts': 0,
            'num_splicing_contexts': 8 ### Hardcoded for now - adjust based on actual data (e.g. number of cell types or conditions with splicing usage data)
        }
    }

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


    #
    # Architecture
    #

    print("\nInitializing model...")
    init_time = time.time()

    model = AlphaGenome(dims, basepairs, dna_embed_width, num_organisms, transformer_kwargs)
    
    # Add splicing heads only (no tracks or contact maps)
    for organism, head_cfg in heads_cfg.items():
        model.add_heads(organism=organism, **head_cfg)
    
    print("\nTotal model parameters:", model.total_parameters)
    
    # Load pretrained weights if specified
    if load_pretrained:
        print(f"Loading pretrained weights from {pretrained_model_version}...")
        model.load_from_official_jax_model(pretrained_model_version, strict=False)
        print("Pretrained weights loaded successfully")
        
        # Optionally freeze backbone
        if freeze_backbone:
            print("Freezing transformer backbone...")
            for param in model.transformer_unet.parameters():
                param.requires_grad = False
            
            # Keep heads trainable
            for organism, heads in model.heads.items():
                for head in heads.values():
                    for param in head.parameters():
                        param.requires_grad = True
            
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable parameters (heads only): {trainable_params:,}")

    # Don't need this because it is handled by Accelerator
    #if torch.cuda.device_count() > 1:
    #    print(f"Using {torch.cuda.device_count()} GPUs.")
    #    model = nn.DataParallel(model)

    print(f"Model initialized in {time.time() - init_time:.2f} seconds.")

    #
    # Train/validation split
    #

    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    print("\nPreparing data loaders...")
    data_time = time.time()

    # Load species mapping from metadata.json
    metadata_path = os.path.join(data_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    species_mapping = metadata.get('species_mapping', {
        'human': 0,
        'mouse': 1,
        'rat': 2
    })
    
    print(f"Species mapping: {species_mapping}")

    # Load splice dataset
    train_dataset = SpliceDataset(
        data_dir=data_dir,
        target_length=seq_len,
        max_donor_sites=max_donor_sites,
        max_acceptor_sites=max_acceptor_sites,
        species_mapping=species_mapping
    )

    # Split into train and validation (stratified by species)
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

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01
    )

    # prepare accelerator for distributed

    accelerate_kwargs = {}

    accelerator = Accelerator(**accelerate_kwargs)

    model, train_loader, val_loader, optimizer = accelerator.prepare(model, train_loader, val_loader, optimizer)

    device = accelerator.device

    # Losses for splicing tasks only

    loss_fns = {
        'splice_logits' : nn.CrossEntropyLoss(),
        'splice_usage' : nn.CrossEntropyLoss(),
        'splice_juncs': JunctionsLoss()
    }

    print(f"\nStarting training for {epochs} epochs...")
    print(f"Train size: {train_size}, Validation size: {val_size}")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}")
    best_val_loss = float('inf')
    patience = 5
    epochs_without_improvement = 0
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Run training
        avg_train_loss, splice_logits_loss, splice_usage_loss, splice_juncs_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fns=loss_fns,
            species_mapping=species_mapping,
            accelerator=accelerator,
            device=device
        )
        msg = f"[Epoch {epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} "
        msg += f"(Logits: {splice_logits_loss:.4f} | "
        msg += f"Usage: {splice_usage_loss:.4f} )"
    
        # Run validation
        avg_val_loss, val_logits_loss, val_usage_loss, val_juncs_loss = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            loss_fns=loss_fns,
            species_mapping=species_mapping,
            accelerator=accelerator,
            device=device
        )
        msg += f"Val Loss: {avg_val_loss:.4f} "
        msg += f"(Splice: {val_logits_loss:.4f} | "
        msg += f"Usage: {val_usage_loss:.4f})"
 
        # Timing
        epoch_time = time.time() - epoch_start_time
        msg += f" Time: {epoch_time:.1f}s"
        print(msg)
        
        # Save checkpoint only if validation loss improved (after epoch 1)
        if accelerator.is_main_process and epoch >= 0:  # Start saving from first epoch
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                save_model(model, optimizer, epoch + 1, save_path)
                print(f"  - Saved best model: {save_path}")
            else:
                epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs (no improvement for {patience} epochs)")
                    break

        accelerator.wait_for_everyone()

    print(f"\nTraining complete! Best validation loss ({best_val_loss:.4f}) achieved at epoch {epoch + 1 - epochs_without_improvement}.")
    print(f"Total training time: {(time.time() - train_time) / 60:.2f} minutes.")
    print(f"Total script runtime: {(time.time() - start_time) / 60:.2f} minutes.")
    
if __name__ == "__main__":
    main()
