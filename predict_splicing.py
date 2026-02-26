import os
import yaml
import json
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from pathlib import Path

from alphagenome_pytorch import AlphaGenome, AlphaGenomeConfig, SpliceDataset
from alphagenome_pytorch.samplers import SpeciesGroupedSampler

from torch.utils.data import DataLoader

from sklearn.metrics import precision_recall_curve, auc

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_pretrained_model(cache_dir='./outputs/checkpoints/pretrained', num_organisms=2):
    """
    Load the pretrained AlphaGenome model with caching.
    
    First time: Downloads and converts JAX model (~5 minutes) and caches it
    Subsequent times: Loads from cache (under 1 minute)
    
    Args:
        cache_dir: Directory to store cached model
        num_organisms: Number of organisms (default: 2 for human and mouse)
    
    Returns:
        model: AlphaGenome model in eval mode
    """
    # Path to cache the converted model
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)
    cached_model_path = cache_dir / 'pretrained_model_all_folds.pt'
    
    model = AlphaGenome(num_organisms=num_organisms)
    model.add_reference_heads("human")
    model.add_reference_heads('mouse')
    
    # Load from cache if available, otherwise download and cache
    if cached_model_path.exists():
        print(f"Loading pretrained model from cache: {cached_model_path}")
        checkpoint = torch.load(cached_model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded from cache (fast!)")
    else:
        print("Downloading and converting JAX model (this will take ~5 minutes)...")
        model.load_from_official_jax_model("all_folds")
        print("Saving to cache for future use...")
        torch.save({'model_state_dict': model.state_dict()}, cached_model_path)
        print(f"Model cached to {cached_model_path}")
    
    model.eval()
    return model

def plot_sse_density(all_data_df: pd.DataFrame, group_by: str, output_fn: str = None, log_fn=print):
        """Plot density of predicted vs true SSE values.
        Args:
            all_data_df: DataFrame with SSE data, it should contain columns 'true_usage' and 'pred_usage' along with grouping columns
            group_by: Column name or list of column names to group by
            output_dir: Directory to save output
            log_fn: Logging function
        """
        
        # Get valid positions where we have both true and predicted SSE
        valid_data = all_data_df.dropna(subset=['true_usage', 'pred_usage'])

        # Get unique combinations of grouping variables
        groups = valid_data.groupby(group_by).size().reset_index(name='count')
        num_groups = len(groups)
        
        # Determine subplot layout
        num_cols = min(3, num_groups)  # Max 3 columns
        num_rows = (num_groups + num_cols - 1) // num_cols
        
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4.8, num_rows * 4), squeeze=False)
        
        for i, (group_idx, group_row) in enumerate(groups.iterrows()):
            # Build filter for this group
            mask = pd.Series([True] * len(valid_data), index=valid_data.index)
            for col in group_by:
                mask &= (valid_data[col] == group_row[col])
            
            condition_data = valid_data[mask]
            
            # Build title from group values
            if len(group_by) == 1:
                title = f"{group_row[group_by[0]]}"
            else:
                title = ", ".join([f"{col}: {group_row[col]}" for col in group_by])
            
            row = i // num_cols
            col = i % num_cols
            ax = axs[row, col]
                    
            # 2D hexbin plot
            hb = ax.hexbin(
                condition_data['true_usage'],
                condition_data['pred_usage'],
                gridsize=30,
                cmap='rocket_r',
                mincnt=1
            )
            plt.colorbar(
                hb,
                ax=ax,
                label='Count',
                pad=0.2
            )
            
            # Add pearson correlation value and number of points to plot
            num_points = len(condition_data)
            if num_points >= 2:
                corr = np.corrcoef(condition_data['true_usage'], condition_data['pred_usage'])[0, 1]
                ax.text(0.05, 0.95, f"r = {corr:.2f}\nn = {num_points}", transform=ax.transAxes, fontsize=10, verticalalignment='top')    
            else:
                ax.text(0.05, 0.95, f"n = {num_points}", transform=ax.transAxes, fontsize=10, verticalalignment='top')
                
            # Top histogram (True SSE)
            ax_histx = ax.inset_axes([0, 1.05, 1, 0.2], sharex=ax)
            ax_histx.hist(condition_data['true_usage'], bins=30, color='gray', alpha=0.7)
            ax_histx.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
            
            # Right histogram (Pred SSE)
            ax_histy = ax.inset_axes([1.05, 0, 0.2, 1], sharey=ax)
            ax_histy.hist(condition_data['pred_usage'], bins=30, orientation='horizontal', color='gray', alpha=0.7)
            ax_histy.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
            
            ax.set_xlabel('True Usage', fontsize=10)
            ax.set_ylabel('Predicted Usage', fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(title, fontsize=12)
            ax.grid()
        
        # Hide unused subplots
        for i in range(num_groups, num_rows * num_cols):
            row = i // num_cols
            col = i % num_cols
            axs[row, col].set_visible(False)
        
        plt.tight_layout()
        if output_fn is not None:
            plt.savefig(output_fn, dpi=150)
            plt.close()
        else:
            plt.show()


from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--config', type=str, help='Path to config file for finetuned model (YAML format)')
parser.add_argument('--test-data', type=str, help='Optional path to test dataset directory (if not specified, test equivalent of train data_dir from config will be used)')
parser.add_argument('--output-dir', type=str, help='Directory to save predictions and plots (if not specified, defaults to new directory "predictions" inside output_dir from config')
parser.add_argument('--skip-prediction', action='store_true', help='Only run final aggregation/plotting, skip prediction loop')
parser.add_argument('--skip-per-batch-plots', action='store_true', help='Do not save per-batch plots')
parser.add_argument('--pretrained', action='store_true', help='Use pretrained model instead of finetuned')
parser.add_argument('--verbose', action='store_true', help='Print detailed per-batch statistics')
parser.add_argument('--batch_size', type=int, default=None, help='Override batch size from config')
parser.add_argument('--max-batches', type=int, default=None, help='Maximum number of batches to process (default: all)')
args = parser.parse_args()

# Resources and parameters
num_threads = 1
num_interop_threads = 1
num_workers = 12
gpu_mem_fraction = 0.5
use_pretrained_model = args.pretrained
use_finetuned_model = not args.pretrained
max_batches = args.max_batches

species_mapping = {
    'human': 0,
    'mouse': 1
}

seed = 1950
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
start_time = time.time()


# Load config
config_file = args.config
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Resources
num_threads = config.get('num_threads', num_threads)
num_interop_threads = config.get('num_interop_threads', num_interop_threads)
num_workers = config.get('num_workers', num_workers)
gpu_mem_fraction = config.get('gpu_mem_fraction', gpu_mem_fraction)
seed = config.get('seed', seed)

# Input data
test_data_dir = args.test_data

# Input and sequence parameters (matching training)
train_data_dir = config.get('data_dir')
test_data_dir = train_data_dir.replace('/train', '/test') if test_data_dir is None else test_data_dir
target_length = config.get('seq_len', 10240)
max_donor_sites = config.get('max_donor_sites', 20)
max_acceptor_sites = config.get('max_acceptor_sites', 20)
batch_size = args.batch_size if args.batch_size is not None else config.get('batch_size', 16)

# Output
model_dir = config.get('output_dir', './outputs/checkpoints/finetuned')
output_dir = args.output_dir
if output_dir is None:
    output_dir = os.path.join(model_dir, 'predictions')
os.makedirs(output_dir, exist_ok=True)
per_batch_dir = os.path.join(output_dir, 'per_batch')
os.makedirs(per_batch_dir, exist_ok=True)

if not args.skip_prediction:

    if use_pretrained_model:
        model = load_pretrained_model()
        output_dir = './outputs/predictions/pretrained_10kb'

    elif use_finetuned_model:
        # Get model architecture parameters (matching training)
        default_cfg = AlphaGenomeConfig()
        model_cfg = config.get('model', {})
        dims = tuple(model_cfg.get('dims', default_cfg.dims))
        basepairs = model_cfg.get('basepairs', default_cfg.basepairs)
        dna_embed_width = model_cfg.get('dna_embed_width', default_cfg.dna_embed_width)
        num_organisms = model_cfg.get('num_organisms', default_cfg.num_organisms)
        transformer_kwargs = model_cfg.get('transformer_kwargs', default_cfg.transformer_kwargs)
        heads_cfg = model_cfg.get('heads_cfg')

        # Create model with same architecture as training
        model = AlphaGenome(dims, basepairs, dna_embed_width, num_organisms, transformer_kwargs)

        # Add heads with same config as training
        for organism, head_cfg in heads_cfg.items():
            model.add_heads(organism=organism, **head_cfg)

        # Load the checkpoint
        model_name = config.get('model_name', 'alphagenome_finetuned')
        checkpoint_path = os.path.join(model_dir, f'{model_name}.pt')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])

        # Set to eval mode
        model.eval()

        print(f"Model loaded from {checkpoint_path}")
        print(f"Trained for {checkpoint['epoch']} epochs")

    # Freeze BatchRMSNorm running variance updates during inference
    from alphagenome_pytorch.alphagenome import set_update_running_var
    set_update_running_var(model, False)
    print("Froze BatchRMSNorm running variance updates")

    # Move model to GPU if available
    model.to(device)
    print(f"Model moved to {device}")

    # Set resources
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_interop_threads)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(gpu_mem_fraction, device=0)

    set_seed(seed)

    # Load test data
    print(f"Loading test dataset from {test_data_dir}")
    test_dataset = SpliceDataset(
        data_dir=test_data_dir,
        target_length=target_length,
        max_donor_sites=max_donor_sites,
        max_acceptor_sites=max_acceptor_sites,
        species_mapping=species_mapping
    )

    meta_csv_fn = os.path.join(test_data_dir, 'metadata.csv')
    metadata_csv = pd.read_csv(meta_csv_fn)

    meta_fn = os.path.join(test_data_dir, 'metadata.json')
    with open(meta_fn) as f:
        metadata = json.load(f)

    species_conds = metadata['species_condition_mapping']

    for org, cond in species_conds.items():
        print(f"Condition for {org}:")
        print([metadata['usage_conditions'][cond_idx] for cond_idx in cond])


    # Create sampler
    print(f"Creating sampler for test dataset with batch size {batch_size}")
    test_sampler = SpeciesGroupedSampler(test_dataset, batch_size=batch_size, shuffle=False)

    # Create dataloader for entire test set
    print("Creating dataloader for test dataset")
    test_loader_full = DataLoader(
        test_dataset, 
        batch_sampler=test_sampler, 
        num_workers=4
    )

    # Pre-generate batch-to-sample mapping from the sampler
    print("Generating batch-to-sample mapping...")
    batch_to_samples = {}  # Maps batch_idx -> list of dataset indices
    for batch_idx, sample_indices in enumerate(test_sampler):
        batch_to_samples[batch_idx] = sample_indices

    # Create batch metadata DataFrame
    batch_metadata_rows = []
    for batch_idx, sample_indices in batch_to_samples.items():
        for sample_idx in sample_indices:
            batch_metadata_rows.append({
                'batch_idx': batch_idx,
                'sample_idx': sample_idx,
                **metadata_csv.iloc[sample_idx].to_dict()
            })

    batch_metadata_df = pd.DataFrame(batch_metadata_rows)
    batch_metadata_df.to_csv(os.path.join(output_dir, 'batch_sample_mapping.csv'), index=False)
    print(f"Saved batch-to-sample mapping to {output_dir}/batch_sample_mapping.csv")
    print(f"Total samples: {len(batch_metadata_df)}, Total batches: {len(batch_to_samples)}")

all_splice_predictions_file = os.path.join(output_dir, '.tmp_splice_predictions.csv.gz')
all_usage_predictions_file = os.path.join(output_dir, '.tmp_usage_predictions.csv.gz')
all_splice_predictions = []
all_usage_predictions = []
splice_flush_interval = 20  # Flush to disk every N batches
usage_flush_interval = 20

if not args.skip_prediction:
    if max_batches is None:
        max_batches = len(test_loader_full)

    # Collect batch stats in a dict for later analysis
    batch_stats = {
        'batch_idx': [],
        'num_samples': [],
        'num_human_samples': [],
        'num_mouse_samples': [],
        'inference_time': []
    }

    for batch_idx, batch in enumerate(test_loader_full):
        
        # Track which samples are in this batch
        # Note: The batch sampler provides indices, but DataLoader might reorder
        # We need to get the actual sample indices from the batch
        # Since the dataset returns sample-specific data, we can infer indices
        # from the batch sampler's output
        

        if batch_idx == 0 and args.verbose:
            print("Batch keys:", batch.keys())
            print("DNA shape:", batch['dna'].shape)
            print("Organism index shape:", batch['organism_index'].shape)
            print("Splice labels shape:", batch['splice_labels'].shape)
            print("Splice usage target shape:", batch['splice_usage_target'].shape)

        if batch_idx >= max_batches:
            print(f"Reached max_batches={max_batches}")
            break

        batch_stats['batch_idx'].append(batch_idx)
        batch_stats['num_samples'].append(batch['dna'].shape[0])

        # Count samples per organism
        num_human = (batch['organism_index'] == species_mapping['human']).sum().item()
        num_mouse = (batch['organism_index'] == species_mapping['mouse']).sum().item()
        batch_stats['num_human_samples'].append(num_human)
        batch_stats['num_mouse_samples'].append(num_mouse)

        if args.verbose:
            print("-"*60)
            print(f"\nStarting prediction for batch {batch_idx+1}/{len(test_loader_full)} with {batch['dna'].shape[0]} samples\n")
            print(f" Human samples: {num_human}, Mouse samples: {num_mouse}")
        else:
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0 or batch_idx == len(test_loader_full) - 1:
                print(f"Batch {batch_idx+1}/{len(test_loader_full)}: {batch['dna'].shape[0]} samples (Human: {num_human}, Mouse: {num_mouse})")

        batch_start_time = time.time()
        
        # Aggressive cleanup before inference to free GPU memory
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        with torch.no_grad():

            # Data to predict for
            dna = batch['dna'].to(device)
            organism_index = batch['organism_index'].to(device)

            # True splice labels
            splice_labels = batch['splice_labels'].to(device)

            # True usage
            splice_usage = batch['splice_usage_target'].to(device)

            # Generate predictions
            predictions = model.inference(
                dna, 
                organism_index=organism_index,
                requested_heads=['splice_sites_classification', 'splice_sites_usage']
            )
            
            # Move predictions to CPU and collect results (inside no_grad block)
            # Important: Use .clone() to ensure we have independent copies
            splice_logits = {}
            for org in predictions:
                org_mask = organism_index == species_mapping[org]
                splice_logits_org = predictions[org]['splice_sites_classification'][org_mask]
                if args.verbose:
                    print(f"{org} splice logits shape: {splice_logits_org.shape}")
                splice_logits[org] = splice_logits_org.detach().cpu().clone()
            
            splice_usage_pred = {}
            for org in predictions:
                org_mask = organism_index == species_mapping[org]
                splice_usage_org = predictions[org]['splice_sites_usage'][org_mask]
                splice_usage_pred[org] = splice_usage_org.detach().cpu().clone()
            
            # Move labels to CPU
            organism_index = organism_index.cpu()
            splice_labels = splice_labels.cpu()
            splice_usage = splice_usage.cpu()
            
            # Free GPU memory
            del dna, predictions
            torch.cuda.empty_cache()

        # Continue processing with CPU tensors (outside no_grad block)

        # Collect predicted labels (already on CPU, outside no_grad block)
        splice_labels_pred = {}
        for org in splice_logits:
            splice_labels_pred_org = splice_logits[org].argmax(dim=-1)
            if args.verbose:
                print(f"{org} splice labels pred shape: {splice_labels_pred_org.shape}")
            splice_labels_pred[org] = splice_labels_pred_org

        splice_labels_true = {}
        for org in splice_logits:
            if (organism_index == species_mapping[org]).int().sum() == 0:
                continue

            # Get count of true classes
            if args.verbose:
                print(f"True labels for {org}:")
            splice_label_org = splice_labels[organism_index == species_mapping[org]]
            splice_labels_true[org] = splice_label_org
            splice_label_flat = splice_label_org.flatten().numpy()        
            for i in range(5):
                count = np.sum(splice_label_flat == i)
                if args.verbose:
                    print(f"  Class {i}: {count}")

            # Get count of predicted classes
            if args.verbose:
                print(f"\nPredicted labels for {org}:")
            splice_label_pred_flat = splice_labels_pred[org].flatten().numpy()
            for i in range(5):
                count = np.sum(splice_label_pred_flat == i)
                if args.verbose:
                    print(f"  Class {i}: {count}")

        # Save to dataframe
        splice_results = {}
        splice_results_df = {}
        
        # Get the actual dataset sample indices for this batch
        batch_sample_indices = batch_to_samples.get(batch_idx, list(range(batch['dna'].shape[0])))

        # Save only vectors for AUPRC calculation: for each class, save y_true and y_score arrays
        import numpy as np
        for org in splice_logits:
            if org not in splice_labels_true:
                continue
            # Convert logits to probabilities using softmax
            logits_tensor = splice_logits[org]  # shape: (N, L, 5), torch.Tensor
            probs_org = torch.softmax(logits_tensor, dim=-1).numpy()  # shape: (N, L, 5)
            labels_org = splice_labels_true[org].numpy()  # shape: (N, L)
            for class_idx in range(5):
                y_true = (labels_org == class_idx).astype(np.uint8).flatten()
                y_score = probs_org[..., class_idx].flatten()
                np.savez_compressed(f"{per_batch_dir}/auprcvec_{org}_class{class_idx}_batch{batch_idx}.npz", y_true=y_true, y_score=y_score)

        # For each organism, save detailed info for splice sites only (label != 4) for later analysis and plotting
        for org in splice_logits:

            if splice_logits[org].shape[0] == 0:
                continue

            splice_logits_flat = splice_logits[org].reshape(-1, splice_logits[org].shape[-1])
            splice_labels_flat = splice_labels_true[org].reshape(-1)

            splice_results[org] = {}
            splice_sites_ = np.where(splice_labels_true[org].numpy() != 4) # Not not-a-splice-site
            
            # Map organism-specific sample index to actual dataset index
            org_mask = organism_index == species_mapping[org]
            org_batch_indices = np.where(org_mask.numpy())[0]
            
            for i in range(len(splice_sites_[0])):
                sample_idx_in_org = splice_sites_[0][i]  # Index within organism subset
                position = splice_sites_[1][i]
                
                # Get actual dataset sample index
                batch_position = org_batch_indices[sample_idx_in_org]
                dataset_sample_idx = batch_sample_indices[batch_position]
                
                splice_labels_ = splice_labels_true[org][sample_idx_in_org, position]
                splice_logits_ = splice_logits[org][sample_idx_in_org, position, :]
                splice_preds_ = splice_logits_.softmax(dim=-1)
                splice_results[org][i] = {
                    'dataset_sample_idx': dataset_sample_idx,
                    'batch_sample_idx': sample_idx_in_org,
                    'position': position,
                    'label': splice_labels_.numpy(),
                    'pred_0': splice_preds_[0].numpy(),
                    'pred_1': splice_preds_[1].numpy(),
                    'pred_2': splice_preds_[2].numpy(),
                    'pred_3': splice_preds_[3].numpy(),
                    'pred_4': splice_preds_[4].numpy()
                }
            splice_df = pd.DataFrame.from_dict(splice_results[org], orient='index')
            splice_results_df[org] = splice_df

        # Merge dataframes for human and mouse
        splice_df = pd.concat(splice_results_df.values(), keys=splice_results_df.keys())
        splice_df.reset_index(level=0, inplace=True)
        splice_df.rename(columns={'level_0': 'organism'}, inplace=True)
        splice_df.to_csv(f"{per_batch_dir}/splice_site_predictions_batch_{batch_idx}.csv.gz", index=False, compression='gzip')
        
        # Collect for final aggregation
        all_splice_predictions.append(splice_df)
        
        # Flush to disk periodically to avoid memory buildup
        if (batch_idx + 1) % splice_flush_interval == 0:
            if all_splice_predictions:
                flush_df = pd.concat(all_splice_predictions, ignore_index=True)
                # Append to temp file
                if os.path.exists(all_splice_predictions_file):
                    flush_df.to_csv(all_splice_predictions_file, mode='a', header=False, index=False, compression='gzip')
                else:
                    flush_df.to_csv(all_splice_predictions_file, mode='w', header=True, index=False, compression='gzip')
                all_splice_predictions.clear()
                del flush_df
                if args.verbose:
                    print(f"Flushed splice predictions to disk at batch {batch_idx+1}")

        # Boxplot of predicted probabilities for each class at true classes
        plt.figure(figsize=(6, 6))
        plt.subplot(2, 2, 1)
        plt.title("Donor + (label=0)")
        plt.boxplot([splice_df[splice_df['label'] == 0][f'pred_{i}'] for i in range(5)], tick_labels=[f'{i}' for i in range(5)])
        plt.ylabel("Predicted probability")
        plt.ylim(0, 1)
        plt.subplot(2, 2, 2)
        plt.title("Acceptor + (label=1)")
        plt.boxplot([splice_df[splice_df['label'] == 1][f'pred_{i}'] for i in range(5)], tick_labels=[f'{i}' for i in range(5)])
        plt.ylabel("Predicted probability")
        plt.ylim(0, 1)
        plt.subplot(2, 2, 3)
        plt.title("Donor - (label=2)")
        plt.boxplot([splice_df[splice_df['label'] == 2][f'pred_{i}'] for i in range(5)], tick_labels=[f'{i}' for i in range(5)])
        plt.ylabel("Predicted probability")
        plt.ylim(0, 1)
        plt.subplot(2, 2, 4)
        plt.title("Acceptor - (label=3)")
        plt.boxplot([splice_df[splice_df['label'] == 3][f'pred_{i}'] for i in range(5)], tick_labels=[f'{i}' for i in range(5)])
        plt.ylabel("Predicted probability")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"{per_batch_dir}/splice_site_bins_batch_{batch_idx}.png")
        plt.close()

        # AUPRC for splice site classification
        pr_auc_scores = {}
        class_labels = {0: 'donor +', 1: 'acceptor +', 2: 'donor -', 3: 'acceptor -', 4: 'no splice site'}
        class_colors = {0: '#ff7f00', 1: '#33a02c', 2:'#fdbf6f', 3: '#b2df8a', 4: '#1f78b4'}

        for org in splice_logits.keys():

            if splice_logits[org].shape[0] == 0:
                continue

            splice_logits_flat = splice_logits[org].reshape(-1, splice_logits[org].shape[-1]).numpy()
            splice_labels_flat = splice_labels_true[org].reshape(-1).numpy()

            plt.figure(figsize=(5, 4))

            for i in range(4):
                color = class_colors[i]
                label = class_labels[i]
                
                class_idx = splice_labels_flat == i
                y_true = np.zeros_like(splice_labels_flat)
                y_true[class_idx] = 1
                y_scores = splice_logits_flat[:, i]

                if len(y_true) > 0 and y_true.sum() > 0:
                    precision, recall, _ = precision_recall_curve(y_true, y_scores)
                    pr_auc = auc(recall, precision)
                    
                    plt.plot(recall, precision, label=f"{label} (AUC={pr_auc:.2f})", color=color)
                else:
                    if args.verbose:
                        print(f"  Warning: No positive samples for {label}")

            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"PR Curve - {org} Splice Site Classification")
            plt.legend(loc='lower left')
            plt.grid()
            plt.savefig(f"{per_batch_dir}/splice_site_pr_curve_batch_{batch_idx}_{org}.png")
            plt.close()

        # Splice usage evaluation
        if args.verbose:
            print("\nEvaluating splice usage predictions at splice site positions")

        # Only evaluate splice site positions (labels != 0)
        splice_site_mask = splice_labels != 4  # (batch, seq_len)
        donor_plus_mask = splice_labels == 0
        acceptor_plus_mask = splice_labels == 1
        donor_minus_mask = splice_labels == 2
        acceptor_minus_mask = splice_labels == 3

        if splice_site_mask.any():

            usage_true_bins_ = {}
            usage_pred_ = {}
            usage_true_ = {}

            for org in splice_usage_pred:
                
                if splice_logits[org].shape[0] == 0:
                    continue

                if args.verbose:
                    print(f"\nEvaluating splice usage predictions for {org} at splice site positions:")

                # Predicted and true splice usage (already on CPU)
                org_sse_pred = splice_usage_pred[org].sigmoid()
                org_sse_true = splice_usage

                # Select only splice site positions
                usage_true_at_sites = org_sse_true[splice_site_mask, :]
                usage_pred_at_sites = org_sse_pred[splice_site_mask, :]

                # Subset by organism
                usage_true_at_sites = usage_true_at_sites[:,species_conds[org]]
                usage_pred_at_sites = usage_pred_at_sites[:,range(len(species_conds[org]))]

                # Convert to numpy for evaluation
                usage_true_org = usage_true_at_sites.numpy()
                usage_pred_org = usage_pred_at_sites.numpy()

                # Remove rows with all zeros in true usage (not used for evaluation)
                non_zero_rows = ~np.all(usage_true_org == 0, axis=1)
                usage_true_org = usage_true_org[non_zero_rows]
                usage_pred_org = usage_pred_org[non_zero_rows]

                # Flatten for plotting
                cols = species_conds[org]  # Get columns corresponding to the condition of the organism
                usage_pred_[org] = usage_pred_org.flatten()
                usage_true_[org] = usage_true_org.flatten()

                # Group true usage in bins: 0 - 0.2 - 0.5 - 0.8 - 1.0
                usage_true_bins = np.zeros_like(usage_true_[org], dtype=int)
                usage_true_bins[usage_true_[org] < 0.2] = 0
                usage_true_bins[(usage_true_[org] >= 0.2) & (usage_true_[org] < 0.5)] = 1
                usage_true_bins[(usage_true_[org] >= 0.5) & (usage_true_[org] < 0.8)] = 2
                usage_true_bins[usage_true_[org] >= 0.8] = 3

                usage_true_bins_[org] = usage_true_bins

                # Count number of values in each bin
                for i in range(4):
                    count = np.sum(usage_true_bins == i)
                    if args.verbose:
                        print(f"True usage bin {i}: {count} values")

        # Boxplot of predicted usage for each true usage bin
        for org in usage_pred_.keys():

            if splice_logits[org].shape[0] == 0:
                continue

            plt.figure(figsize=(4, 4))
            plt.title(f"{org} splice usage")
            plt.boxplot([usage_pred_[org][usage_true_bins_[org] == i] for i in range(4)], tick_labels=['0-0.2', '0.2-0.5', '0.5-0.8', '0.8-1.0'])
            plt.ylabel("Predicted splice usage")
            plt.xlabel("True splice usage bins")
            plt.savefig(f"{per_batch_dir}/splice_usage_bins_batch_{batch_idx}_{org}.png")
            plt.close()

        # Correlation scatterplot
        for org in usage_pred_.keys():

            if splice_logits[org].shape[0] == 0:
                continue

            # Calculate correlation between true and predicted usage at splice sites
            correlation = np.corrcoef(usage_true_[org], usage_pred_[org])[0, 1]
            if args.verbose:
                print(f"Correlation between true and predicted splice usage for {org}: {correlation:.3f}")

            # Plot scatter plot of predicted usage vs true usage
            plt.figure(figsize=(4, 4))
            plt.title(f"{org} predicted vs true splice usage")
            plt.scatter(usage_true_[org], usage_pred_[org], alpha=0.5, s=10)
            plt.xlabel("True splice usage")
            plt.ylabel("Predicted splice usage")
            plt.xlim(0, 1)
            plt.ylim(0, 1)

            # Add correlation text
            plt.text(0.05, 0.95, f"Correlation: {correlation:.3f}", transform=plt.gca().transAxes, verticalalignment='top')

            # Save plot
            plt.savefig(f"{per_batch_dir}/splice_usage_scatter_batch_{batch_idx}_{org}.png")
            plt.close()

            # Combine in a DataFrame: org, condition_idx, true_sse, pred_sse
            data_arrays = {
                'organism': [],
                'dataset_sample_idx': [],
                'batch_sample_idx': [],
                'position': [],
                'true_usage': [],
                'pred_usage': [],
                'condition_idx': [],
                'condition_name': [],
                'tissue': [],
                'timepoint': []
            }
            
            # Get the actual dataset sample indices for this batch
            batch_sample_indices = batch_to_samples.get(batch_idx, list(range(batch['dna'].shape[0])))

            for org in splice_usage_pred:
                if args.verbose:
                    print(f"\nBuilding dataframe for {org}:")
                
                # Get organism-specific mask
                org_mask = organism_index == species_mapping[org]
                
                # Get organism-specific predictions and true values (already on CPU)
                org_sse_pred = splice_usage_pred[org].sigmoid()
                org_sse_true = splice_usage[org_mask]
                
                # Get splice site positions for this organism
                org_splice_mask = splice_site_mask[org_mask]
                
                # Get sequence and position indices for splice sites
                splice_site_indices = torch.where(org_splice_mask)
                sequence_indices = splice_site_indices[0].numpy()  # Which sequence (dim 0) within organism
                position_indices = splice_site_indices[1].numpy()  # Which position (dim 1)
                
                # Map organism-specific batch indices to actual dataset indices
                org_batch_indices = np.where(org_mask.numpy())[0]
                dataset_indices = np.array([batch_sample_indices[idx] for idx in org_batch_indices])
                
                # Select only splice site positions
                usage_true_at_sites = org_sse_true[org_splice_mask, :]
                usage_pred_at_sites = org_sse_pred[org_splice_mask, :]
                
                # Get condition indices for this organism
                org_conds = species_conds[org]
                num_sites = usage_true_at_sites.shape[0]
                num_conds = len(org_conds)  # shape: (num_sites, num_total_conditions)
                pred_vals = usage_pred_at_sites
                
                # Convert to numpy once
                true_vals = usage_true_at_sites  # shape: (num_sites, num_total_conditions)
                pred_vals = usage_pred_at_sites  # shape: (num_sites, num_org_conditions)
                
                # Build arrays for this organism
                for cond_idx_pred, cond_idx_true in enumerate(org_conds):
                    cond_name = metadata['usage_conditions'][cond_idx_true]
                    tissue, timepoint = cond_name.split('_')
                    data_arrays['organism'].append(np.full(num_sites, org, dtype=object))
                    data_arrays['dataset_sample_idx'].append(dataset_indices[sequence_indices])
                    data_arrays['batch_sample_idx'].append(sequence_indices)
                    data_arrays['position'].append(position_indices)
                    data_arrays['true_usage'].append(true_vals[:, cond_idx_true])
                    data_arrays['pred_usage'].append(pred_vals[:, cond_idx_pred])
                    data_arrays['condition_idx'].append(np.full(num_sites, cond_idx_true, dtype=int))
                    data_arrays['condition_name'].append(np.full(num_sites, cond_name, dtype=object))
                    data_arrays['tissue'].append(np.full(num_sites, tissue, dtype=object))
                    data_arrays['timepoint'].append(np.full(num_sites, timepoint, dtype=object))
                if args.verbose:
                    print(f"  Added {num_sites * num_conds} rows for {org}")

            # Concatenate all arrays at once
            all_data_df = pd.DataFrame({
                'organism': np.concatenate(data_arrays['organism']),
                'dataset_sample_idx': np.concatenate(data_arrays['dataset_sample_idx']),
                'batch_sample_idx': np.concatenate(data_arrays['batch_sample_idx']),
                'position': np.concatenate(data_arrays['position']),
                'true_usage': np.concatenate(data_arrays['true_usage']),
                'pred_usage': np.concatenate(data_arrays['pred_usage']),
                'condition_idx': np.concatenate(data_arrays['condition_idx']),
                'condition_name': np.concatenate(data_arrays['condition_name']),
                'tissue': np.concatenate(data_arrays['tissue']),
                'timepoint': np.concatenate(data_arrays['timepoint'])
            })

            if args.verbose:
                print(f"\nTotal rows in dataframe: {len(all_data_df)}")

            # How many conditions/tissues/timepoints have true_usage > 0 for each site?
            site_stats = all_data_df[all_data_df['true_usage'] > 0].groupby(['dataset_sample_idx', 'position']).agg({
                'condition_idx': 'nunique',
                'timepoint': 'nunique',
                'tissue': 'nunique'
            }).rename(columns={
                'condition_idx': 'num_conditions',
                'timepoint': 'num_timepoints',
                'tissue': 'num_tissues'
            })

            # Merge back to the original dataframe
            all_data_df = all_data_df.merge(site_stats, on=['dataset_sample_idx', 'position'], how='left').fillna(0)
            all_data_df['num_conditions'] = all_data_df['num_conditions'].astype(int)
            all_data_df['num_timepoints'] = all_data_df['num_timepoints'].astype(int)
            all_data_df['num_tissues'] = all_data_df['num_tissues'].astype(int)

            # Only evaluate position with usage data
            all_data_df = all_data_df[all_data_df['num_conditions'] > 0]

            if args.verbose:
                print(f"\nTotal rows in dataframe: {len(all_data_df)}")

            # Save usage predictions at splice sites to CSV
            all_data_df.to_csv(f"{per_batch_dir}/splice_usage_batch_{batch_idx}.csv.gz", index=False, compression='gzip')
            
            # Collect for final aggregation
            all_usage_predictions.append(all_data_df)
            
            # Flush to disk periodically to avoid memory buildup
            if (batch_idx + 1) % usage_flush_interval == 0:
                if all_usage_predictions:
                    flush_df = pd.concat(all_usage_predictions, ignore_index=True)
                    # Append to temp file
                    if os.path.exists(all_usage_predictions_file):
                        flush_df.to_csv(all_usage_predictions_file, mode='a', header=False, index=False, compression='gzip')
                    else:
                        flush_df.to_csv(all_usage_predictions_file, mode='w', header=True, index=False, compression='gzip')
                    all_usage_predictions.clear()
                    del flush_df
                    if args.verbose:
                        print(f"Flushed usage predictions to disk at batch {batch_idx+1}")

            # Density plot of predicted usage at true usage > 0 positions, grouped by organism
            plot_sse_density(
                all_data_df,
                group_by=['organism'],
                output_fn=f"{per_batch_dir}/splice_usage_density_batch_{batch_idx}_by_organism.png"
            )
            plot_sse_density(
                all_data_df,
                group_by=['organism', 'tissue'],
                output_fn=f"{per_batch_dir}/splice_usage_density_batch_{batch_idx}_by_organism_tissue.png"
            )
            
            batch_time = time.time() - batch_start_time
            batch_stats['inference_time'].append(batch_time)
            if args.verbose:
                print(f"\nFinished evaluation for batch {batch_idx+1}/{len(test_loader_full)} in {batch_time/60:.2f} minutes\n")
            
            # Clear remaining references and free memory
            del splice_logits, splice_usage_pred, splice_labels_pred, splice_labels_true
            del organism_index, splice_labels, splice_usage
            del batch  # Also delete the batch dict
            if 'all_data_df' in locals():
                del all_data_df
            if 'splice_df' in locals():
                del splice_df
            if 'splice_results' in locals():
                del splice_results, splice_results_df
            
            # Aggressive garbage collection every batch to prevent memory buildup
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            if (batch_idx + 1) % 10 == 0:
                if args.verbose:
                    print(f"Performed garbage collection at batch {batch_idx+1}")
            if args.verbose:
                print("-"*60)

total_elapsed_time = time.time() - start_time
print(f"\nAll batches completed in {total_elapsed_time/60:.2f} minutes")

# ============================================================================
# Create final aggregated plots and metrics
# ============================================================================
print("\n" + "="*60)
print("Generating Final Aggregated Plots")
print("="*60)

# Concatenate all splice predictions (including flushed data)
print("\nCreating final splice site classification plots...")

# First, flush any remaining data
if all_splice_predictions:
    flush_df = pd.concat(all_splice_predictions, ignore_index=True)
    if os.path.exists(all_splice_predictions_file):
        flush_df.to_csv(all_splice_predictions_file, mode='a', header=False, index=False, compression='gzip')
    else:
        flush_df.to_csv(all_splice_predictions_file, mode='w', header=True, index=False, compression='gzip')
    all_splice_predictions.clear()
    del flush_df

# Read back all data
if os.path.exists(all_splice_predictions_file):
    final_splice_df = pd.read_csv(all_splice_predictions_file, compression='gzip')
    final_splice_df.to_csv(f"{output_dir}/splice_site_predictions_all.csv.gz", index=False, compression='gzip')
    os.remove(all_splice_predictions_file)  # Clean up temp file
    print(f"  Saved: {output_dir}/splice_site_predictions_all.csv.gz")
    
    # Final boxplot of predicted probabilities
    plt.figure(figsize=(6, 6))
    plt.subplot(2, 2, 1)
    plt.title("Donor + (label=0)")
    plt.boxplot([final_splice_df[final_splice_df['label'] == 0][f'pred_{i}'] for i in range(5)], 
                tick_labels=[f'{i}' for i in range(5)])
    plt.ylabel("Predicted probability")
    plt.ylim(0, 1)
    plt.subplot(2, 2, 2)
    plt.title("Acceptor + (label=1)")
    plt.boxplot([final_splice_df[final_splice_df['label'] == 1][f'pred_{i}'] for i in range(5)], 
                tick_labels=[f'{i}' for i in range(5)])
    plt.ylabel("Predicted probability")
    plt.ylim(0, 1)
    plt.subplot(2, 2, 3)
    plt.title("Donor - (label=2)")
    plt.boxplot([final_splice_df[final_splice_df['label'] == 2][f'pred_{i}'] for i in range(5)], 
                tick_labels=[f'{i}' for i in range(5)])
    plt.ylabel("Predicted probability")
    plt.ylim(0, 1)
    plt.subplot(2, 2, 4)
    plt.title("Acceptor - (label=3)")
    plt.boxplot([final_splice_df[final_splice_df['label'] == 3][f'pred_{i}'] for i in range(5)], 
                tick_labels=[f'{i}' for i in range(5)])
    plt.ylabel("Predicted probability")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/splice_site_predictions_final.png", dpi=150)
    plt.close()
    print(f"  Saved: {output_dir}/splice_site_predictions_final.png")

    # --- Final AUPRC calculation using saved vectors ---
    import glob
    class_labels = {0: 'donor +', 1: 'acceptor +', 2: 'donor -', 3: 'acceptor -', 4: 'no splice site'}
    class_colors = {0: '#ff7f00', 1: '#33a02c', 2: '#fdbf6f', 3: '#b2df8a', 4: '#1f78b4'}
    import numpy as np
    orgs = set()
    for fn in glob.glob(f"{per_batch_dir}/auprcvec_*_class0_batch*.npz"):
        org = fn.split("auprcvec_")[1].split("_class")[0]
        orgs.add(org)
    for org in orgs:
        plt.figure(figsize=(5, 4))
        final_auprc = {}
        for class_idx in range(4):
            # Gather all y_true and y_score arrays for this org/class
            files = sorted(glob.glob(f"{per_batch_dir}/auprcvec_{org}_class{class_idx}_batch*.npz"))
            print(f"Processing {len(files)} files for {org} class {class_idx}")
            y_true_all = []
            y_score_all = []
            for fn in files:
                arr = np.load(fn)
                y_true_all.append(arr['y_true'])
                y_score_all.append(arr['y_score'])
            if y_true_all:
                y_true = np.concatenate(y_true_all)
                y_score = np.concatenate(y_score_all)
                color = class_colors[class_idx]
                label = class_labels[class_idx]
                if y_true.sum() > 0:
                    print(f"Calculating AUPRC for {org} class {label} with {len(y_true)} samples and {y_true.sum()} positives")
                    precision, recall, _ = precision_recall_curve(y_true, y_score)
                    pr_auc = auc(recall, precision)
                    final_auprc[label] = pr_auc
                    plt.plot(recall, precision, label=f"{label} (AUC={pr_auc:.2f})", color=color)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Final PR Curve (All Sites) - {org}")
        plt.legend(loc='lower left')
        plt.grid()
        plt.savefig(f"{output_dir}/splice_site_pr_curve_final_all_sites_{org}.png", dpi=150)
        plt.close()
        print(f"  Saved: {output_dir}/splice_site_pr_curve_final_all_sites_{org}.png")

# Concatenate all usage predictions (including flushed data)
print("\nCreating final splice usage plots...")

# First, flush any remaining data
if all_usage_predictions:
    flush_df = pd.concat(all_usage_predictions, ignore_index=True)
    if os.path.exists(all_usage_predictions_file):
        flush_df.to_csv(all_usage_predictions_file, mode='a', header=False, index=False, compression='gzip')
    else:
        flush_df.to_csv(all_usage_predictions_file, mode='w', header=True, index=False, compression='gzip')
    all_usage_predictions.clear()
    del flush_df

# Read back all data
if os.path.exists(all_usage_predictions_file):
    final_usage_df = pd.read_csv(all_usage_predictions_file, compression='gzip')
    final_usage_df.to_csv(f"{output_dir}/splice_usage_predictions_all.csv.gz", index=False, compression='gzip')
    os.remove(all_usage_predictions_file)  # Clean up temp file
    print(f"  Saved: {output_dir}/splice_usage_predictions_all.csv.gz")
    
    # Final density plots
    plot_sse_density(
        final_usage_df,
        group_by=['organism'],
        output_fn=f"{output_dir}/splice_usage_density_final_by_organism.png"
    )
    print(f"  Saved: {output_dir}/splice_usage_density_final_by_organism.png")
    
    plot_sse_density(
        final_usage_df,
        group_by=['organism', 'tissue'],
        output_fn=f"{output_dir}/splice_usage_density_final_by_organism_tissue.png"
    )
    print(f"  Saved: {output_dir}/splice_usage_density_final_by_organism_tissue.png")
    
    # Final correlation by organism
    print("\nFinal correlations by organism:")
    for org in final_usage_df['organism'].unique():
        org_data = final_usage_df[final_usage_df['organism'] == org]
        valid_data = org_data.dropna(subset=['true_usage', 'pred_usage'])
        if len(valid_data) > 0:
            corr = np.corrcoef(valid_data['true_usage'], valid_data['pred_usage'])[0, 1]
            print(f"  {org}: r = {corr:.3f} (n={len(valid_data)})")


# Save batch statistics only if prediction was run (batch_stats defined)
if not args.skip_prediction:
    if batch_stats['batch_idx']:
        batch_stats_df = pd.DataFrame(batch_stats)
        batch_stats_df.to_csv(f"{output_dir}/batch_statistics.csv", index=False)
        print(f"\nSaved: {output_dir}/batch_statistics.csv")

print("\n" + "="*60)
print("Output Files Summary")
print("="*60)
print(f"\nBatch-to-Sample Mapping:")
print(f"  {output_dir}/batch_sample_mapping.csv")
print(f"  - Contains mapping of batch_idx to dataset sample indices")
print(f"  - Includes all genomic region metadata (chromosome, gene_id, etc.)")
print(f"\nPer-Batch Files:")
print(f"  {per_batch_dir}/")
print(f"  - splice_site_predictions_batch_*.csv.gz")
print(f"  - splice_usage_batch_*.csv.gz")
print(f"  - Various plots per batch")
print(f"\nFinal Aggregated Files:")
print(f"  {output_dir}/splice_site_predictions_all.csv.gz")
print(f"  {output_dir}/splice_usage_predictions_all.csv.gz")
print(f"  {output_dir}/batch_statistics.csv")
print(f"  {output_dir}/splice_site_predictions_final.png")
print(f"  {output_dir}/splice_site_pr_curve_final_*.png")
print(f"  {output_dir}/splice_usage_density_final_*.png")
print(f"\nTo map predictions to genomic coordinates:")
print(f"  1. Load batch_sample_mapping.csv")
print(f"  2. Join with prediction files on 'dataset_sample_idx'")
print(f"  3. Access chromosome, gene_id, window_start, window_end, etc.")
print("="*60)