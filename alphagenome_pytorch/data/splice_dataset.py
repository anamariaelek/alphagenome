"""Dataset for splice site prediction using memory-mapped processed data"""

import os
import json
import gzip
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from torch.utils.data import Dataset


class SpliceDataset(Dataset):
    """Dataset for splice site prediction with memory-mapped data"""
    
    def __init__(
        self,
        data_dir,
        target_length=None,
        max_donor_sites=20,
        max_acceptor_sites=20,
        max_sequence_length=1048576,
        species_mapping=None,
        load_alpha=False,
        load_beta=False,
    ):
        """
        Args:
            data_dir: Directory containing memory-mapped data files
            target_length: Target sequence length. If None (default), each
                sequence is padded/cropped to the smallest multiple of 2048
                that is >= its length. Can also be an explicit int.
            max_donor_sites: Maximum number of donor sites per sequence
            max_acceptor_sites: Maximum number of acceptor sites per sequence
            species_mapping: Dict mapping species names to organism indices
            load_alpha: Whether to load alpha values
            load_beta: Whether to load beta values
        """
        self.data_dir = data_dir
        self.target_length = target_length
        self.max_sequence_length = max_sequence_length
        self.max_donor_sites = max_donor_sites
        self.max_acceptor_sites = max_acceptor_sites
        self.load_alpha = load_alpha
        self.load_beta = load_beta

        # Default species mapping
        self.species_mapping = species_mapping or {
            'human': 0,
            'mouse': 1,
            'rat': 2
        }
        
        # Load metadata JSON
        meta_path = os.path.join(self.data_dir, 'metadata.json')
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)
        
        # Load metadata CSV
        meta_csv_path = os.path.join(self.data_dir, 'metadata.csv')
        self.meta_csv = pd.read_csv(meta_csv_path)
        self.species = self.meta_csv['species_id']

        #
        # Sequences
        #

        # Load sequences based on specified format
        if self.meta.get('sequences_format') is None or self.meta.get('sequences_format') == 'mmap':
            # Load sequences (one-hot encoded)
            seq_path = os.path.join(self.data_dir, 'sequences.mmap')
            seq_dtype = np.dtype(self.meta.get('sequences_dtype'))
            seq_shape = tuple(self.meta.get('sequences_shape'))
            self.sequences = np.memmap(seq_path, dtype=seq_dtype, mode='r', shape=seq_shape)
            self.lengths = self.sequences.shape[1] if len(seq_shape) == 3 else self.sequences.shape[0]
        elif self.meta.get('sequences_format') == 'fasta.gz' or self.meta.get('sequences_format') == 'fasta':
            # Load sequences from fasta
            fasta_path = os.path.join(self.data_dir, 'sequences.' + self.meta.get('sequences_format'))
            opener = gzip.open if str(fasta_path).endswith('.gz') else open
            self.sequences = []
            self.lengths = []
            with opener(fasta_path, 'rt') as f:
                for record in SeqIO.parse(f, 'fasta'):    
                    seq_str = str(record.seq).upper()
                    # Store sequence lengths for bucketing
                    self.lengths.append(len(seq_str))
                    seq_array = np.zeros((len(seq_str), 4), dtype=np.uint8)
                    for i, base in enumerate(seq_str):
                        if base == 'A':
                            seq_array[i, 0] = 1
                        elif base == 'C':
                            seq_array[i, 1] = 1
                        elif base == 'G':
                            seq_array[i, 2] = 1
                        elif base == 'T':
                            seq_array[i, 3] = 1
                    self.sequences.append(seq_array)
            # Keep as list — sequences may have different lengths
        else:
            raise ValueError(f"Unsupported sequences format: {self.meta.get('sequences_format')}")
        
        # Keep context: do not crop here
        self.context_size = self.meta.get('context_size', 0)
        self.window_size = self.meta.get('window_size', 0)

        #
        # Labels
        #
        
        # Load labels (splice site annotations)
        lbl_path = os.path.join(self.data_dir, 'labels.parquet') # 0=not splice site 1=donor 2=acceptor
        self.labels = pd.read_parquet(lbl_path)

        # Split by strand
        self.labels['strand'] = self.labels['sample_idx'].map(self.meta_csv['strand'])
        self.labels.loc[self.labels['strand'] == '-', 'label'] += 2 # 0=not splice site, 1=donor+, 2=acceptor+, 3=donor-, 4=acceptor-

        # Match alphagenome classes
        self.labels['label'] = self.labels['label'].replace({0:4, 1:0, 2:1, 3:2, 4:3}) # 0=donor+ 1=acceptor+ 2=donor- 3=acceptor- 4=not splice site
            
        # Account for context
        self.labels['position'] = self.labels['position'].astype(np.int64)  # Ensure position is int64 for safe addition
        self.labels['position'] += self.context_size

        # Account for different splice site encoding
        self.labels.loc[self.labels['label'] == 0, 'position'] += -2 
        self.labels.loc[self.labels['label'] == 3, 'position'] += -2 
        
        #
        # Usage (SSE)
        #

        # Load SSE (splice site strength estimate) if available
        sse_path = os.path.join(self.data_dir, 'usage.parquet')
        if os.path.exists(sse_path):
            self.sse = pd.read_parquet(sse_path)
        else:
            self.sse = None

        # Account for context
        self.sse['position'] = self.sse['position'].astype(np.int64)  # Ensure position is int64 for safe addition
        self.sse['position'] += self.context_size

        # Add splice class
        labels_ = self.labels.set_index(['sample_idx', 'position', 'strand'], inplace=False)
        usage_ = self.sse.set_index(['sample_idx', 'position', 'strand'], inplace=False)
        self.sse = usage_.join(labels_, how='left').reset_index()

        # Account for different splice site encoding
        self.sse.loc[self.sse['label'] == 0, 'position'] += -2 
        self.sse.loc[self.sse['label'] == 3, 'position'] += -2 

        # Load conditions for species 
        meta_path = os.path.join(self.data_dir, 'metadata.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        self.condition_map = meta.get('species_condition_mapping', {})
        
        print(f"Loaded dataset from {self.data_dir}")

    def __len__(self):
        return self.sequences.shape[0] if isinstance(self.sequences, np.ndarray) else len(self.sequences)
    
    def __getitem__(self, idx):
        # Convert one-hot to integer encoding
        seq = self.sequences[idx]  # (seq_len, 4) or (4, seq_len)
        seq_len = seq.shape[0] if seq.shape[-1] == 4 else seq.shape[1]

        # Reverse complement strand - sequences
        strand = self.meta_csv.loc[idx, 'strand']
        if strand == '-':
            if seq.shape[-1] == 4:
                seq = seq[:, ::-1]  # Complement (seq_len, 4)
            else:
                seq = seq[::-1, :]  # Complement (4, seq_len)

        # Convert to integer encoding (0=A, 1=C, 2=G, 3=T, -1=padding)
        if seq.shape[-1] == 4:
            dna = np.argmax(seq, axis=-1)  # (seq_len,)
        else:
            dna = np.argmax(seq, axis=0)  # (seq_len,)

        # Determine target length
        original_length = len(dna)
        if self.target_length is None or str(self.target_length).lower() == 'none':
            # Round up to the nearest multiple of 2048
            target_length = int(np.ceil(original_length / 2048)) * 2048
        else:
            target_length = int(self.target_length)
        self.max_sequence_length = int(self.max_sequence_length)
        target_length = min(target_length, self.max_sequence_length)

        # Crop/pad to target length
        current_length = len(dna)
        if current_length > target_length:
            crop_start = (current_length - target_length) // 2
            crop_end = crop_start + target_length
            dna_cropped = dna[crop_start:crop_end]
        elif current_length < target_length:
            crop_start = 0
            crop_end = current_length
            padding = target_length - current_length
            dna_cropped = np.pad(dna, (0, padding), constant_values=-1)
        else:
            crop_start = 0
            crop_end = target_length
            dna_cropped = dna

        # # Labels

        # Extract and adjust splice site positions
        label_seq = self.labels[self.labels['sample_idx'] == idx]

        # Create dense label array from sparse DataFrame
        splice_labels_dense = np.full(current_length, fill_value=4, dtype=np.int64)
        for _, row in label_seq.iterrows():
            pos = int(row['position'])
            label = int(row['label'])
            if 0 <= pos < current_length:
                splice_labels_dense[pos] = label
        
        # Crop/pad splice labels to target length
        if current_length > target_length:
            splice_labels_final = splice_labels_dense[crop_start:crop_end]
        elif current_length < target_length:
            padding = target_length - current_length
            splice_labels_final = np.pad(splice_labels_dense, (0, padding), constant_values=4)
        else:
            splice_labels_final = splice_labels_dense

        # Find donor and acceptor positions for junctions
        donor_pos = np.where(np.isin(splice_labels_final, [0, 2]))[0]
        acceptor_pos = np.where(np.isin(splice_labels_final, [1, 3]))[0]

        # Store actual counts of acceptor/donor sites
        num_donors = len(donor_pos)
        num_acceptors = len(acceptor_pos)

        # Fallback if no sites found
        if len(donor_pos) == 0:
            donor_pos = np.array([0])
        if len(acceptor_pos) == 0:
            acceptor_pos = np.array([0])

        # Pad to fixed size
        donor_padded = np.pad(donor_pos[:self.max_donor_sites],
                             (0, max(0, self.max_donor_sites - len(donor_pos))),
                             mode='edge')[:self.max_donor_sites]
        acceptor_padded = np.pad(acceptor_pos[:self.max_acceptor_sites],
                                (0, max(0, self.max_acceptor_sites - len(acceptor_pos))),
                                mode='edge')[:self.max_acceptor_sites]

        # # Usage

        # Get organism index
        species_id = self.species.iloc[idx]
        if isinstance(species_id, (int, np.integer)):
            organism_idx = int(species_id)
        else:
            organism_idx = self.species_mapping.get(species_id, 0)

        # Get organism name
        organism_idx_to_name = {v: k for k, v in self.species_mapping.items()}
        organism_name = organism_idx_to_name.get(organism_idx, 'unknown')

        # Get usage SSE for this sequence if available
        if self.sse is not None:
            n_conds = self.sse['condition_idx'].nunique()
            sse_seq = self.sse[self.sse['sample_idx'] == idx]
            sse = np.zeros((current_length, n_conds), dtype=np.float32)
            if self.load_alpha:
                alpha = np.zeros((current_length, n_conds), dtype=np.float32)
            if self.load_beta:
                beta = np.zeros((current_length, n_conds), dtype=np.float32)
            for _, row in sse_seq.iterrows():
                position = int(row['position'])
                condition_idx = int(row['condition_idx'])
                sse_value = row['sse']
                if self.load_alpha:
                    alpha_value = row.get('alpha', 0)
                    alpha_value = int(alpha_value) if alpha_value is not None else 0
                if self.load_beta:
                    beta_value = row.get('beta', 0)
                    beta_value = int(beta_value) if beta_value is not None else 0
                if 0 <= position < current_length:
                    sse[position, condition_idx] = sse_value
                    if self.load_alpha:
                        alpha[position, condition_idx] = alpha_value
                    if self.load_beta:
                        beta[position, condition_idx] = beta_value
            if current_length > target_length:
                sse_target = sse[crop_start:crop_end]
                if self.load_alpha:
                    alpha_target = alpha[crop_start:crop_end]
                if self.load_beta:
                    beta_target = beta[crop_start:crop_end]
            elif current_length < target_length:
                padding = target_length - current_length
                sse_target = np.pad(sse, ((0, padding), (0, 0)), constant_values=0)
                if self.load_alpha:
                    alpha_target = np.pad(alpha, ((0, padding), (0, 0)), constant_values=0)
                if self.load_beta:
                    beta_target = np.pad(beta, ((0, padding), (0, 0)), constant_values=0)
            else:
                sse_target = sse
                if self.load_alpha:
                    alpha_target = alpha
                if self.load_beta:
                    beta_target = beta
        else:
            sse_target = np.zeros((target_length, 1), dtype=np.float32)
            if self.load_alpha:
                alpha_target = np.zeros((target_length, 1), dtype=np.float32)
            if self.load_beta:
                beta_target = np.zeros((target_length, 1), dtype=np.float32)

        # Get conditions mask for this sequence (per-sequence depending on organism)
        conditions_mask_array = self.condition_map.get(organism_name, [])

        # Compute gene region relative coordinates
        row = self.meta_csv.iloc[idx]
        if 'central_gene_start' in row and 'central_gene_end' in row and 'window_with_context_start' in row and 'window_with_context_end' in row:
            gene_start_abs = int(row['central_gene_start'])
            gene_end_abs = int(row['central_gene_end'])
            window_with_context_start = int(row['window_with_context_start'])
            window_with_context_end = int(row['window_with_context_end'])
            # Relative to sequence start (with context)
            gene_start_rel = gene_start_abs - window_with_context_start
            gene_end_rel = gene_end_abs - window_with_context_start
            # Adjust for cropping
            if current_length > target_length:
                gene_start_rel = gene_start_rel - crop_start
                gene_end_rel = gene_end_rel - crop_start
            # Clamp to valid range
            gene_start_rel = max(0, min(target_length, gene_start_rel))
            gene_end_rel = max(0, min(target_length, gene_end_rel))
        else:
            gene_start_rel = 0
            gene_end_rel = target_length

        return_dict = {
            'dna': torch.tensor(dna_cropped, dtype=torch.long),
            'organism_index': torch.tensor(organism_idx, dtype=torch.long),
            'splice_donor_idx': torch.tensor(donor_padded, dtype=torch.long),
            'splice_acceptor_idx': torch.tensor(acceptor_padded, dtype=torch.long),
            'num_donors': torch.tensor(num_donors, dtype=torch.long),
            'num_acceptors': torch.tensor(num_acceptors, dtype=torch.long),
            'splice_labels': torch.tensor(splice_labels_final, dtype=torch.long),
            'splice_usage_target': torch.tensor(sse_target, dtype=torch.float32),
            'conditions_mask': torch.tensor(conditions_mask_array, dtype=torch.long),  # (num_contexts,)
            'gene_region': torch.tensor([gene_start_rel, gene_end_rel], dtype=torch.long),  # [start, end] relative to sequence
        }
        if self.load_alpha:
            return_dict['alpha_target'] = torch.tensor(alpha_target, dtype=torch.float32)
        if self.load_beta:
            return_dict['beta_target'] = torch.tensor(beta_target, dtype=torch.float32)
        
        return return_dict
