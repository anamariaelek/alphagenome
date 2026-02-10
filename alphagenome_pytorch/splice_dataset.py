"""Dataset for splice site prediction using memory-mapped processed data"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SpliceDataset(Dataset):
    """Dataset for splice site prediction with memory-mapped data"""
    
    def __init__(
        self,
        data_dir,
        target_length=4096,
        max_donor_sites=20,
        max_acceptor_sites=20,
        species_mapping=None
    ):
        """
        Args:
            data_dir: Directory containing memory-mapped data files
            target_length: Target sequence length (must be power of 2)
            max_donor_sites: Maximum number of donor sites per sequence
            max_acceptor_sites: Maximum number of acceptor sites per sequence
            species_mapping: Dict mapping species names to organism indices
        """
        self.data_dir = data_dir
        self.target_length = target_length
        self.max_donor_sites = max_donor_sites
        self.max_acceptor_sites = max_acceptor_sites
        
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

        # Load sequences (one-hot encoded)
        seq_path = os.path.join(self.data_dir, 'sequences.mmap')
        seq_dtype = np.dtype(self.meta.get('sequences_dtype'))
        seq_shape = tuple(self.meta.get('sequences_shape'))
        self.sequences = np.memmap(seq_path, dtype=seq_dtype, mode='r', shape=seq_shape)

        # Handle context size if present
        context_size = self.meta.get('context_size', 0)
        if context_size > 0:
            self.sequences = self.sequences[:, context_size:-context_size]

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

        # Add splice class
        labels_ = self.labels.set_index(['sample_idx', 'position'], inplace=False)
        usage_ = self.sse.set_index(['sample_idx', 'position'], inplace=False)
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
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Convert one-hot to integer encoding
        seq = self.sequences[idx]  # (seq_len, 4) or (4, seq_len)
               
        # Reverse complement strand - sequences
        if self.meta_csv.loc[idx, 'strand'] == '-':
            if seq.shape[-1] == 4:
                seq = seq[:, ::-1]  # Complement (seq_len, 4)
            else:
                seq = seq[:, ::-1]  # Complement (4, seq_len)

        # Convert to integer encoding (0=A, 1=C, 2=G, 3=T, -1=padding)
        if seq.shape[-1] == 4:
            dna = np.argmax(seq, axis=-1)  # (seq_len,)
        else:
            dna = np.argmax(seq, axis=0)  # (seq_len,)

        # Crop/pad to target length
        current_length = len(dna)
        if current_length > self.target_length:
            # Center crop
            crop_start = (current_length - self.target_length) // 2
            dna = dna[crop_start:crop_start + self.target_length]
        elif current_length < self.target_length:
            # Pad with -1s
            padding = self.target_length - current_length
            dna = np.pad(dna, (0, padding), constant_values=-1)
            crop_start = 0
        else:
            crop_start = 0
        crop_end = crop_start + self.target_length

        # Extract and adjust splice site positions
        label_seq = self.labels[self.labels['sample_idx'] == idx]

        # Labels on - strand are now indexed from the end of the sequence, so we need to adjust positions
        #if self.meta_csv.loc[idx, 'strand'] == '-':
        #    label_seq.loc[:, 'position'] = current_length - 1 - label_seq['position']
        
        # Create dense label array from sparse DataFrame
        splice_labels_dense = np.full(current_length, fill_value=4, dtype=np.int64)
        for _, row in label_seq.iterrows():
            pos = int(row['position'])
            label = int(row['label'])
            if 0 <= pos < current_length:
                splice_labels_dense[pos] = label
        
        # Find donor and acceptor positions
        donor_pos = label_seq[(label_seq['label'] == 0) | (label_seq['label'] == 2)]['position'].values
        acceptor_pos = label_seq[(label_seq['label'] == 1) | (label_seq['label'] == 3)]['position'].values
        
        # Adjust for cropping
        donor_pos = donor_pos[(donor_pos >= crop_start) & (donor_pos < crop_end)] - crop_start
        acceptor_pos = acceptor_pos[(acceptor_pos >= crop_start) & (acceptor_pos < crop_end)] - crop_start
        
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
        
        # Crop/pad splice labels to target length
        if current_length > self.target_length:
            # Center crop
            splice_labels_final = splice_labels_dense[crop_start:crop_end]
        elif current_length < self.target_length:
            # Pad with zeros
            padding = self.target_length - current_length
            splice_labels_final = np.pad(splice_labels_dense, (0, padding), constant_values=4) # Pad with not-a-splice-site label
        else:
            splice_labels_final = splice_labels_dense
        
        # Get organism index
        species_id = self.species.iloc[idx]
        # Handle both string species names and integer indices
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
            for _, row in sse_seq.iterrows():
                position = int(row['position'])
                condition_idx = int(row['condition_idx'])
                sse_value = row['sse']
                if 0 <= position < current_length:
                    sse[position, condition_idx] = sse_value
            # Crop/pad to target length
            if current_length > self.target_length:
                # Center crop
                sse_target = sse[crop_start:crop_start + self.target_length]
            elif current_length < self.target_length:
                # Pad with zeros
                padding = self.target_length - current_length
                sse_target = np.pad(sse, ((0, padding), (0, 0)), constant_values=0)
            else:
                sse_target = sse
        else:
            sse_target = np.zeros((self.target_length, 1), dtype=np.float32)
        
        # Get conditions mask for this sequence (per-sequence depending on organism)
        conditions_mask_array = self.condition_map.get(organism_name, [])

        return {
            'dna': torch.tensor(dna, dtype=torch.long),
            'organism_index': torch.tensor(organism_idx, dtype=torch.long),
            'splice_donor_idx': torch.tensor(donor_padded, dtype=torch.long),
            'splice_acceptor_idx': torch.tensor(acceptor_padded, dtype=torch.long),
            'num_donors': torch.tensor(num_donors, dtype=torch.long),
            'num_acceptors': torch.tensor(num_acceptors, dtype=torch.long),
            'splice_labels': torch.tensor(splice_labels_final, dtype=torch.long),
            'splice_usage_target': torch.tensor(sse_target, dtype=torch.float32),
            'conditions_mask': torch.tensor(conditions_mask_array, dtype=torch.long),  # (num_contexts,)
        }
