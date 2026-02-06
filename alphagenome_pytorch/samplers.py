"""Custom data sampler"""

import numpy as np
from torch.utils.data import BatchSampler

# Custom sampler for species-specific batches
class SpeciesGroupedSampler(BatchSampler):
    """Sampler that ensures each batch contains sequences from only one species."""
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group indices by species
        self.species_indices = {}
        for idx in range(len(dataset)):
            if hasattr(dataset, 'dataset'):  # Handle Subset
                base_dataset = dataset.dataset
                actual_idx = dataset.indices[idx]
            else:
                base_dataset = dataset
                actual_idx = idx
            
            species_id = base_dataset.species.iloc[actual_idx]
            if isinstance(species_id, (int, np.integer)):
                org_idx = int(species_id)
            else:
                org_idx = base_dataset.species_mapping.get(species_id, 0)
            
            if org_idx not in self.species_indices:
                self.species_indices[org_idx] = []
            self.species_indices[org_idx].append(idx)
    
    def __iter__(self):
        # Create batches for each species
        batches = []
        for org_idx, indices in self.species_indices.items():
            org_indices = indices.copy()
            if self.shuffle:
                np.random.shuffle(org_indices)
            
            # Split into batches
            for i in range(0, len(org_indices), self.batch_size):
                batch = org_indices[i:i + self.batch_size]
                batches.append(batch)
        
        # Shuffle batch order if requested
        if self.shuffle:
            np.random.shuffle(batches)
        
        for batch in batches:
            yield batch
    
    def __len__(self):
        total_batches = sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.species_indices.values()
        )
        return total_batches