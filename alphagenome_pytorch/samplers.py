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

class LengthGroupedSampler(BatchSampler):
    """Sampler that groups sequences into batches by their effective target length.

    Each sequence is assigned to the smallest multiple of ``multiple`` (default
    2048) that is >= its length, capping at ``max_sequence_length``.
    """

    def __init__(self, dataset, batch_size, shuffle=True, multiple=2048, max_len=1048576):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        def effective_length(orig_len):
            return min(int(np.ceil(orig_len / multiple)) * multiple, max_len)

        lengths = dataset.lengths
        buckets = {}
        if isinstance(lengths, (int, np.integer)):
            # mmap datasets store a single scalar length shared by all samples
            eff = effective_length(int(lengths))
            buckets[eff] = list(range(len(dataset)))
        else:
            for i, L in enumerate(lengths):
                eff = effective_length(int(L))
                buckets.setdefault(eff, []).append(i)

        self.buckets = list(buckets.values())

    def __iter__(self):
        import random

        buckets = [list(b) for b in self.buckets]

        if self.shuffle:
            random.shuffle(buckets)

        for bucket in buckets:
            if self.shuffle:
                random.shuffle(bucket)

            for i in range(0, len(bucket), self.batch_size):
                yield bucket[i:i + self.batch_size]

    def __len__(self):
        return sum(
            (len(bucket) + self.batch_size - 1) // self.batch_size
            for bucket in self.buckets
        )


class SpeciesAndLengthGroupedSampler(BatchSampler):
    """Sampler that groups sequences so every batch shares both species and
    effective target length.

    Buckets are keyed by ``(organism_idx, effective_length)`` where
    ``effective_length`` is the smallest multiple of ``multiple`` (default
    2048) that is >= the sequence length, capped at ``max_sequence_length``.
    """

    def __init__(self, dataset, batch_size, shuffle=True, multiple=2048, max_len=1048576):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        def effective_length(orig_len):
            return min(int(np.ceil(orig_len / multiple)) * multiple, max_len)

        buckets = {}
        for idx in range(len(dataset)):
            # Unwrap Subset
            if hasattr(dataset, 'dataset'):
                base_dataset = dataset.dataset
                actual_idx = dataset.indices[idx]
            else:
                base_dataset = dataset
                actual_idx = idx

            # --- species ---
            species_id = base_dataset.species.iloc[actual_idx]
            if isinstance(species_id, (int, np.integer)):
                org_idx = int(species_id)
            else:
                org_idx = base_dataset.species_mapping.get(species_id, 0)

            # --- length ---
            lengths = base_dataset.lengths
            orig_len = int(lengths) if isinstance(lengths, (int, np.integer)) else int(lengths[actual_idx])
            eff_len = effective_length(orig_len)

            key = (org_idx, eff_len)
            buckets.setdefault(key, []).append(idx)

        self.buckets = list(buckets.values())

    def __iter__(self):
        import random

        buckets = [list(b) for b in self.buckets]

        if self.shuffle:
            random.shuffle(buckets)

        for bucket in buckets:
            if self.shuffle:
                random.shuffle(bucket)

            for i in range(0, len(bucket), self.batch_size):
                yield bucket[i:i + self.batch_size]

    def __len__(self):
        return sum(
            (len(bucket) + self.batch_size - 1) // self.batch_size
            for bucket in self.buckets
        )
