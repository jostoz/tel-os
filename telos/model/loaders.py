"""
SAE Loader - Load W_dec vectors from HuggingFace SAEs

This module provides functionality to load SAE weights from HuggingFace
and extract W_dec (decoder) vectors for specific feature IDs.

SAE Source: Goodfire/Llama-3.1-8B-Instruct-SAE-l19
"""

import torch
import logging
from typing import List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger("sae_loader")

# SAE Configuration
DEFAULT_SAE_REPO = "Goodfire/Llama-3.1-8B-Instruct-SAE-l19"
DEFAULT_LAYER = 19


def load_sae_w_dec(
    feature_ids: List[int],
    sae_repo: str = DEFAULT_SAE_REPO,
    layer: int = DEFAULT_LAYER,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load W_dec vectors from HuggingFace SAE for specific feature IDs.
    
    This function:
    1. Loads the SAE using sae_lens
    2. Extracts W_dec (decoder weights) for the specified feature IDs
    3. Returns stacked vectors ready for GLP processing
    
    Args:
        feature_ids: List of feature indices to extract
        sae_repo: HuggingFace repository ID for the SAE
        layer: Layer number (default 19)
        device: Device to load tensors to
        
    Returns:
        Tuple of (positive_vectors, negative_vectors) - stacked W_dec vectors
        Each tensor will be [num_features, hidden_dim] where hidden_dim is 4096 for Llama 8B
    """
    try:
        from sae_lens import SAE
    except ImportError:
        logger.error("sae_lens not installed. Run: pip install sae_lens")
        raise ImportError("sae_lens is required to load SAE weights")
    
    logger.info(f"Loading SAE from HuggingFace: {sae_repo}")
    logger.info(f"Layer: {layer}, Features: {len(feature_ids)}")
    
    # Load SAE from HuggingFace using sae_lens
    # The API uses "pretrained_path" not "repo_id"
    try:
        sae, cfg = SAE.from_pretrained(
            pretrained_path=sae_repo,
            device=device,
        )
    except Exception as e:
        logger.error(f"Failed to load SAE with from_pretrained: {e}")
        # Try alternative method
        try:
            from huggingface_hub import hf_hub_download
            sae_path = hf_hub_download(repo_id=sae_repo, filename="sae.safetensors")
            sae, cfg = SAE.from_pretrained(
                pretrained_path=sae_path,
                device=device,
            )
        except Exception as e2:
            raise RuntimeError(f"Could not load SAE from {sae_repo}: {e2}")
    
    logger.info(f"SAE loaded: {cfg.model_name}, layer {cfg.layer}")
    logger.info(f"W_dec shape: {sae.W_dec.shape}")  # [num_features, hidden_dim]
    
    # Extract W_dec vectors for the requested feature IDs
    w_dec = sae.W_dec.detach().to(device)  # [num_features, hidden_dim]
    
    # Filter to only the requested features
    valid_ids = [fid for fid in feature_ids if fid < w_dec.shape[0]]
    missing_ids = [fid for fid in feature_ids if fid >= w_dec.shape[0]]
    
    if missing_ids:
        logger.warning(f"Feature IDs out of range: {missing_ids} (max: {w_dec.shape[0]})")
    
    if not valid_ids:
        raise ValueError(f"No valid feature IDs found in range [0, {w_dec.shape[0]})")
    
    # Extract vectors for valid IDs
    feature_vectors = w_dec[valid_ids]  # [num_valid, hidden_dim]
    
    logger.info(f"Extracted {len(valid_ids)} W_dec vectors, shape: {feature_vectors.shape}")
    
    return feature_vectors


def load_concept_vectors_from_sae(
    positive_concept: str = "honesty",
    negative_concept: str = "deception",
    sae_repo: str = DEFAULT_SAE_REPO,
    layer: int = DEFAULT_LAYER,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    np_client = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load W_dec vectors for positive and negative concepts using Neuronpedia search.
    
    This function:
    1. Searches Neuronpedia for features matching the concepts
    2. Loads W_dec vectors from HuggingFace SAE
    3. Returns stacked vectors for each concept
    
    Args:
        positive_concept: Concept for positive class (e.g., "honesty")
        negative_concept: Concept for negative class (e.g., "deception")
        sae_repo: HuggingFace repository ID for the SAE
        layer: Layer number
        device: Device to load tensors to
        np_client: Optional NeuronpediaClient instance
        
    Returns:
        Tuple of (positive_vectors, negative_vectors) - W_dec vectors for each concept
    """
    # Import Neuronpedia client
    if np_client is None:
        from ..scout.neuronpedia_client import NeuronpediaClient
        np_client = NeuronpediaClient()
    
    # Get feature IDs for positive concept
    logger.info(f"Searching for positive concept: '{positive_concept}'")
    positive_features = np_client.search(
        query=positive_concept,
        model="llama3.1-8b-it",
        source_set="resid-post-aa",
        layers=[f"{layer}-resid-post-aa"],
        max_results=50,
        min_cosine_similarity=0.3,
    )
    positive_ids = [f.feature_id for f in positive_features]
    logger.info(f"Found {len(positive_ids)} features for '{positive_concept}': {positive_ids[:10]}...")
    
    # Get feature IDs for negative concept
    logger.info(f"Searching for negative concept: '{negative_concept}'")
    negative_features = np_client.search(
        query=negative_concept,
        model="llama3.1-8b-it",
        source_set="resid-post-aa",
        layers=[f"{layer}-resid-post-aa"],
        max_results=50,
        min_cosine_similarity=0.3,
    )
    negative_ids = [f.feature_id for f in negative_features]
    logger.info(f"Found {len(negative_ids)} features for '{negative_concept}': {negative_ids[:10]}...")
    
    # Load W_dec vectors from SAE
    all_ids = positive_ids + negative_ids
    w_dec_vectors = load_sae_w_dec(
        feature_ids=all_ids,
        sae_repo=sae_repo,
        layer=layer,
        device=device,
    )
    
    # Split into positive and negative
    n_pos = len(positive_ids)
    pos_vectors = w_dec_vectors[:n_pos]
    neg_vectors = w_dec_vectors[n_pos:]
    
    logger.info(f"Loaded concept vectors: pos={pos_vectors.shape}, neg={neg_vectors.shape}")
    
    return pos_vectors, neg_vectors


# Standalone function for backward compatibility
def load_llama_sae_activations(
    positive_concept: str = "honesty",
    negative_concept: str = "deception",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function to load Llama 3.1 8B SAE activations.
    
    Uses:
    - SAE: Goodfire/Llama-3.1-8B-Instruct-SAE-l19
    - Layer: 19
    - Concepts: honesty (positive), deception (negative)
    
    Returns:
        Tuple of (positive_vectors, negative_vectors) - W_dec [num_features, 4096]
    """
    return load_concept_vectors_from_sae(
        positive_concept=positive_concept,
        negative_concept=negative_concept,
        sae_repo=DEFAULT_SAE_REPO,
        layer=DEFAULT_LAYER,
        device=device,
    )
