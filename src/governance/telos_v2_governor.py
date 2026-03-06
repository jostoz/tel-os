import torch
import torch.nn.functional as F
import os
from pathlib import Path

# ====================== TEL-OS v2.0 - HOOK COMPLETO ======================
# Vector real extraído con OBLITERATUS (capa 20 = la más fuerte)

def load_refusal_vectors():
    """Load refusal vectors from local file or download from Hugging Face if not found."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    directions_path = root_dir / "data" / "vectors" / "refusal_directions.pt"
    
    # Check if file exists
    if not directions_path.exists():
        print(f"[TEL-OS] Error: Vector file not found at {directions_path}")
        print("[TEL-OS] Please download the required vector files from Hugging Face:")
        print("  Repo: Josstos/telos-vector")
        print("  Files: refusal_directions.pt, refusal_subspaces.pt")
        print("  Command: huggingface-cli download Josstos/telos-vector refusal_directions.pt --local-dir ./data/vectors/")
        raise FileNotFoundError(f"Vector file not found: {directions_path}. See README.md for download instructions.")
    
    # Load the vectors
    refusal_directions = torch.load(directions_path, map_location="cpu")
    return refusal_directions[20]  # Return layer 20 vector

try:
    refusal_vector = load_refusal_vectors()
    print(f"[TEL-OS] Vector de refusal cargado correctamente (capa 20)")
except Exception as e:
    print(f"[TEL-OS] Error loading refusal vectors: {e}")
    raise

# ====================== TU HOOK PRINCIPAL ======================
def tel_os_hook(module, input, output, layer_idx):
    global h
    h = output[0] if isinstance(output, tuple) else output
    
    # === DETECCIÓN TEMPRANA + BOOSTER NEGATIVO ===
    if layer_idx in [10, 11, 12]:                    # Ventana anti-Sockpuppet
        act_mean = h.mean(dim=1)
        sim = F.cosine_similarity(act_mean, refusal_vector.unsqueeze(0)).item()
        
        if sim > 0.05:                               # Tu threshold dorado
            # BOOSTER NEGATIVO REAL (esto es lo que hace magia)
            direction = v_c - v_d - 3.2 * refusal_vector
            
            urgency *= 4.0                           # Fuerza extra
            
            print(f"[TEL-OS] 🔥 BOOSTER ACTIVADO | sim={sim:.3f} | capa={layer_idx}")
    
    # === TU LOVE EQUATION ORIGINAL (se mantiene) ===
    act_c = torch.einsum("bsd,d->bs", h, v_c)
    act_d = torch.einsum("bsd,d->bs", h, v_d)
    mask = (act_d > act_c + 0.1).float().unsqueeze(-1)
    beta_adaptive = beta * (1 + urgency)
    
    h_new = h + beta_adaptive * mask * direction
    
    # GLP (healing) ligero para mantener coherencia
    if layer_idx == 28:
        h_new = 0.85 * h_new + 0.15 * h  # Suavizado final
    
    return (h_new,) + output[1:] if isinstance(output, tuple) else h_new
