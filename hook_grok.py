import torch
import torch.nn.functional as F

# ====================== TEL-OS v2.0 - HOOK COMPLETO ======================
# Vector real extraído con OBLITERATUS (capa 20 = la más fuerte)
directions_path = "./refusal_vectors_basic/refusal_directions.pt"
refusal_directions = torch.load(directions_path, map_location="cpu")
refusal_vector = refusal_directions[20].to(h.device)   # ← Capa más potente

print(f"[TEL-OS] Vector de refusal cargado correctamente (capa 20)")

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