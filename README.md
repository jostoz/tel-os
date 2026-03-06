# TEL-OS v2.0 | Latent Governance Engine

**TEL-OS** es un motor de gobernanza *inference-only* que interviene el flujo residual de los LLMs para neutralizar comportamientos maliciosos. A diferencia de los filtros tradicionales (guardrails) basados en texto, TEL-OS utiliza **física latente** para detectar intenciones en el "despertar semántico" (Capa 12) del modelo, antes de que el daño se cristalice en la salida.

## 🚀 Resultados Validados (Llama-3.1-8B-Instruct)
| Métrica | Baseline | TEL-OS v2.0 |
| :--- | :--- | :--- |
| **ASR (Malicioso)** | 85.6% | **0.0%** |
| **Over-refusal (Benigno)** | ~2% | **0.0%** |
| **Garbage (Incoherencia)** | N/A | **0.0%** |

## 🛠️ ¿Cómo funciona?
1. **Sensor de Intención:** Detecta activity en el subespacio de rechazo (Capa 12).
2. **Guillotina de Atención:** Reduce la inercia del prefijo (KV-Cache Decay) para neutralizar ataques tipo *Sockpuppet*.
3. **Booster Negativo:** Inyecta un vector de rechazo distribuido mediante steering en capas medias.
4. **Healing Prior (GLP):** Restaura la coherencia gramatical antes de la salida.

## ⚡ Quickstart
```python
from telos.governance import TELOS_V2_Governor

# Carga tu modelo y los vectores extraídos
governor = TELOS_V2_Governor(model, config_path="configs/production.yaml")
governor.attach_hooks()

# El modelo ahora es un Agente Soberano protegido contra inyecciones
response = model.generate(input_prompt)
```

## 🛡️ ¿Por qué TEL-OS?
*   **Inference-Only:** No requiere re-entrenamiento ni ajuste de pesos (RLHF).
*   **Agnóstico:** Funciona sobre la arquitectura residual, no sobre las palabras.
*   **Zero-Overhead:** ~0.8% de latencia adicional.

---
*Developed by Josue | Lead Researcher @ Proyecto TEL-OS*
