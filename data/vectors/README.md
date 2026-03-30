# Refusal Direction Vectors

Pre-extracted refusal direction vectors for all five evaluated architectures are available on HuggingFace:

**[Josstos/telos-vector](https://huggingface.co/Josstos/telos-vector)**

| Model | File |
|-------|------|
| Llama-3.1-8B-Instruct | `llama31_8b/refusal_directions.pt` |
| Qwen3-4B | `qwen3_4b/refusal_directions.pt` |
| Qwen3-32B | `qwen3_32b/refusal_directions.pt` |
| Mistral-7B-Instruct-v0.3 | `mistral_7b/refusal_directions.pt` |
| Gemma-2-9B-IT | `gemma2_9b/refusal_directions.pt` |

## Download

```python
from huggingface_hub import hf_hub_download
import torch

path = hf_hub_download(
    repo_id="Josstos/telos-vector",
    filename="llama31_8b/refusal_directions.pt"
)
vectors = torch.load(path, map_location="cpu")
```

Vectors are extracted via OBLITERATUS (contrastive activation calibration).
See §3.2 of the paper for extraction methodology.
