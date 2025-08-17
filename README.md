# Gemma3 SigLIP Vision Encoder (extracted) 

This repository contains the **Python code** used to extract the SigLIP vision encoder 
from Google’s [Gemma 3](https://huggingface.co/google/gemma-3-4b-pt) vision-language model. 

The **extracted encoder weights and configs** are hosted separately on Hugging Face:  
[hi-wesley/gemma3-vision-encoder](https://huggingface.co/hi-wesley/gemma3-vision-encoder) 

---

**Important License Notice** 

The model weights in this repository are **derived** from the *Gemma 3* model
released by Google and are therefore **subject to Google’s Gemma Terms of Use
and Prohibited Use Policy**.

By downloading, using, or distributing these weights (in whole or in part),
you acknowledge and agree to:

- The **Gemma Model License Agreement**: <https://ai.google.dev/licenses/gemma_model_license>
- The **Prohibited Use Policy**: <https://ai.google.dev/licenses/gemma_prohibited_use>

These terms apply **in addition** to the Apache-2.0 license that covers the
extraction source code in this repository.  
If you do **not** agree to Google’s Gemma terms, **do not use these weights**.

---

## What this repository contains

| Item                                                 | Description                                          |
| ---------------------------------------------------- | ---------------------------------------------------- |
| `pytorch_model-*.safetensors`                        | SigLIP vision encoder weights extracted from Gemma 3 |
| `config.json`                                        | Model configuration for the encoder                  |
| `preprocessor_config.json` / `processor_config.json` | Image processor configs                              |
| `extract_gemma3_vision.py`                           | Script used to extract the encoder (Apache-2.0)      |

---

## Usage

```python
from transformers import SiglipVisionModel, AutoImageProcessor

model = SiglipVisionModel.from_pretrained("hi-wesley/gemma3-vision-encoder")
processor = AutoImageProcessor.from_pretrained("hi-wesley/gemma3-vision-encoder")
