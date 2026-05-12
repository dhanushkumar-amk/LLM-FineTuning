# LLM-FineTuning

Fine-tuning Qwen2.5-0.5B using **QLoRA + DoRA** for a domain-specific healthcare assistant on consumer GPUs.

---

# 🚀 Project Overview

This project demonstrates how to build a specialized AI assistant using:

- Qwen2.5-0.5B-Instruct
- QLoRA (4-bit Quantized LoRA)
- DoRA Adapters
- Unsloth
- Hugging Face Transformers
- TRL SFTTrainer

The model was trained on ~500 healthcare instruction-response examples and optimized to run efficiently on small GPUs.

---

# 📌 Why This Project?

General-purpose LLMs are powerful, but they often struggle with domain-specific reasoning.

This project shows how lightweight fine-tuning techniques can dramatically improve model quality for specialized tasks without requiring massive infrastructure.

---

# 🧠 Technologies Used

- Python
- PyTorch
- Unsloth
- Hugging Face
- TRL
- QLoRA
- DoRA
- Qwen2.5

---

# ⚡ Features

- 4-bit QLoRA fine-tuning
- DoRA adapter training
- Small GPU compatible
- Healthcare domain adaptation
- Hugging Face Hub integration
- Before vs after model comparison
- Memory-efficient training

---

# 🏗️ Model Details

| Component | Value |
|---|---|
| Base Model | Qwen2.5-0.5B-Instruct |
| Fine-Tuning Method | QLoRA + DoRA |
| Dataset Size | ~500 Examples |
| Training Time | ~15 Minutes |
| Adapter Size | ~5MB |
| Precision | 4-bit |
| Framework | Unsloth |

---

# 📂 Project Structure

```bash
LLM-FineTuning/
│
├── FineTuning-Example.ipynb
├── healthcare_data.json
├── requirements.txt
├── README.md
```

---

# 📊 Dataset Format

```json
[
  {
    "instruction": "What are early symptoms of diabetes?",
    "output": "Common symptoms include increased thirst..."
  }
]
```

---

# 🔥 Fine-Tuning Techniques

## 1. LoRA (Low-Rank Adaptation)

LoRA freezes the original model and trains only small adapter matrices.

### Benefits
- Lower memory usage
- Faster training
- Tiny checkpoint sizes
- Near full fine-tuning quality

### Research Paper
https://arxiv.org/abs/2106.09685

---

## 2. QLoRA (Quantized LoRA)

QLoRA compresses the base model into 4-bit precision before training.

### Benefits
- Massive VRAM reduction
- Consumer GPU compatible
- Minimal quality loss

### Research Paper
https://arxiv.org/abs/2305.14314

---

## 3. DoRA (Weight-Decomposed LoRA)

DoRA improves LoRA by separately learning:
- Weight magnitude
- Weight direction

### Benefits
- Better stability
- Improved adaptation quality
- Closer to full fine-tuning performance

### Research Paper
https://arxiv.org/abs/2402.09353

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/dhanushkumar-amk/LLM-FineTuning.git
cd LLM-FineTuning
```

---

## Install Dependencies

```bash
pip install unsloth trl datasets transformers accelerate bitsandbytes
```

---

# 🚀 Training

Run the notebook:

```bash
FineTuning-Example.ipynb
```

The notebook includes:
- Dataset loading
- Dataset formatting
- QLoRA setup
- DoRA adapter configuration
- Training
- Inference
- Before vs after comparison
- Hugging Face upload

---

# 🧪 Example Training Configuration

```python
r = 16
lora_alpha = 32
lora_dropout = 0.05
load_in_4bit = True
use_dora = True
```

---

# 📈 Results

The fine-tuned model produced:
- More domain-aware responses
- Better healthcare terminology understanding
- Improved instruction-following
- More reliable outputs compared to the base model

---

# 🤗 Hugging Face Model

https://huggingface.co/Dhanushkumaramk/healthcare-qwen2.5-0.5B-dora

---

# 💡 Hardware Used

This project was trained on a small GPU using:
- 4-bit quantization
- Parameter-efficient fine-tuning
- Lightweight adapters

No enterprise hardware required.

---

# 📚 Learning Outcomes

This project demonstrates:
- Practical LLM fine-tuning
- Memory-efficient training
- Adapter-based optimization
- Domain adaptation
- Real-world AI engineering workflows

---

# 🔮 Future Improvements

- Larger healthcare dataset
- RAG integration
- Evaluation benchmarks
- Multi-domain fine-tuning
- Quantized inference deployment

---

# 👨‍💻 Author

Dhanush Kumar

### GitHub
https://github.com/dhanushkumar-amk

### Hugging Face
https://huggingface.co/Dhanushkumaramk

---

# ⭐ If You Found This Useful

Star the repository and share it with others learning AI engineering and LLM fine-tuning.
