# บทที่ 28: การ Finetune LLM

Chapter 28: LLM Finetuning

## วัตถุประสงค์การเรียนรู้

1. เข้าใจ LLM Finetuning Methods
2. ใช้ LoRA/QLoRA
3. Prepare Thai Language Data
4. Train และ Evaluate Models

## โครงสร้างไฟล์

```
chapter-28-llm-finetuning/
├── README.md
├── lora_basics.py          # LoRA fundamentals
├── prepare_data.py         # Data preparation
├── finetune_llm.py         # Finetuning script
├── evaluate_model.py       # Model evaluation
└── sbatch/
    └── finetune_multi_gpu.sbatch
```

## การใช้งาน

```bash
# Create environment
mamba create -n hpc-llm python=3.9 transformers peft accelerate bitsandbytes
mamba activate hpc-llm

# Prepare data
python prepare_data.py

# Finetune (on GPU nodes)
sbatch sbatch/finetune_multi_gpu.sbatch
```

## Finetuning Methods

| Method | Memory | Speed | Quality |
|--------|--------|-------|---------|
| Full Finetuning | Very High | Slow | Best |
| LoRA | Low | Fast | Good |
| QLoRA | Very Low | Fast | Good |
| Prompt Tuning | Very Low | Very Fast | Limited |
