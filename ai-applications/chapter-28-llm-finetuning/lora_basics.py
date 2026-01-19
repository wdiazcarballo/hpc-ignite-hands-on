#!/usr/bin/env python3
"""
LoRA Basics - พื้นฐาน LoRA Finetuning
Chapter 28: LLM Finetuning

LoRA = Low-Rank Adaptation of Large Language Models
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed")


def explain_lora_concept():
    """อธิบายแนวคิด LoRA"""
    print("\n1. LoRA Concept:")
    print("   Instead of updating W directly, we learn:")
    print("   W' = W + BA")
    print("   ")
    print("   Where:")
    print("   - W: Original weights (frozen)")
    print("   - B: Low-rank matrix (d x r)")
    print("   - A: Low-rank matrix (r x k)")
    print("   - r: Rank (much smaller than d and k)")
    print("   ")
    print("   Benefits:")
    print("   - Much fewer parameters to train")
    print("   - Original weights preserved")
    print("   - Can merge adapters for inference")


def demonstrate_parameter_reduction():
    """แสดงการลดพารามิเตอร์"""
    print("\n2. Parameter Reduction:")

    # Example dimensions (like a transformer layer)
    d = 4096  # Hidden size
    k = 4096  # Output size
    r = 8     # LoRA rank

    full_params = d * k
    lora_params = d * r + r * k

    print(f"   Original layer size: {d} x {k}")
    print(f"   Full finetuning params: {full_params:,}")
    print(f"   LoRA rank: {r}")
    print(f"   LoRA params: {lora_params:,}")
    print(f"   Reduction: {full_params / lora_params:.0f}x fewer parameters!")


def lora_forward_demo():
    """สาธิต LoRA forward pass"""
    print("\n3. LoRA Forward Pass Demo:")

    if not TORCH_AVAILABLE:
        print("   [Requires PyTorch]")
        return

    # Simulated linear layer
    d, k = 512, 512  # Input/output dimensions
    r = 8  # LoRA rank

    # Original weight (frozen)
    W = torch.randn(k, d) * 0.01

    # LoRA adapters
    A = torch.randn(r, d) * 0.01  # Down projection
    B = torch.zeros(k, r)  # Up projection (initialized to zero)

    # Input
    x = torch.randn(1, d)

    # Forward pass
    # Original output
    original_out = x @ W.T

    # LoRA output: W'x = Wx + BAx
    lora_delta = (x @ A.T) @ B.T
    adapted_out = original_out + lora_delta

    print(f"   Input shape: {x.shape}")
    print(f"   Original output shape: {original_out.shape}")
    print(f"   LoRA delta shape: {lora_delta.shape}")
    print(f"   Final output shape: {adapted_out.shape}")
    print(f"\n   Note: At initialization, B=0, so LoRA delta is zero")
    print(f"   This preserves original model behavior at start of training")


def show_lora_config():
    """แสดงตัวอย่าง LoRA config"""
    print("\n4. Typical LoRA Configuration:")

    config = {
        'r': 8,  # Rank
        'lora_alpha': 16,  # Scaling factor
        'lora_dropout': 0.05,  # Dropout
        'target_modules': ['q_proj', 'v_proj'],  # Which layers to adapt
        'bias': 'none',  # Don't train biases
        'task_type': 'CAUSAL_LM'  # Task type
    }

    print("   LoRAConfig(")
    for key, value in config.items():
        print(f"       {key}={repr(value)},")
    print("   )")

    print("\n   Explanation:")
    print("   - r: Lower = fewer params, higher = more capacity")
    print("   - lora_alpha: Scaling factor, often 2*r")
    print("   - target_modules: Usually attention layers")


def estimate_memory():
    """ประมาณการใช้หน่วยความจำ"""
    print("\n5. Memory Estimation (7B Model):")

    model_params = 7e9  # 7 billion
    bytes_per_param_fp16 = 2
    bytes_per_param_4bit = 0.5

    base_memory_fp16 = model_params * bytes_per_param_fp16 / 1e9
    base_memory_4bit = model_params * bytes_per_param_4bit / 1e9

    print(f"\n   Base Model (7B params):")
    print(f"   - FP16: ~{base_memory_fp16:.0f} GB")
    print(f"   - 4-bit: ~{base_memory_4bit:.0f} GB")

    # LoRA additional
    num_layers = 32
    hidden_size = 4096
    r = 8
    lora_params = 2 * num_layers * (hidden_size * r * 2)  # q and v, up and down
    lora_memory = lora_params * 4 / 1e6  # FP32 for training

    print(f"\n   LoRA Adapters (r={r}):")
    print(f"   - Trainable params: ~{lora_params/1e6:.1f}M")
    print(f"   - Memory: ~{lora_memory:.0f} MB")

    print(f"\n   Full Finetuning: ~{base_memory_fp16 * 4:.0f} GB (weights + gradients + optimizer)")
    print(f"   QLoRA (4-bit): ~{base_memory_4bit + lora_memory/1000:.1f} GB")


def show_training_example():
    """แสดงตัวอย่างการ train"""
    print("\n6. Training Example (Pseudocode):")

    code = '''
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        load_in_4bit=True,  # QLoRA
        device_map="auto"
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )

    # Get PEFT model
    model = get_peft_model(model, lora_config)

    # Train
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        args=TrainingArguments(
            output_dir="./lora-adapter",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
        )
    )
    trainer.train()

    # Save adapter only
    model.save_pretrained("./lora-adapter")
    '''

    print(code)


def main():
    print("=" * 60)
    print("   LoRA Basics - Low-Rank Adaptation")
    print("   Chapter 28: LLM Finetuning")
    print("=" * 60)

    explain_lora_concept()
    demonstrate_parameter_reduction()
    lora_forward_demo()
    show_lora_config()
    estimate_memory()
    show_training_example()

    print("\n" + "=" * 60)
    print("   LoRA basics complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
