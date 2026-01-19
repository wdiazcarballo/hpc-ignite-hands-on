#!/usr/bin/env python3
"""
Prompt Basics - พื้นฐาน Prompt Engineering
Chapter 13: Prompt Engineering for HPC

หลักการสร้าง prompt ที่มีประสิทธิภาพ
"""


def explain_prompt_structure():
    """อธิบายโครงสร้าง Prompt"""
    print("\n1. Prompt Structure:")

    structure = """
    ┌─────────────────────────────────────────┐
    │  System Instruction (Role & Context)    │
    ├─────────────────────────────────────────┤
    │  Background / Domain Knowledge          │
    ├─────────────────────────────────────────┤
    │  Task Description                       │
    ├─────────────────────────────────────────┤
    │  Input Data                             │
    ├─────────────────────────────────────────┤
    │  Output Format Specification            │
    ├─────────────────────────────────────────┤
    │  Examples (Few-shot)                    │
    └─────────────────────────────────────────┘
    """
    print(structure)


def show_prompt_techniques():
    """แสดง Prompt Techniques"""
    print("\n2. Prompt Techniques:")

    techniques = {
        'Zero-shot': 'Direct instruction without examples',
        'Few-shot': 'Provide examples before the task',
        'Chain-of-Thought': 'Ask to show reasoning steps',
        'Role-playing': 'Assign a specific role/persona',
        'Structured Output': 'Request specific format (JSON, CSV)',
    }

    print(f"\n   {'Technique':<20} {'Description':<40}")
    print("   " + "-" * 60)
    for tech, desc in techniques.items():
        print(f"   {tech:<20} {desc:<40}")


def example_zero_shot():
    """ตัวอย่าง Zero-shot Prompt"""
    print("\n3. Zero-shot Example:")

    prompt = """
    Analyze this SLURM job output and identify any issues:

    Job ID: 12345
    State: FAILED
    Exit Code: 137
    Memory Usage: 128GB / 64GB
    Runtime: 2:45:30

    Provide a brief analysis and recommendation.
    """
    print(prompt)

    expected_response = """
    Analysis:
    - Exit code 137 indicates OOM (Out of Memory) kill
    - Memory usage (128GB) exceeded allocation (64GB)

    Recommendation:
    - Increase memory allocation: #SBATCH --mem=150G
    - Or optimize memory usage in your code
    """
    print("   Expected Response:" + expected_response)


def example_few_shot():
    """ตัวอย่าง Few-shot Prompt"""
    print("\n4. Few-shot Example:")

    prompt = '''
    Convert SLURM directives to sbatch commands.

    Example 1:
    Input: Request 4 GPUs
    Output: #SBATCH --gpus=4

    Example 2:
    Input: Set job name to "training"
    Output: #SBATCH --job-name=training

    Example 3:
    Input: Request 64GB memory
    Output: #SBATCH --mem=64G

    Now convert:
    Input: Request 2 nodes with 8 tasks each
    Output:
    '''
    print(prompt)

    expected = """
    #SBATCH --nodes=2
    #SBATCH --ntasks-per-node=8
    """
    print("   Expected:" + expected)


def example_chain_of_thought():
    """ตัวอย่าง Chain-of-Thought Prompt"""
    print("\n5. Chain-of-Thought Example:")

    prompt = """
    Calculate the optimal batch size for training on LANTA.

    Given:
    - Model: ResNet50 (~25M parameters)
    - GPU: A100 40GB
    - Input size: 224x224x3
    - FP16 training

    Think step by step:
    1. Estimate model memory footprint
    2. Estimate per-sample memory
    3. Account for gradients and optimizer states
    4. Calculate maximum batch size
    5. Recommend practical batch size
    """
    print(prompt)


def example_structured_output():
    """ตัวอย่าง Structured Output Prompt"""
    print("\n6. Structured Output Example:")

    prompt = '''
    Analyze this HPC job configuration and return JSON:

    #SBATCH --job-name=ml-train
    #SBATCH --nodes=2
    #SBATCH --ntasks-per-node=4
    #SBATCH --gpus-per-node=4
    #SBATCH --time=24:00:00
    #SBATCH --mem=256G

    Return a JSON object with:
    {
        "job_name": string,
        "total_gpus": number,
        "total_tasks": number,
        "memory_per_node_gb": number,
        "estimated_cost_compute_units": number,
        "recommendations": [string]
    }
    '''
    print(prompt)


def example_hpc_specific():
    """ตัวอย่าง HPC-specific Prompts"""
    print("\n7. HPC-Specific Prompts:")

    prompts = {
        'Debug': '''
        You are an HPC debugging assistant. Analyze this error:

        srun: error: Node compute-001: task 5: Out Of Memory
        srun: Terminating job step 12345.0

        Explain the error and provide solutions.
        ''',

        'Optimize': '''
        Review this SLURM script for optimization:

        #!/bin/bash
        #SBATCH -N 1
        #SBATCH --ntasks=1
        #SBATCH --mem=500G
        #SBATCH -t 72:00:00

        python train.py

        Suggest improvements for efficiency and resource usage.
        ''',

        'Translate': '''
        Convert this PBS script to SLURM:

        #PBS -l nodes=2:ppn=16
        #PBS -l walltime=24:00:00
        #PBS -q gpu

        Provide the equivalent SLURM directives.
        '''
    }

    for category, prompt in prompts.items():
        print(f"\n   {category} Prompt:")
        print(f"   {'='*50}")
        for line in prompt.strip().split('\n'):
            print(f"   {line}")


def best_practices():
    """Best Practices"""
    print("\n8. Prompt Engineering Best Practices:")

    practices = [
        "Be specific and clear about the task",
        "Provide relevant context and constraints",
        "Use structured formats when possible",
        "Include examples for complex tasks",
        "Specify the desired output format",
        "Break complex tasks into steps",
        "Test and iterate on prompts",
        "Consider edge cases in instructions",
    ]

    for i, practice in enumerate(practices, 1):
        print(f"   {i}. {practice}")


def main():
    print("=" * 60)
    print("   Prompt Basics")
    print("   Chapter 13: Prompt Engineering for HPC")
    print("=" * 60)

    explain_prompt_structure()
    show_prompt_techniques()
    example_zero_shot()
    example_few_shot()
    example_chain_of_thought()
    example_structured_output()
    example_hpc_specific()
    best_practices()

    print("\n" + "=" * 60)
    print("   Prompt basics complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
