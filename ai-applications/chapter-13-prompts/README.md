# บทที่ 13: Prompt Engineering

Chapter 13: Prompt Engineering for HPC

## วัตถุประสงค์การเรียนรู้

1. เข้าใจหลักการ Prompt Engineering
2. สร้าง Effective Prompts
3. ใช้ LLM APIs
4. ประยุกต์กับงาน HPC

## โครงสร้างไฟล์

```
chapter-13-prompts/
├── README.md
├── prompt_basics.py        # Prompt fundamentals
├── few_shot_learning.py    # Few-shot examples
├── chain_of_thought.py     # CoT prompting
├── code_generation.py      # Code generation prompts
└── hpc_assistant.py        # HPC-specific prompts
```

## การใช้งาน

```bash
# Install dependencies
pip install openai anthropic

# Set API key
export ANTHROPIC_API_KEY="your-key"

# Run examples
python prompt_basics.py
python chain_of_thought.py
```

## Prompt Structure

```
[System instruction]
[Context / Background]
[Task description]
[Input data]
[Output format specification]
[Examples (optional)]
```
