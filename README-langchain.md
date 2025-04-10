# Anthropic API Implementation for MiniF2F

This implementation replaces the vLLM server with an API-based inference system for the MiniF2F dataset.

## Setup

1. Install the required dependencies:

```bash
conda activate llmonk-minif2f
pip install -r requirements_minif2f.txt
```

2. Set up your Anthropic API key:

   Option 1: Create a `.env` file (recommended):
   ```bash
   # Copy the example .env file
   cp .env.example .env
   
   # Edit the .env file with your API key
   nano .env  # or any text editor
   ```

   Option 2: Set as an environment variable:
   ```bash
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

## Usage - Traditional Mode

Run the script with the model parameter pointing to your desired Anthropic model:

```bash
conda activate llmonk-minif2f && python -m llmonk.generate.minif2f \
    --model="claude-3-sonnet-20240229" \
    --save_dir="output/claude-3-sonnet-minif2f" \
    --num_samples=10 \
    --batch_size=5 \
    --max_tokens=1024 \
    --temperature=0.7
```

## Usage - Inspect AI Mode

The script now also supports running as an Inspect AI task, which provides more robust evaluation capabilities:

```bash
conda activate llmonk-minif2f && inspect eval llmonk/generate/minif2f.py:minif2f_generate \
    --model="anthropic/claude-3-sonnet-20240229" \
    -T save_dir="output/claude-3-sonnet-minif2f" \
    -T max_tokens=1024 \
    -T temperature=0.7 \
    -T num_samples=10
```

## Supported Models

You can use any Anthropic model, including:

- `claude-3-sonnet-20240229`
- `claude-3-opus-20240229`
- `claude-3-haiku-20240307` (o3-mini)
- `claude-3.5-sonnet-20240620`
- `claude-3.7-sonnet-20240612`

When using Inspect AI mode, prefix the model name with the provider: `anthropic/claude-3-sonnet-20240229`

## Implementation Options

The codebase now offers two different implementation approaches:

### 1. Traditional Pydra-Based Implementation 
- Uses multiprocessing for parallelization
- Compatible with the original script interface
- Uses Inspect AI under the hood for model generation
- Maintains the existing configuration schema

### 2. Inspect AI Task-Based Implementation
- Fully integrated with Inspect AI's task framework
- Provides more detailed logging and evaluation metrics
- Can be run using Inspect's CLI tools
- Offers more extensibility for creating evaluation pipelines

## Configuration

Key parameters:
- `model`: Model name (with provider prefix in Inspect AI mode)
- `save_dir`: Directory to save the results
- `num_samples`: Total number of samples to generate
- `batch_size`: Number of samples to generate in parallel (traditional mode)
- `max_tokens`: Maximum number of tokens to generate
- `temperature`: Sampling temperature
- `top_p`: Top-p sampling parameter