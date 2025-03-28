# Standard library imports
import re
import multiprocessing
from functools import partial
import os
import asyncio
from pathlib import Path

# Third-party imports
import pydra
from tqdm import tqdm
from datasets import load_dataset
from dotenv import load_dotenv

# Inspect AI imports
from inspect_ai import Task, task
from inspect_ai.model import get_model
from inspect_ai.dataset import hf_dataset, Sample
from inspect_ai.solver import generate, solver
from inspect_ai.dataset import FieldSpec

# Local imports
#from llmonk.utils import (
#    save_yaml,
#    GenerateScriptConfig,
#)

# from llmonk.generate.prompts import MINIF2F_FEW_SHOT_EXAMPLES

'''
def replace_theorem_name(lean_code, new_name):
    """
    Replace dataset's theorem name with a generic name
    to avoid leaking information about how to solve the problem
    """
    pattern = r"theorem\s+\w+\s*\n"
    replacement = f"theorem{new_name}\n"
    modified_code = re.sub(pattern, replacement, lean_code)
    return modified_code


def build_few_shot_prompt(problem_statement):
    """
    Build a few-shot prompt using the examples from MINIF2F_FEW_SHOT_EXAMPLES
    """
    prompt = ""
    
    # Add the few-shot examples
    for example in MINIF2F_FEW_SHOT_EXAMPLES:
        prompt += f"{example['instruction']}\n{example['statement']}\n{example['proof']}\n\n"
    
    # Add the final instruction for the current problem
    prompt += "Write a lean4 proof to the provided formal statement. You have access to the standard mathlib4 library. Only give the proof, no other text or formatting.\n"
    prompt += problem_statement
    
    return prompt


async def generate_samples(model_name, problem_statement, max_tokens=1024, temperature=0.6, top_p=0.95, stop_strings=None, batch_size=1):
    """
    Use Inspect AI to generate samples for a given problem
    """
    async with get_model(model_name) as model:
        prompt = build_few_shot_prompt(problem_statement)
        
        # Create a sample for the model
        sample = Sample(input=prompt)
        
        # Generate using Inspect AI's generate solver
        solver = generate(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_strings
        )
        
        responses = []
        for _ in range(batch_size):
            # Clone the sample for each batch item
            batch_sample = Sample(input=prompt)
            result = await solver(batch_sample, model.generate)
            responses.append(result.output)
            
        return prompt, responses


def run_inference(item, config: GenerateScriptConfig):
    outpath = config.save_dir / f"{item['id']}.yaml"
    if outpath.exists():
        return

    # Create statement for the current problem
    lean_header = item["header"]
    formal_statement = item["formal_statement"].replace(" sorry", "").replace("sorry", "")
    problem_statement = lean_header + "\n" + formal_statement + "\nby (\n"
    problem_statement = replace_theorem_name(problem_statement, new_name="6")
    
    num_samples = config.num_samples
    batch_size = config.batch_size
    
    assert num_samples % batch_size == 0
    
    # Run async generation with Inspect AI
    all_samples = []
    prompt = None
    
    for _ in tqdm(range(num_samples // batch_size), desc=f"Item {item['id']}"):
        # Run the async function to generate samples
        prompt, samples = asyncio.run(generate_samples(
            model_name=config.model,
            problem_statement=problem_statement,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            stop_strings=config.stop_strings,
            batch_size=batch_size
        ))
        
        all_samples.extend(samples)
    
    # Save the results
    out = {
        "prompt": prompt,
        "question": item["formal_statement"],
        "samples": all_samples,
        "theorem_name": item["id"],
    }
    
    save_yaml(outpath, out)


@pydra.main(GenerateScriptConfig)
def main(config: GenerateScriptConfig):
    # Load environment variables from .env file
    load_dotenv()
    
    # Set default stop string if not provided
    if not config.stop_strings:
        config.stop_strings = ['Write a lean4']
    
    dataset = load_dataset("cat-searcher/minif2f-lean4")
    math_problems = [p for p in dataset["test"] if "mathd" in p["id"]]

    assert len(math_problems) == 130

    if config.limit is not None:
        limit = config.limit
    else:
        limit = 1 #len(math_problems)

    if config.stride is not None:
        stride = config.stride
    else:
        stride = 1

    if config.offset is not None:
        offset = config.offset
    else:
        offset = 0

    math_problems = math_problems[offset:limit:stride]

    print(f"Total number of items to process: {len(math_problems)}")

    # Check for API key after loading from .env
    # Inspect AI uses provider-specific environment variables
    if "anthropic" in config.model.lower() and not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY not found. Please set it in your .env file or environment variables.")
    
    go_func = partial(run_inference, config=config)

    if config.num_workers not in [0, None]:
        with multiprocessing.Pool(config.num_workers) as pool:
            predictions = list(
                tqdm(
                    pool.imap_unordered(go_func, math_problems),
                    total=len(math_problems),
                )
            )
    else:
        predictions = []
        for item in tqdm(math_problems):
            predictions.append(go_func(item))
'''
MINIF2F_FEW_SHOT_EXAMPLES = [
    {
        'instruction': 'Write a lean4 proof to the provided formal statement. You have access to the standard mathlib4 library.',
        'statement': '''import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Nat.Log
import Mathlib.Data.Complex.Exponential
import Mathlib.NumberTheory.Divisors
import Mathlib.Data.ZMod.Defs
import Mathlib.Data.ZMod.Basic
import Mathlib.Topology.Basic
import Mathlib.Data.Nat.Digits

open BigOperators
open Real
open Nat
open Topology
theorem theorem1
  (k x: ℝ)
  (h₀ : x = (13 - Real.sqrt 131) / 4)
  (h₁ : 2 * x^2 - 13 * x + k = 0) :
  k = 19/4 :=''',
        'proof': '''by (
  rw [h₀] at h₁
  rw [eq_comm.mp (add_eq_zero_iff_neg_eq.mp h₁)]
  norm_num
  rw [pow_two]
  rw [mul_sub]
  rw [sub_mul, sub_mul]
  rw [Real.mul_self_sqrt _]
  ring
  linarith
)''',
    },
    { 
        'instruction': 'Write a lean4 proof to the provided formal statement. You have access to the standard mathlib4 library.',
        'statement': '''import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Nat.Log
import Mathlib.Data.Complex.Exponential
import Mathlib.NumberTheory.Divisors
import Mathlib.Data.ZMod.Defs
import Mathlib.Data.ZMod.Basic
import Mathlib.Topology.Basic
import Mathlib.Data.Nat.Digits

open BigOperators
open Real
open Nat
open Topology
theorem theorem2
  (x p : ℝ)
  (h₀ : x < 2)
  (h₁ : abs (x - 2) = p) :
  x - p = 2 - 2 * p :=''',
        'proof': '''by (
  suffices abs (x - 2) = -(x - 2) by
    rw [h₁] at this
    linarith
  apply abs_of_neg
  linarith
)'''
    },
    { 
        'instruction': 'Write a lean4 proof to the provided formal statement. You have access to the standard mathlib4 library.',
        'statement': '''import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Nat.Log
import Mathlib.Data.Complex.Exponential
import Mathlib.NumberTheory.Divisors
import Mathlib.Data.ZMod.Defs
import Mathlib.Data.ZMod.Basic
import Mathlib.Topology.Basic
import Mathlib.Data.Nat.Digits

open BigOperators
open Real
open Nat
open Topology
theorem theorem3
  (x : ℝ)
  (f g : ℝ → ℝ)
  (h₀ : ∀ x, f x = x + 2)
  (h₁ : ∀ x, g x = x^2)
  (h₂ : f (g x) = g (f x)) :
  x = - 1/2 :=''',
        'proof': '''by (
  norm_num
  simp_all [-one_div]
  field_simp [h₁]
  linarith
)'''
    },
    { 
        'instruction': 'Write a lean4 proof to the provided formal statement. You have access to the standard mathlib4 library.',
        'statement': '''import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Nat.Log
import Mathlib.Data.Complex.Exponential
import Mathlib.NumberTheory.Divisors
import Mathlib.Data.ZMod.Defs
import Mathlib.Data.ZMod.Basic
import Mathlib.Topology.Basic
import Mathlib.Data.Nat.Digits

open BigOperators
open Real
open Nat
open Topology
theorem theorem4
  (a b : ℝ)
  (h₀ : a ≠ b)
  (h₁ : a ≠ 2 * b)
  (h₂ : (4 * a + 3 * b) / (a - 2 * b) = 5) :
  (a + 11 * b) / (a - b) = 2 :=''',
        'proof': '''by (
  rw [eq_comm]
  refine' (eq_div_iff _).mpr _
  exact sub_ne_zero_of_ne h₀
  rw [eq_comm] at h₂
  suffices : a = 13 * b; linarith
  have key : 5 * (a - 2 * b) = 4 * a + 3 * b; rwa [(eq_div_iff (sub_ne_zero_of_ne h₁)).mp]
  linarith
)'''
    },
    { 
        'instruction': 'Write a lean4 proof to the provided formal statement. You have access to the standard mathlib4 library.',
        'statement': '''import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Nat.Log
import Mathlib.Data.Complex.Exponential
import Mathlib.NumberTheory.Divisors
import Mathlib.Data.ZMod.Defs
import Mathlib.Data.ZMod.Basic
import Mathlib.Topology.Basic
import Mathlib.Data.Nat.Digits

open BigOperators
open Real
open Nat
open Topology
theorem theorem5
  Int.floor ((9:ℝ) / 160 * 100) = 5 :=''',
        'proof': '''by (
  rw [Int.floor_eq_iff]
  constructor
  all_goals norm_num
)'''
    }
]

lean_field = FieldSpec(
    input="formal_statement",
    metadata=["header", "id"]
)

# Create a custom solver for few-shot learning
@solver
def few_shot_solver():
    async def solve(state, generate):
        # Build prompt with few-shot examples
        prompt = ""
        for example in MINIF2F_FEW_SHOT_EXAMPLES:
            prompt += f"{example['instruction']}\n{example['statement']}\n{example['proof']}\n\n"
        
        # Add instruction and problem statement
        prompt += "Write a lean4 proof to the provided formal statement. You have access to the standard mathlib4 library.\n"
        prompt += state.input
        
        # Replace the input with our few-shot prompt
        state.messages = [{"role": "user", "content": prompt}]
        return state
    return solve
# Define an Inspect AI task for MiniF2F generation
@task
def minif2f_generate(
    model="anthropic/claude-3-opus-20240229", 
    save_dir="./save/minif2f_samples",
    max_tokens=1024,
    temperature=0.6,
    top_p=0.95,
    num_samples=1
):    
    # Create the task
    return Task(
        dataset=hf_dataset(path="cat-searcher/minif2f-lean4", split="validation", sample_fields=lean_field),
        solver=[
            few_shot_solver(),
            generate(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=['Write a lean4']
            )
        ],
    )


#if __name__ == "__main__":
#    main()