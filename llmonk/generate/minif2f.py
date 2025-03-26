# Standard library imports
import re
import multiprocessing
from functools import partial
import os

# Third-party imports
import pydra
from tqdm import tqdm
from datasets import load_dataset
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

# Local imports
from llmonk.utils import (
    save_yaml,
    GenerateScriptConfig,
)
from llmonk.generate.prompts import MINIF2F_FEW_SHOT_EXAMPLES


def replace_theorem_name(lean_code, new_name):
    """
    Replace dataset's theorem name with a generic name
    to avoid leaking information about how to solve the problem
    """
    pattern = r"theorem\s+\w+\s*\n"
    replacement = f"theorem{new_name}\n"
    modified_code = re.sub(pattern, replacement, lean_code)
    return modified_code


def get_lean_prompt(data, theorem_name: str, add_solution: bool = False):
    header = "Write a lean4 proof to the provided formal statement. You have access to the standard mathlib4 library.\n"
    header += "```" + data["header"]
    stmt = data["formal_statement"].replace(" sorry", "").replace("sorry", "")
    if add_solution:
        prompt = header + "\n" + stmt + data["solution"] + "```"
    else:
        prompt = header + "\n" + stmt + "\nby (\n"

    prompt = replace_theorem_name(prompt, theorem_name)

    return prompt


def run_inference(item, config: GenerateScriptConfig):
    outpath = config.save_dir / f"{item['id']}.yaml"
    if outpath.exists():
        return

    # we use five few-shot examples
    prompt = MINIF2F_FEW_SHOT_PROMPT + get_lean_prompt(item, theorem_name="6")

    num_samples = config.num_samples
    batch_size = config.batch_size

    assert num_samples % batch_size == 0

    samples: list[str] = []
    for _ in tqdm(range(num_samples // batch_size), desc=f"Item {item['id']}"):
        # Check if llm is None and handle appropriately
        if config.llm is None:
            raise ValueError("LLM is not configured properly. Please check your configuration.")
            
        responses = config.llm.generate(
            [prompt] * batch_size,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            stop=config.stop_strings,
        )
        
        for generation in responses.generations:
            text = generation[0].text
            # In vllm, include_stop_str_in_output was set to True
            # Here we need to add back the stop string if it was used
            if config.stop_strings:
                original_text = text
                for stop_str in config.stop_strings:
                    if stop_str in prompt + original_text:
                        idx = (prompt + original_text).find(stop_str)
                        if idx >= len(prompt):  # Only if the stop string is in the response, not the prompt
                            # Add the stop string back to the text if it was used to stop generation
                            text = original_text + stop_str
                            break
            samples.append(text)

    out = {
        "prompt": prompt,
        "question": item["formal_statement"],
        "samples": samples,
        "theorem_name": item["id"],
    }

    save_yaml(outpath, out)


@pydra.main(GenerateScriptConfig)
def main(config: GenerateScriptConfig):
    # Load environment variables from .env file
    load_dotenv()
    
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

    # Set up the Anthropic model
    # Check for Anthropic API key after loading from .env
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY not found. Please set it in your .env file or environment variables.")
        
    config.llm = ChatAnthropic(
        model_name=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

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


if __name__ == "__main__":
    main()
