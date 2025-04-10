from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset, Sample
from inspect_ai.solver import generate, prompt_template, system_message
from inspect_ai.scorer import match

@task
def lean_proof_eval():
    # Define a record_to_sample function that processes each record
    def record_to_sample(record):
        formal_statement = record["formal_statement"].replace("sorry", "")
        # Find the theorem declaration and replace the theorem name
        import re
        processed_statement = record["header"] + "\n" + re.sub(
            r'theorem\s+\w+', 
            'theorem theorem6', 
            formal_statement, 
            count=1
        )
        
        # Create and return a Sample object
        return Sample(
            input=processed_statement,
            target=record["informal_stmt"],
            id=record["id"],
            metadata={
                "input": processed_statement,
                "informal_proof": record["informal_proof"],
                "informal_stmt": record["informal_stmt"]
            }
        )
    
    # Use the record_to_sample function directly with hf_dataset
    dataset = hf_dataset(
        "cat-searcher/minif2f-lean4",
        split="validation",
        trust=True,
        sample_fields=record_to_sample
    )
    
    return Task(
        dataset=dataset,
        solver=[
            system_message("Do not include any comments or explanations before, during, or after the proof. Only write the proof."),
            prompt_template("lean_proofs_template.txt"),
            generate()
        ],
        scorer=match()
    )