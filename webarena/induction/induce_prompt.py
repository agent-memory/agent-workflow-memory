import os
import json
import random
import argparse

import openai
from openai import OpenAI
from huggingface_hub import InferenceClient

from utils import extract_think_and_action, get_abstract_actions

def format_trajectory(think_list: list[str], action_list: list[list[str]]) -> str:
    trajectory = []
    for t, a in zip(think_list, action_list):
        acts = '\n'.join(a)
        trajectory.append(f"<think>\n{t}\n</think>\n<action>\n{acts}\n</action>")
    return '\n\n'.join(trajectory)

def random_group_sample(d: dict, n) -> list:
    """Randomly sample n groups from the dictionary."""
    return [ex for v in d.values() for ex in random.sample(v, min(n, len(v)))]

# %% prompt model
def format_examples(examples: list[dict]) -> str:
    """Format examples to the prompt."""
    formatted_examples = []
    for ex in examples:
        trajectory = format_trajectory(ex["think_list"], ex["action_list"])
        formatted_examples.append(f"Query: {ex['query']}\nActions:\n{trajectory}")
    return '\n\n'.join(["## Concrete Examples"] + formatted_examples + ["## Summary Workflows"])

def llm_generate(llm_client, examples: list[dict], args, verbose: bool = False):
    """Call gpt model to generate workflows."""
    prompt = format_examples(examples)
    prompt = '\n\n'.join([args.INSTRUCTION, args.ONE_SHOT, prompt])
    if verbose: print("Prompt:\n", prompt, '\n\n')

    # HFInfClient Alias for OpenAI format
    response = llm_client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0 if "gpt" in args.model else 0.6,
        max_tokens=2048,
    )

    response = response.choices[0].message.content
    if verbose: print(response)
    return response


def main():
    # collect result directories, e.g., ["results/webarena.0", ...]
    args.result_dir = args.result_dir.split()
    if args.criteria == "gt":
        file_dirs = [
            os.path.join(res_dir, f) for res_dir in args.result_dir for f in os.listdir(res_dir) 
            if json.load(
                open(os.path.join(res_dir, f, "summary_info.json"))
            )["cum_reward"]
        ]
    elif args.criteria == "autoeval":
        file_dirs = []
        for res_dir in args.result_dir:
            for f in os.listdir(res_dir):
                output_model_name = args.model.replace("/","_")
                record_path = os.path.join(res_dir, f, f"{output_model_name}_autoeval.json")
                if not os.path.exists(record_path): continue
                record = json.load(open(record_path))
                if record[0]["rm"]:
                    file_dirs.append(os.path.join(res_dir, f))
    else:
        raise ValueError(f"Invalid criteria: {args.criteria}.")
    
    print(f"Collected {len(file_dirs)} result directories.")

    # template id based deduplication
    template_dict = {}
    for f in file_dirs:
        # get query -> task objective
        task_id = f.split('/')[-1].split("_")[0].split(".")[1]
        config_path = os.path.join("config_files", f"{task_id}.json")
        config = json.load(open(config_path))
        query = config["intent"]

        template_id = config["intent_template_id"] # for deduplication

        # parse trajectory
        log_path = os.path.join(f, "experiment.log")
        try:
            think_list, action_list = extract_think_and_action(log_path)
        except:
            continue

        # add to template dict
        wdict = {"query": query, "think_list": think_list, "action_list": action_list}
        if template_id not in template_dict: template_dict[template_id] = []
        template_dict[template_id].append(wdict)
    selected_examples = random_group_sample(template_dict, args.num_samples)
    print(f"#{len(selected_examples)} result dirs after template dedup..")

    if "gpt" in args.model:
        openai.api_key = os.environ["OPENAI_API_KEY"]
        llm_client = OpenAI()
    else:
        llm_client = InferenceClient(args.model, token = os.environ.get('HF_TOKEN'))
    
    # prompt model to induce workflows
    workflows = llm_generate(llm_client, selected_examples, args, True)
    workflows += "\n\nclick('id') # input string id value for all actions\n\nselect_option('id', 'value') # for dropdown menu"

    if args.output_path is None:
        website = config["sites"][0]  # assumes all results are about the same website
        args.output_path = f"workflow/{website}_neural.txt"
        print(f"[Warning] no output path specified, using '{args.output_path}' by default")
        
    with open(args.output_path, 'w') as fw:
        fw.write(workflows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="results",
                        help="Path to the result directory. Support multiple directories separated by space.")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to the output file.")
    parser.add_argument("--criteria", type=str, default="autoeval", 
                        choices=["gt", "autoeval"],
                        help="'gt': only use examples with gold reward, 'autoeval': use examples with autoeval reward.")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        choices=["gpt-3.5", "gpt-4", "gpt-4o",
                                 "meta-llama/Llama-3.1-70B-Instruct","meta-llama/Llama-3.1-8B-Instruct"])
    parser.add_argument("--num_samples", type=int, default=1, help="Max number of samples to input per template.")
    args = parser.parse_args()

    args.INSTRUCTION = open("prompt/instruction.txt", 'r').read()
    args.ONE_SHOT = open("prompt/one_shot.txt", 'r').read()

    main()
