import os
import json
import argparse
import traceback
from autoeval.evaluator import Evaluator
from autoeval.clients import CLIENT_DICT
import tiktoken

def load_blocks(path: str) -> list[list[str]]:
    """Load blank-line separated blocks from the log file."""
    # Rules/strings for different types of lines
    thought_str = 'browsergym.experiments.loop - INFO -'
    warning_str = 'root - WARNING'
    http_str = 'httpx - INFO'
    failure_str = 'root - INFO - Query failed. Retrying' # TODO also find failure for GPT-4
    action_str = 'action:'

    blocks, block = [], []
    
    for line in open(path, 'r'):
        if action_str in line or thought_str in line or failure_str in line:
            if len(block) > 0 and failure_str not in block[0]: # If failure block do not add block
                blocks.append(block)
            block = []
            block.append(line.strip())
        else:
            if line.strip():
                if warning_str in line or http_str in line:
                    continue # Do not add warning or HTTP Info lines
                block.append(line.strip())

    blocks.append(block) # Add Last block

    if len(blocks) > 0 and 'Python version' in blocks[0][0]:
        blocks = blocks[1:] # remove conda env output
    if len(blocks) > 0 and 'Running experiment' in blocks[0][0]:
        blocks = blocks[1:] # remove initial prompt output

    assert len(blocks) % 2 == 0
    return blocks

def remove_invalid_steps(actions: list[str]) -> list[str]:
    """Remove invalid steps from the action sequence."""
    valid_actions = []
    for a in actions:
        if "click(" in a:
            arg = a[a.index("(")+1: a.index(")")]
            try:
                if type(eval(arg)) == str:
                    valid_actions.append(a)
            except Exception as e:
                print(f"Error in remove_invalid_steps: {e}")
                continue
        elif "fill(" in a:
            arg = a[a.index("(")+1: a.index(",")].strip()
            try:
                if type(eval(arg)) == str:
                    valid_actions.append(a)
            except Exception as e:
                print(f"Error in remove_invalid_steps: {e}")
                continue
        else:
            valid_actions.append(a)
    return valid_actions

def extract_think_and_action(path: str) -> tuple[list[str], list[str]]:
    """Extract the task trajectory from the log file."""
    blocks = load_blocks(path)
    think_list, action_list = [], []
    for i in range(1, len(blocks), 2):
        # action
        b = blocks[i]
        actions = remove_invalid_steps(b[1:])
        if len(actions) == 0: continue
        action_list.append(actions)
        # think
        b = blocks[i-1]
        # Handling multiple lines of thoughts and stripping off the prefix in first line
        idx = b[0].index("browsergym.experiments.loop - INFO -")
        b[0] = b[0][idx+36: ].strip()
        thought = ' '.join(b).strip()
        think_list.append(thought)
    
    assert len(think_list) == len(action_list)
    
    # TODO: merge same actions
    return think_list, action_list

def extract_response(action: str) -> str:
    s, e = action.index("(")+1, action.index(")")
    return action[s: e]

def get_number_tokens(prompt_text, model_name="gpt-4o"): 
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(prompt_text) 
    return len(tokens) 

def process_sample(
    idx: str, traj_info: dict, log_save_path,
    model: str, eval_version: str,
) -> list[dict]:
    clients = {model: CLIENT_DICT[model](model_name=model)}
    evaluator = Evaluator(clients, log_save_path=log_save_path + "/trajs")
    try:
        out, prompt, full_output = evaluator(traj_info, model, eval_version)
        prompt_tokens, output_tokens = get_number_tokens(prompt), get_number_tokens(full_output)
        eval_result = None
        if out["status"].lower() == "success": eval_result = True
        else: eval_result = False
        return [{
                "idx": idx,
                "gt": traj_info["eval"],
                "rm": eval_result,
                "thoughts": out["thoughts"], 
                "uid": traj_info["traj_name"],
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
        }]
    except Exception as e:
        print(f"Error on {idx}, {e}")
        print(traceback.format_exc())
        return [{
            "idx": idx,
            "gt": traj_info["eval"],
            "rm": None,
            "thoughts": None, 
            "uid": traj_info["traj_name"],
            "prompt_tokens": -1,
            "output_tokens": -1
        }]


def main():
    # load task config
    task_id = args.result_dir.split('/')[-1].split(".")[1]
    config_path = os.path.join("config_files", f"{task_id}.json")
    config = json.load(open(config_path))

    # load trajectory log
    log_path = os.path.join(args.result_dir, "experiment.log")
    think_list, action_list = extract_think_and_action(log_path)
    actions = [act for acts in action_list for act in acts]
    if "send_msg_to_user" in action_list[-1][0]:
        response = extract_response(action_list[-1][0])
    else:
        response = ""
    
    # load summary info
    summary_path = os.path.join(args.result_dir, "summary_info.json")
    summary = json.load(open(summary_path, 'r'))

    # collect traj info
    image_paths = [
        os.path.join(args.result_dir, f) for f in os.listdir(args.result_dir) 
        if f.startswith("screenshot_step_") and f.endswith(".jpg")
    ]
    image_paths = sorted(image_paths, key=lambda x: int(x.split('/')[-1].split("_")[-1].split(".")[0]))
    traj_info = {
        "intent": config["intent"],
        "response": response,
        "captions": think_list,
        "actions": actions,
        "traj_name": config["task_id"],
        "image_paths": image_paths,
        "images": image_paths,
        "eval": summary["cum_reward"]
    }

    # evaluate trajectory
    log_save_path = os.path.join("autoeval/log", args.result_dir.split('/')[-1])
    print("Log Save Path:", log_save_path)
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)
        os.makedirs(log_save_path + "/trajs")
    eval_info = process_sample(
        idx=config["task_id"], traj_info=traj_info,
        log_save_path=log_save_path, 
        model=args.model, eval_version=args.prompt,
    )
    output_model_name = args.model.replace("/","_")
    output_eval_path = os.path.join(args.result_dir, f"{output_model_name}_autoeval.json")
    json.dump(eval_info, open(output_eval_path, 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Path to the result directory, e.g., 'webarena.0'.")
    # autoeval
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                        choices=["gpt-3.5", "gpt-4", "gpt-4o","llama3.1-8b",
                                 "meta-llama/Llama-3.1-70B-Instruct","meta-llama/Llama-3.1-8B-Instruct"])
    parser.add_argument("--prompt", type=str, default="text",
                        choices=["text", "vision"])

    args = parser.parse_args()

    if args.model == "gpt-4o" and args.prompt != "vision":
        print(f"Waring: use vision prompt by default for {args.model}.")
        args.prompt = "vision"

    main()
