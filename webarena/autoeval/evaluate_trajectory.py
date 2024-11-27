import os
import json
import argparse
import traceback
from autoeval.evaluator import Evaluator
from autoeval.clients import CLIENT_DICT
import tiktoken
from transformers import AutoTokenizer
from induction.utils import extract_think_and_action

def extract_response(action: str) -> str:
    s, e = action.index("(")+1, action.index(")")
    return action[s: e]

def get_number_tokens(prompt_text, model_name="gpt-4o"): 
    if "gpt" in model_name:
        encoding = tiktoken.encoding_for_model(model_name)
    else:
        encoding = AutoTokenizer.from_pretrained(model_name)
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
