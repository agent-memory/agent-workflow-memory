import os
import json
import argparse
from subprocess import Popen
import time

def main():
    # collect examples
    config_files = [
        os.path.join("config_files", f) for f in os.listdir("config_files")
        if f.endswith(".json") and f.split(".")[0].isdigit()
    ]
    config_files = sorted(config_files, key=lambda x: int(x.split("/")[-1].split(".")[0]))
    config_list = [json.load(open(f)) for f in config_files]
    config_flags = [config["sites"][0] == args.website for config in config_list]
    task_ids = [config["task_id"] for config, flag in zip(config_list, config_flags) if flag]

    if args.induce_strategy == "rule":
        induction_path = "induce_rule.py"
    elif args.induce_strategy == "neural":
        induction_path = "induce_prompt.py"
    elif args.induce_strategy == "ngram":
        induction_path = "induce_ngram.py"

    if args.end_index == None: args.end_index = len(task_ids)
    for tid in task_ids[args.start_index: args.end_index]:
        print(f"Started tid:{tid}", flush=True)
        start_time = time.time()
        # step 1: run inference
        process = Popen([
            "python", "run.py", 
            "--task", f"webarena.{tid}",
            "--workflow_path", f"workflow/{args.website}.txt",
            "--model_name", f"{args.model_name}",
            "--headless", "True" if args.headless else "False",
        ])
        process.wait()

        # step 2: run evaluation
        process = Popen([
            "python", "-m", "autoeval.evaluate_trajectory",
            "--result_dir", f"results/webarena.{tid}",
            "--model", f"{args.model_name}"
        ])
        process.wait()

        # step 3: update workflow
        command = [
            "python", induction_path,
            "--result_dir", "results",
            "--output_path", f"workflow/{args.website}.txt",
            "--model", f"{args.model_name}",
            "--tid", f"{tid}",
        ]
        if args.auto and args.induce_strategy in ['rule', 'ngram']:
            command.append("--auto")
        if args.prefix_workflow_path:
            command.extend(["--prefix_workflow_path", args.prefix_workflow_path])

        process = Popen(command)
        process.wait()

        print(f"Completed tid:{tid}, time taken:{time.time() - start_time}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--website", type=str, required=True,
                        choices=["shopping", "shopping_admin", "gitlab", "reddit", "map"])
    parser.add_argument("--model_name", type=str, default="openai/gpt-4o")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--induce_strategy", type=str, default="rule", choices=["rule", "neural", "ngram"])
    parser.add_argument("--prefix_workflow_path", type=str, default=None)
    parser.add_argument("--auto", action="store_true", default=False)
    parser.add_argument("--headless", action="store_true", default=False)
    args = parser.parse_args()

    main()
