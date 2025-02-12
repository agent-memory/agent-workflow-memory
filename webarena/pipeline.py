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
    template_ids = [config["intent_template_id"] for config, flag in zip(config_list, config_flags) if flag]


    if args.induce_strategy == "rule":
        induction_path = "induce_rule.py"
    elif args.induce_strategy == "neural":
        induction_path = "induce_prompt.py"
    elif args.induce_strategy == "ngram":
        induction_path = "induce_ngram.py"
    elif args.induce_strategy == "ngram_prompt":
        induction_path = "induce_ngram_prompt.py"

    if args.end_index == None: args.end_index = len(task_ids)

    if args.template_first:
        # Rearranging tasks to have 1 of each template in the start
        print("Rearranging tasks to have 1 of each template in the start")
        grouped_tasks = {}
        # Iterate over both lists and group task_ids by template_ids
        for task, template in zip(task_ids, template_ids):
            if template not in grouped_tasks:
                grouped_tasks[template] = []
            grouped_tasks[template].append(task)

        # Convert defaultdict to a regular dict (optional)
        grouped_tasks = dict(grouped_tasks)

        # Output the result
        print(grouped_tasks)

        task_ids_rearranged = []
        for group in grouped_tasks.keys():
            first_task = grouped_tasks[group][0]
            task_ids_rearranged.append(first_task)

        for id in task_ids_rearranged:
            task_ids.remove(id)
        task_ids_rearranged = task_ids_rearranged + task_ids
        task_ids = task_ids_rearranged
    
    print(task_ids)
    print(f"Executing {len(task_ids[args.start_index: args.end_index])} tasks")

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
            "--temperature", f"{args.agent_temperature}"
        ])
        process.wait()

        if args.induce_strategy != "no_induce":
            # step 2: run evaluation
            process = Popen([
                "python", "-m", "autoeval.evaluate_trajectory",
                "--result_dir", f"results/webarena.{tid}",
                "--model", f"{args.model_name}"
            ])
            process.wait()

            # step 3: update workflow
            command = [
                "python", f"induction/{induction_path}",
                "--result_dir", "results",
                "--output_path", f"workflow/{args.website}.txt",
                "--model", f"{args.model_name}",
                "--tid", f"{tid}",
            ]
            if args.auto and args.induce_strategy in ['rule', 'ngram', 'ngram_prompt']:
                command.append("--auto")
            if args.add_failures and args.induce_strategy == "ngram_prompt": # Failure only supported for ngram prompt
                command.append("--add_failures")
            if args.prefix_workflow_path:
                command.extend(["--prefix_workflow_path", args.prefix_workflow_path])

            process = Popen(command)
            process.wait()

        print(f"Completed tid:{tid}, time taken:{time.time() - start_time}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--website", type=str, required=True,
                        choices=["shopping", "shopping_admin", "gitlab", "reddit", "map"])
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--induce_strategy", type=str, default="rule", choices=["rule", "neural", "ngram", "ngram_prompt", "no_induce"])
    parser.add_argument("--prefix_workflow_path", type=str, default=None)
    parser.add_argument("--auto", action="store_true", default=False)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--template_first", default=False, action="store_true", help="Rearrange tasks to perform 1 task of each template before other tasks")
    parser.add_argument("--add_failures", default=False, action="store_true", help="Add failures to the workflow")
    parser.add_argument("--agent_temperature", type=float, default=0.1, help="Temperature for the LLM Agent")
    args = parser.parse_args()

    main()
