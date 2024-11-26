import os
import json
import random
import argparse
import datetime

from utils import extract_think_and_action, get_abstract_actions

def format_trajectory(think_list: list[str], action_list: list[list[str]]) -> str:
    trajectory = []
    for t, a in zip(think_list, action_list):
        acts = '\n'.join(a)
        trajectory.append(f"<think>\n{t}\n</think>\n<action>\n{acts}\n</action>")
    return '\n\n'.join(trajectory)

def get_abstract_trajectory(action_list: list[list[str]]) -> str:
    abstract = []
    for acts in action_list:
        for a in acts:
            s = a.index("(")
            e = a.index(',', s) if ',' in a[s:] else a.index(")", s)
            action = a[:s]
            if action != "send_msg_to_user":
                arg = a[s+1: e]
                abstract.append(f"{action}({arg})")
            else:
                abstract.append(f"{action}")
    return '_'.join(abstract)

def random_group_sample(d: dict, n) -> list:
    """Randomly sample n groups from the dictionary."""
    return [ex for v in d.values() for ex in random.sample(v, n)]


def main():
    # collect result directories, e.g., ["results/webarena.0", ...]
    if args.prefix_worflow_path:
        with open(args.prefix_worflow_path, 'r') as fr:
            prefix_worflow = fr.read()
    else:
        prefix_worflow = "## Concrete Examples"

    args.result_dir = args.result_dir.split()
    if args.criteria == "gt":
        file_dirs = [
            os.path.join(res_dir, f) for res_dir in args.result_dir for f in os.listdir(res_dir) 
            if os.path.exists(os.path.join(res_dir, f, "summary_info.json")) and json.load(
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
                try:
                    if record[0]["rm"]:
                        file_dirs.append(os.path.join(res_dir, f))
                except:
                    print(f"Autoeval format issue: tid {f}")
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
    selected_workflows = random_group_sample(template_dict, 1)
    print(f"#{len(selected_workflows)} result dirs after template dedup..")
    
    # deduplicate by abstract trajectory
    abstraj_dict = {}
    for w in selected_workflows:
        abs_traj = get_abstract_trajectory(w['action_list'])
        if abs_traj not in abstraj_dict:
            abstraj_dict[abs_traj] = []
        abstraj_dict[abs_traj].append(w)
    selected_workflows = random_group_sample(abstraj_dict, 1)
    print(f"#{len(selected_workflows)} result dirs after trajectory dedup..")

    # manual inspection
    def get_workflow(d: dict) -> str:
        return f"Query: {d['query']}\n" + format_trajectory(d['think_list'], d['action_list'])
    manual_workflows = []
    for w in selected_workflows:
        w = get_workflow(w)
        if args.auto: 
            to_add = 'y'
        else:
            to_add = input("Workflow: \n" + w + "\n\nAdd? (y/n): ")
        if to_add == 'y':
            manual_workflows.append(w)
    print(f"#{len(manual_workflows)} result dirs after manual inspection..")



    if args.output_path is None:
        website = config["sites"][0]  # assumes all results are about the same website
        args.output_path = f"workflow/{website}.txt"
        print(f"[Warning] no output path specified, using '{args.output_path}' by default")
        
    with open(args.output_path, 'w') as fw:
        fw.write('\n\n\n'.join([prefix_worflow] + manual_workflows))

    if not os.path.exists('step_wise_workflows'):
        os.makedirs('step_wise_workflows')
        print(f"Directory 'step_wise_workflows' created.")

    exp_folder = f'step_wise_workflows/rule_{args.model.replace("/", "_")}/'
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
        print(f"Directory {exp_folder} created.")

    step_wise_workflow_dir = os.listdir(exp_folder)
    if len(step_wise_workflow_dir) == 0:
        step_idx = 0
    else:
        step_ids = [int(workflow_dir.split('step_')[-1].split('.txt')[0]) for workflow_dir in step_wise_workflow_dir]
        last_idx = sorted(step_ids)[-1]
        step_idx = last_idx + 1

    # TODO add the workflows to the task folder in results itself

    with open(f'{exp_folder}/workflow_task_{args.tid}_step_{step_idx}.txt','w') as fw:
        fw.write('\n\n\n'.join([prefix_worflow] + manual_workflows))


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
    parser.add_argument("--auto", action="store_true", help="w/o manual workflow inspections.")
    parser.add_argument("--tid", default=None, help="Task id when induction called")
    parser.add_argument("--prefix_worflow_path", default=None, help="Path to the prefix workflow file.")
    args = parser.parse_args()

    main()
