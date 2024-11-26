import os
import json
import random
import argparse

from utils import extract_think_and_action, get_abstract_actions

def generate_ngrams(abstract_action_list, n):
    abstract_ngrams = []
    ngram_idxs = []
    # action_ngrams = []
    # Loop through the actions to generate unique n-gram action sequences
    for i in range(len(abstract_action_list) - n + 1):
        abs_ngram = abstract_action_list[i:i+n]
        if abs_ngram not in abstract_ngrams:
            abstract_ngrams.append(abs_ngram)
            ngram_idxs.append(list(range(i,i+n)))
    return abstract_ngrams, ngram_idxs

def get_n_gram_action_sequences(abstract_action_list, n_range =[2,3]):
    # n-gram creation based on abstract, store actions
    n_gram_dict = {}
    for n in n_range:
        abs_ngram, ngram_idxs = generate_ngrams(abstract_action_list, n)
        n_gram_dict[n] = (abs_ngram, ngram_idxs)
    return n_gram_dict

def remove_hallucinating_actions(action_list):

    # truncate to obtain non-repeating sub actions
    # remove based on actions, not abstract action
    prev_action  = ''
    idx = 0
    for subactions in action_list:
        curr_action = "/".join(subactions)
        if curr_action == prev_action:
            idx -= 1 # remove prev action also as it is repeated
            return idx
        prev_action = curr_action
        idx += 1
    return idx


def main():
    # collect result directories, e.g., ["results/webarena.0", ...]
    prefix_worflow = "## Subtrajectory Examples"

    args.result_dir = args.result_dir.split()
    if args.criteria == "gt":
        file_dirs = [
            os.path.join(res_dir, f) for res_dir in args.result_dir for f in os.listdir(res_dir) 
            if os.path.exists(os.path.join(os.path.join(res_dir, f, "summary_info.json"))) and json.load(
                open(os.path.join(res_dir, f, "summary_info.json"))
            )["cum_reward"]
        ]
    elif args.criteria == "autoeval":
        file_dirs = []
        for res_dir in args.result_dir:
            for f in os.listdir(res_dir):
                ## Adding all paths irrespective of autoeval trajectory
                # record_path = os.path.join(res_dir, f, f"{args.model}_autoeval.json")
                # if not os.path.exists(record_path): continue
                # record = json.load(open(record_path))
                # try:
                #     if record[0]["rm"]:
                #         file_dirs.append(os.path.join(res_dir, f))
                # except:
                #     print(f"Autoeval format issue: tid {f}")
                file_dirs.append(os.path.join(res_dir, f))
    else:
        raise ValueError(f"Invalid criteria: {args.criteria}.")
    
    print(f"Collected {len(file_dirs)} result directories.")

    # template id based deduplication
    template_dict = {}
    all_task_action_think = {}
    for f in file_dirs:
        # get query -> task objective
        task_id = f.split('/')[-1].split("_")[0].split(".")[1]
        config_path = os.path.join("config_files", f"{task_id}.json")
        config = json.load(open(config_path))
        query = config["intent"]

        # template_id = config["intent_template_id"] # for deduplication

        # parse trajectory
        log_path = os.path.join(f, "experiment.log")
        try:
            think_list, action_list = extract_think_and_action(log_path)
        except:
            think_list, action_list = [],[]

        print(f"Here Task ID: {task_id}")
        print(action_list)
        # add to template dict
        abstract_traj = get_abstract_actions(action_list)

        assert len(abstract_traj) == len(action_list) == len(think_list)

        ## Remove hallucinating actions - find max index before repeated actions (compared based on both sub-action and arg)
        truncation_idx = remove_hallucinating_actions(action_list)
        abstract_traj = abstract_traj[:truncation_idx]
        action_list = action_list[:truncation_idx]
        think_list = think_list[:truncation_idx]

        assert len(abstract_traj) == len(action_list) == len(think_list)

        subaction_list = action_list # [subaction for action in action_list for subaction in action] # If doing subaction as ngrams - based on the granularity
        # assert len(abstract_traj) == len(subaction_list)
        action_ngrams_dict = get_n_gram_action_sequences(abstract_traj) # indices are used to index original thought and action pairs
        wdict = {"task_id":task_id,"query": query, "think_list": think_list, "action_list": action_list,
                 "abstract_traj": abstract_traj, "subaction_list": subaction_list, "action_ngrams": action_ngrams_dict}
        # print(task_id)
        # print(abstract_traj)
        # print(action_ngrams_dict)
        # print("=======")
        all_task_action_think[task_id] = wdict

    test_n = 2
    ngram_groups = {}
    # Count overlapping ngrams - for each ngram store the task id and the corresponding action sequence
    for tid in all_task_action_think:
        trajectory_log = all_task_action_think[tid]
        ngrams = trajectory_log['action_ngrams'][test_n]
        if len(ngrams) != 0: # If ngrams exist
            for ngram_action, n_gram_idx in zip(ngrams[0], ngrams[1]):
                ngram_key = ", ".join(ngram_action) # using the abstract action list
                if ngram_key not in ngram_groups:
                    ngram_groups[ngram_key] = []
                ngram_groups[ngram_key].append({'task_id':trajectory_log['task_id'], 'ngram_idxs':n_gram_idx})

    ngram_freq = {}
    for key in ngram_groups.keys():
        task_ids = [task['task_id'] for task in ngram_groups[key]]
        ngram_freq[key] = len(task_ids)
    
    # Group frequency for debugging
    # for key, value in sorted(ngram_freq.items(), key=lambda item: item[1], reverse=True):
    #     print("-"* 80)
    #     print(f"{key.ljust(70)} | {str(value).rjust(5)}|")

    # Only induct if the key occurs more than 3 times


    def format_trajectory(think_list: list[str], action_list: list[list[str]]) -> str:
        trajectory = []
        for t, a in zip(think_list, action_list):
            acts = '\n'.join(a)
            trajectory.append(f"<think>\n{t}\n</think>\n<action>\n{acts}\n</action>")
        return '\n\n'.join(trajectory)

    def get_workflow(ngram_format, think_action_pair_str) -> str:
        return f"Examples with given sub-trajectory: {ngram_format}\n\nExample:\n" + "\n\nExample:\n".join(think_action_pair_str)
    
    def get_nl_desc_ngram(ngram_key):
        nl_desc = []
        actions = ngram_key.split(",")
        for i, subactions in enumerate(actions):
            subactions = subactions.split("/")
            subactions_cmd = ['send_msg_to_user' if 'send_msg_to_user' in a else a[:a.index("(")] for a in subactions]
            subactions_str = " --> ".join(subactions_cmd)
            nl_desc.append(f"Action {i+1}: {subactions_str}")
        nl_desc_str = ", ".join(nl_desc)
        return nl_desc_str

    workflows = []
    induction_thr = 2
    ctr = 0
    for key in ngram_groups.keys():
        if ngram_freq[key] >= induction_thr:
            random_sample = random.sample(ngram_groups[key], 2) # should be less than induction_thr
            ngram_format = get_nl_desc_ngram(key)
            examples_strs = []
            for task in random_sample:
                tid = task['task_id']
                ngram_idxs = task['ngram_idxs']
                think_list = [all_task_action_think[tid]['think_list'][i] for i in ngram_idxs]
                action_list = [all_task_action_think[tid]['action_list'][i] for i in ngram_idxs]

                task_action_paired_str = format_trajectory(think_list, action_list)
                examples_strs.append(task_action_paired_str)

            workflow_formatted = get_workflow(ngram_format, examples_strs)
            workflows.append(workflow_formatted)

            # print(workflow_formatted)
            # print("=================")
            # break
            ctr += 1

    print(f"#{len(workflows)} result dirs for ngram based selection..")

    # print(ctr)

    #     if template_id not in template_dict: template_dict[template_id] = []
    #     template_dict[template_id].append(wdict)
    # selected_workflows = random_group_sample(template_dict, 1)
    # print(f"#{len(selected_workflows)} result dirs after template dedup..")
    
    # # deduplicate by abstract trajectory
    # abstraj_dict = {}
    # for w in selected_workflows:
    #     abs_traj = get_abstract_trajectory(w['action_list'])
    #     if abs_traj not in abstraj_dict:
    #         abstraj_dict[abs_traj] = []
    #     abstraj_dict[abs_traj].append(w)
    # selected_workflows = random_group_sample(abstraj_dict, 1)
    # print(f"#{len(selected_workflows)} result dirs after trajectory dedup..")

    # # manual inspection
    # def get_workflow(d: dict) -> str:
    #     return f"Query: {d['query']}\n" + format_trajectory(d['think_list'], d['action_list'])
    # manual_workflows = []
    # for w in selected_workflows:
    #     w = get_workflow(w)
    #     if args.auto: 
    #         to_add = 'y'
    #     else:
    #         to_add = input("Workflow: \n" + w + "\n\nAdd? (y/n): ")
    #     if to_add == 'y':
    #         manual_workflows.append(w)
    # print(f"#{len(manual_workflows)} result dirs after manual inspection..")


    if args.output_path is None:
        website = config["sites"][0]  # assumes all results are about the same website
        args.output_path = f"workflow/{website}.txt"
        print(f"[Warning] no output path specified, using '{args.output_path}' by default")
        
    with open(args.output_path, 'w') as fw:
        fw.write('\n\n\n'.join([prefix_worflow] + workflows))

    if not os.path.exists('step_wise_workflows'):
        os.makedirs('step_wise_workflows')
        print(f"Directory 'step_wise_workflows' created.")

    exp_folder = f'step_wise_workflows/ngram_{args.model.replace("/", "_")}/'
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
        fw.write('\n\n\n'.join([prefix_worflow] + workflows))

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
                        choices=["gpt-3.5", "gpt-4", "gpt-4o","gpt-4-turbo",
                                 "meta-llama/Llama-3.1-70B-Instruct","meta-llama/Llama-3.1-8B-Instruct"])
    parser.add_argument("--auto", action="store_true", help="w/o manual workflow inspections.")
    parser.add_argument("--tid", default=None, help="Task id when induction called")
    args = parser.parse_args()

    main()
