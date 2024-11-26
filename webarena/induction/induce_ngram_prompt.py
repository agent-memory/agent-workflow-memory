import os
import json
import random
import argparse
import re

import openai
from openai import OpenAI
from huggingface_hub import InferenceClient

from induction.utils import extract_think_and_action, get_abstract_actions

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

def get_n_gram_action_sequences(abstract_action_list, n_range =[2,3,4,5]):
    # n-gram creation based on abstract, store actions
    n_gram_dict = {}
    for n in n_range:
        abs_ngram, ngram_idxs = generate_ngrams(abstract_action_list, n)
        n_gram_dict[n] = (abs_ngram, ngram_idxs)
    return n_gram_dict

def llm_validate_subtrajectory(llm_client, subtraj, examples, args, verbose: bool = False):
    """Call gpt model to validate if sub-trajectory is correct and provide explanation."""

    steps = "\n".join([step.strip() for step in subtraj.split(",")]) # Abstract Sub-trajectory that we want to validate
    query_format = f"Sub-trajectory query sequence:\n{steps}"
    
    for example in examples:
        query_format += f"\n\nActual trajectory:\n{example}"

    prompt = '\n\n'.join([args.INSTRUCTION, args.ONE_SHOT, query_format])

    if verbose: print("Prompt:\n", prompt, '\n\n')

    print(f"Model called for induction: {args.model}")
    if "gpt" in args.model:
        response = llm_client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            max_tokens=1024,
        )
    else:
        response = llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                max_tokens=2048,
        )

    response = response.choices[0].message.content
    
    inducted_workflow = None
    # Response parts: 0 - Validity 1- Explanation 2- NL Query 3- Example subtraj
    validity_match = re.search(r"<VALIDITY>\s*(\w+)", response)
    validity = validity_match.group(1) if validity_match else None
    if validity is not None and "True" in validity:
        is_valid = True
    else:
        is_valid = False
    if is_valid:
        query_match = re.search(r"<NATURAL_LANG_QUERY>\s*(.*?)<SUBTRAJECTORY_EXAMPLE_CALL>", response, re.DOTALL)
        nl_query = query_match.group(1).strip() if query_match else None
        
        subtrajectory_examples = response.split("<SUBTRAJECTORY_EXAMPLE_CALL>")[-1]
        subtrajectory_examples = "\n".join([x.strip() for x in subtrajectory_examples.split("\n")])
        
        inducted_workflow = f"Workflow: {nl_query}\n\nExample:{subtrajectory_examples}\n"
    if verbose: 
        print("=====================") 
        print(response)
    # return validity, inducted_workflow
    return is_valid, inducted_workflow

def main():
    # collect result directories, e.g., ["results/webarena.0", ...] 
    prefix_worflow = "## Subtrajectory Examples"
    
    # ======  Read the workflow file and write it out (As before induction is the input workflow) ===== 
    if not os.path.exists('step_wise_workflows'):
        os.makedirs('step_wise_workflows')
        print(f"Directory 'step_wise_workflows' created.")

    exp_folder = f'step_wise_workflows/ngram_prompt_{args.model.replace("/", "_")}/'
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
        print(f"Directory {exp_folder} created.")
    
    tid_folder = os.path.join(args.result_dir, f'webarena.{args.tid}') # This folder will exist

    step_wise_workflow_dir = os.listdir(exp_folder)
    if len(step_wise_workflow_dir) == 0:
        step_idx = 0
    else:
        step_ids = [int(workflow_dir.split('step_')[-1].split('.txt')[0]) for workflow_dir in step_wise_workflow_dir]
        last_idx = sorted(step_ids)[-1]
        step_idx = last_idx + 1

    # Read the workflow file
    prev_workflows = open(args.output_path, 'r').read()
    with open(f'{tid_folder}/workflow_task_{args.tid}_step_{step_idx}.txt','w') as fw:
        fw.write(prev_workflows)
    # ================

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
                # Adding all paths irrespective of autoeval trajectory
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

        # add to template dict
        abstract_traj = get_abstract_actions(action_list)

        action_ngrams_dict = get_n_gram_action_sequences(abstract_traj) # indices are used to index original thought and action pairs
        wdict = {"task_id":task_id,"query": query, "think_list": think_list, "action_list": action_list,
                 "abstract_traj": abstract_traj, "action_ngrams": action_ngrams_dict}
        all_task_action_think[task_id] = wdict

    # ngram and threshold schedule
    if len(file_dirs) < 40:
         test_n = 2
         induction_thr = 2
    elif len(file_dirs) < 80:
         test_n = 3
         induction_thr = 3
    else:
         test_n = 4
         induction_thr = 5

    print("N in n-gram:{}".format(test_n))

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
        print(f"Ngram:{key}, freq:{ngram_freq[key]}")

    def format_trajectory(think_list: list[str], action_list: list[list[str]]) -> str:
        trajectory = []
        for t, a in zip(think_list, action_list):
            acts = '\n'.join(a)
            trajectory.append(f"<think>\n{t}\n</think>\n<action>\n{acts}\n</action>")
        return '\n\n'.join(trajectory)

    workflows = []
    print("Induction threshold:{}".format(induction_thr))

    if "gpt" in args.model:
        openai.api_key = os.environ["OPENAI_API_KEY"]
        llm_client = OpenAI()
    else:
        llm_client = InferenceClient(args.model, token = os.environ.get('HF_TOKEN'))

    ## TODO: Load N-grams of each n, and check if the validation output is cached or not
    ## TODO: Right now checking only based on the format in 1 trajectory
    ctr = 0
    for key in ngram_groups.keys():
        if ngram_freq[key] >= induction_thr:
            random_sample = random.sample(ngram_groups[key], 1) # should be less than induction_thr
            examples_strs = []
            for task in random_sample:
                tid = task['task_id']
                think_list = all_task_action_think[tid]['think_list']
                action_list = all_task_action_think[tid]['action_list']
                task_action_paired_str = format_trajectory(think_list, action_list)
                examples_strs.append(task_action_paired_str)

            example_traj = "\n".join(examples_strs)
            is_valid, inducted_workflow = llm_validate_subtrajectory(llm_client, key, examples_strs, args, verbose=False)

            # print(is_valid)
            # print("***************")
            # print(inducted_workflow)

            # workflow_formatted = get_workflow(ngram_format, examples_strs)
            if is_valid:
                # import pdb; pdb.set_trace()
                workflows.append(inducted_workflow)
            # break
            ctr += 1

    # return 

    print(f"#{len(workflows)} result dirs for ngram based selection..")

    print('Workflows', workflows)

    if args.output_path is None:
        website = config["sites"][0]  # assumes all results are about the same website
        args.output_path = f"workflow/{website}.txt"
        print(f"[Warning] no output path specified, using '{args.output_path}' by default")
        
    with open(args.output_path, 'w') as fw:
        fw.write('\n\n\n'.join([prefix_worflow] + workflows))

    # Write in stepwise workflow folder
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

    args.INSTRUCTION = open("prompt/instruction_subtraj.txt", 'r').read()
    args.ONE_SHOT = open("prompt/one_shot_subtraj.txt", 'r').read()

    main()
