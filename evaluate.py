import os
import json
import matplotlib.pyplot as plt
import tiktoken
from transformers import AutoTokenizer
import argparse

def success_rate(model_prefix, result_dir, plot_path=None):
    # for file in all_tasks:
    #     with open(f"{task_dir}/{file}") as f_in:
    #         try:
    #             json_obj = json.load(f_in)
    #             sites = json_obj['sites']
    #             site_key = "_".join(sites)
    #             if site_key in site_ctr.keys():
    #                 site_ctr[site_key] += 1
    #             else:
    #                 site_ctr[site_key] = 1
    #             if task in sites:
    #                 sub_tasks.append(int(file.split('.json')[0]))
    #         except:
    #             continue # expected for for test/raw json files

    # print("Distribution across tasks")
    # print(site_ctr)

    # sub_tasks = sorted(sub_tasks)
    # print(f"Total tasks:{len(sub_tasks)}")

    tasks_ran = os.listdir(result_dir)
    tasks_ran = sorted([int(t.split('webarena.')[-1]) for t in tasks_ran if '_' not in t])

    print(f"Result files:{len(tasks_ran)}")

    # Calculate cumulative success rate
    cum_reward_list = []
    success_rate_list = []
    autoeval_exec = []
    autoeval_rewards = []
    autoeval_gts = []
    autoeval_tasks = []

    for i, file in enumerate(tasks_ran):
        with open(f"{result_dir}/webarena.{file}/summary_info.json") as f_in:
            try:
                json_obj = json.load(f_in)
                reward = int(json_obj['cum_reward'])
                cum_reward_list.append(reward)
                # Calculate the average success rate up to this task
                avg_success_rate = sum(cum_reward_list) / (i+1)
                success_rate_list.append(avg_success_rate)
            except:
                continue  # expected for for test/raw json files

        if model_prefix is not None:
            try:
                # check if autoeval file is present
                with open(f"{result_dir}/webarena.{file}/{model_prefix}_autoeval.json") as f_in:
                    json_obj = json.load(f_in)[0]
                    autoeval_reward = int(json_obj['rm'])
                    gt = int(json_obj['gt'])
                    autoeval_exec.append(True)
                    autoeval_tasks.append(file)
                    autoeval_gts.append(gt)
                    autoeval_rewards.append(autoeval_reward)
            except:
                autoeval_exec.append(False)

    print("% of correct trajectories:")
    print(sum(cum_reward_list)/len(cum_reward_list))
    print("Cum reward with file")
    print([ (task_file,reward) for task_file,reward in zip(tasks_ran,cum_reward_list)])

    # Print the average success rate at each step/trajectory
    print("Average success rate at each step:")
    print(success_rate_list)

    if plot_path is not None:
        plt.plot(range(1, len(success_rate_list) + 1), success_rate_list, marker='.', color='b', linestyle='-')
        plt.xlabel('Trajectory')
        plt.ylabel('Average Success Rate')
        plt.title('Average Success Rate vs Trajectory')
        plt.grid()
        plt.savefig(plot_path)
        plt.close()

    if model_prefix is not None:
        print("% of trajectories autoeval executed for:")
        print(sum(autoeval_exec) / len(autoeval_exec))
        print("% of trajectories autoeval RM True:")
        print(sum(autoeval_rewards) / len(autoeval_rewards))
        print("% of trajectories autoeval GT True:")
        print(sum(autoeval_gts) / len(autoeval_gts))

        print("Autoeval exec, gt, reward")

        assert len(autoeval_tasks) == len(autoeval_gts)
        print([(task_file, autoeval_executed) for task_file, autoeval_executed in zip(autoeval_tasks, autoeval_exec)])
        print([(task_file, curr_reward) for task_file, curr_reward in zip(autoeval_tasks, autoeval_rewards)])
        print([(task_file, curr_gt) for task_file, curr_gt in zip(autoeval_tasks, autoeval_gts)])

def memory_efficiency(model, stepwise_workflow_path, plot_path=None):
    if "gpt" in model:
        enc = tiktoken.encoding_for_model(model)
    else:
        enc = AutoTokenizer.from_pretrained(model)

    memory_tokens = [] 

    stepwise_workflows = os.listdir(stepwise_workflow_path)
    stepwise_workflows = sorted(stepwise_workflows, key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Sort by the number itself

    for file in stepwise_workflows:
        with open(f"{stepwise_workflow_path}/{file}") as f_in:
            workflow_text = f_in.read()
            # Calculate tokens with tokenizer huggingface / tiktoken 
            tokens = len(enc.encode(workflow_text))
            memory_tokens.append(tokens)
    
    print("Memory tokens")
    print(memory_tokens)

    if plot_path is not None:
        plt.plot(range(1, len(memory_tokens) + 1), memory_tokens, marker='.', color='b', linestyle='-')
        plt.xlabel('Trajectory')
        plt.ylabel('Memory Tokens')
        plt.title('Memory Tokens vs Trajectory')
        plt.grid()
        plt.savefig(plot_path)
        plt.close()

def main(args):
    if not os.path.exists(args.plot_path):
        os.makedirs(args.plot_path)

    success_rate(args.model.replace("/", "_"), args.result_dir, plot_path=f'{args.plot_path}/success_rate.png')
    if args.stepwise_workflow_path is not None:
        memory_efficiency(args.model, args.stepwise_workflow_path, plot_path=f'{args.plot_path}/memory_efficiency.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Model prefix for autoeval")
    parser.add_argument("--result_dir", type=str, default=None,
                        help="Result directory")
    parser.add_argument("--stepwise_workflow_path", type=str, default=None,
                        help="Path to the stepwise workflow directory")
    parser.add_argument("--plot_path", type=str, default=None,
                        help="Path to save the plot")
    args = parser.parse_args()
    main(args)