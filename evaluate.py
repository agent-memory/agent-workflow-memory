import os
import json

task_dir = 'webarena/config_files'
all_tasks = os.listdir(task_dir)
sub_tasks = []
site_ctr = {}

task = 'map' # 'shopping'

# model_prefix = 'gpt-4-turbo' # 'meta-llama_Meta-Llama-3.1-70B-Instruct' # Find from autoeval in results dir
# result_dir = 'all_results/webarena/results_gpt4'

model_prefix = 'gpt-4o' # If used autoeval
result_dir = 'all_results/webarena/results_gpt4o'

# model_prefix = None
# result_dir = 'webarena/results'

for file in all_tasks:
    with open(f"{task_dir}/{file}") as f_in:
        try:
            json_obj = json.load(f_in)
            sites = json_obj['sites']
            site_key = "_".join(sites)
            if site_key in site_ctr.keys():
                site_ctr[site_key] += 1
            else:
                site_ctr[site_key] = 1
            if task in sites:
                sub_tasks.append(int(file.split('.json')[0]))
        except:
            continue # expected for for test/raw json files

print("Distribution across tasks")
print(site_ctr)

sub_tasks = sorted(sub_tasks)
print(f"Total tasks:{len(sub_tasks)}")

tasks_ran = os.listdir(result_dir)
tasks_ran = sorted([int(t.split('webarena.')[-1]) for t in tasks_ran if '_' not in t])

print(f"Result files:{len(tasks_ran)}")

# Calculate cumulative success rate
cum_reward_list = []
autoeval_exec = []
autoeval_rewards = []
autoeval_gts = []
autoeval_tasks = []

for file in tasks_ran:
    with open(f"{result_dir}/webarena.{file}/summary_info.json") as f_in:
        try:
            json_obj = json.load(f_in)
            reward = int(json_obj['cum_reward'])
            cum_reward_list.append(reward)
        except:
            continue # expected for for test/raw json files

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

if model_prefix is not None:
    print("% of trajectories autoeval executed for:")
    print(sum(autoeval_exec)/len(autoeval_exec))
    print("% of trajectories autoeval RM True:")
    print(sum(autoeval_rewards)/len(autoeval_rewards))
    print("% of trajectories autoeval GT True:")
    print(sum(autoeval_gts)/len(autoeval_gts))

    print("Autoeval exec, gt, reward")

    assert len(autoeval_tasks) == len(autoeval_gts)
    print([ (task_file,autoeval_executed) for task_file,autoeval_executed in zip(autoeval_tasks,autoeval_exec)])
    print([ (task_file,curr_reward) for task_file,curr_reward in zip(autoeval_tasks,autoeval_rewards)])
    print([ (task_file,curr_gt) for task_file,curr_gt in zip(autoeval_tasks,autoeval_gts)])




