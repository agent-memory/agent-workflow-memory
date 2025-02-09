def load_blocks(path: str) -> list[list[str]]:
    """Load blank-line separated blocks from the log file."""
    # Rules/strings for different types of lines
    thought_str = 'browsergym.experiments.loop - INFO -'
    warning_str = 'WARNING'
    http_str = 'httpx - INFO'
    failure_str = 'root - INFO - Query failed. Retrying' # TODO also find failure for GPT-4
    content_failure_str = "Error('Execution context was destroyed, most likely because of a navigation')"
    recovery_fix = "browsergym.core.env - INFO - The active page and / or page history has changed during task.validate()"
    action_str = 'action:'

    blocks, block = [], []
    
    for line in open(path, 'r'):
        if action_str in line or thought_str in line or failure_str in line or recovery_fix in line:
            if len(block) > 0 and failure_str not in block[0]: # If failure block do not add block
                blocks.append(block)
            if action_str in line and thought_str in line: # to fix deepseek bug # TODO: Handle in parsing logic of browsergym 
                blocks.append(['']) # empty thought str
            block = []
            block.append(line.strip())
        else:
            if line.strip():
                if warning_str in line or http_str in line or content_failure_str in line:
                    continue # Do not add warning or HTTP Info lines
                block.append(line.strip())

    blocks.append(block) # Add Last block

    if len(blocks) > 0 and 'Python version' in blocks[0][0]:
        blocks = blocks[1:] # remove conda env output
    if len(blocks) > 0 and 'Running experiment' in blocks[0][0]:
        blocks = blocks[1:] # remove initial prompt output
    if len(blocks) > 0 and recovery_fix in blocks[-1][0]:
        blocks = blocks[:-1] # remove final extra fix

    assert len(blocks) % 2 == 0
    return blocks

def remove_invalid_steps(actions: list[str]) -> list[str]:
    """Remove invalid steps from the action sequence."""
    valid_actions = []
    for a in actions:
        if "click(" in a:
            arg = a[a.index("(")+1: a.index(")")]
            try:
                if type(eval(arg)) == str and type(eval(arg[1:-1])) == int:
                    valid_actions.append(a)
            except:
                continue
        elif "fill(" in a:
            arg = a[a.index("(")+1: a.index(",")].strip()
            if type(eval(arg)) == str:
                valid_actions.append(a)
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
        if "browsergym.experiments.loop - INFO -" in b[0]:
            idx = b[0].index("browsergym.experiments.loop - INFO -")
            b[0] = b[0][idx+36: ].strip()
        thought = ' '.join(b).strip()
        think_list.append(thought)
    
    assert len(think_list) == len(action_list)
    
    # TODO: merge same actions
    return think_list, action_list

def get_abstract_actions(action_list: list[list[str]]) -> list[str]:
    # Remove the arguments
    abstract = []
    for acts in action_list:
        # curr_action = []
        for a in acts:
            try:
                s = a.index("(")
                e = a.index(',', s) if ',' in a[s:] else a.index(")", s) # Only uses the first argument of the input (eg: fill has 2)
                action = a[:s]
                if action != "send_msg_to_user":
                    arg = a[s+1: e]
                    abstract.append(f"{action}({arg})")
                else:
                    abstract.append(f"{action}")
            except:
                print(f"Error in get_abstract_actions: {a}")
                continue
        # abstract.append("/".join(curr_action)) ### Earlier abstract.append
    # Need 1:1 corresp between abstract actions and overall actions
    return abstract

if __name__ == "__main__":
    load_blocks('results/webarena.573/experiment.log')
