"""Adapted from https://github.com/ysymyth/ReAct/blob/master/alfworld.ipynb"""

import os
import sys
import json
import yaml
import openai
import numpy as np
import time
import importlib
from utils import Model, get_chat
from eth_env import ETHTradingEnv
from env_history import EnvironmentHistory

from typing import List, Dict, Any, Tuple
 
def llm(prompt, model, seed):
    try:
        text = get_chat(prompt=prompt, model=model, seed=seed)  # stop_strs=['\n']
        return text
    except Exception as e:
        print(prompt)
        print(e)
        import sys
        sys.exit(1)

def debug_print(s, response=None, title=''):
    print(f'\n*** START {title} ***')
    print(s)
    if response is not None:
        print(f'*** {title} RESPONSE ***')
        print(response)
    print(f'*** END {title} ***\n')

def eth_run(env, base_prompt, memory, starting_state, args):
    to_print = args.to_print
    model = args.model
    seed = args.seed

    if len(memory) > 3:
        env_history = EnvironmentHistory(base_prompt, starting_state, memory[-3:], [], args)
    else:
        env_history = EnvironmentHistory(base_prompt, starting_state, memory, [], args)
    if to_print:
        print_state = {k: v for k, v in starting_state.items() if k != 'news'}
        debug_print(print_state, None, 'STATE')
    cur_step = 0
    returns = []
    done = False
    while not done:
        use_news = args.use_news
        use_reflection = args.use_reflection
        price_s, news_s, reflection_s, template_s = env_history.get_prompt()

        onchain_analysis = llm(price_s, model, seed).strip()
        if to_print:
            print(f"********* START STEP {cur_step} *********")
            debug_print(price_s, onchain_analysis, 'ONCHAIN ANALYST')

        if use_news:
            news_analysis = llm(news_s, model, seed).strip()
            if to_print:
                debug_print(news_s, news_analysis, 'NEWS ANALYST')
        else:
            news_analysis = 'N/A'

        if use_reflection:
            reflection = llm(reflection_s, model, seed).strip()
            if to_print:
                debug_print(reflection_s, reflection, 'REFLECTION ANALYST')
        else:
            reflection = 'N/A'

        trader_prompt = template_s.format(onchain_analysis, news_analysis, reflection)
        trader_response = llm(trader_prompt, model, seed).strip()
        if to_print:
            debug_print(trader_prompt, trader_response, 'TRADER')

        state, reward, done, info = env.step(trader_response)
        raw_action = info['raw_action']
        actual_action = f"{info['actual_action']:.1f}"
        env_history.add("trader_response", trader_response)
        env_history.add("action", actual_action)
        env_history.add("state", state)
        returns.append(state['today_roi'])
        if to_print:
            print_state = {k: v for k, v in state.items() if k != 'news'}
            debug_print(actual_action, None, 'ACTUAL ACTION')
            debug_print(print_state, None, 'STATE')
            
            total_return = state['roi']
            tmp_returns = np.array(returns) * 100
            return_mean = np.mean(tmp_returns)
            return_std = np.std(tmp_returns)
            risk_free_rate = 0  # same as sociodojo
            sharpe_ratio = (return_mean - risk_free_rate) / return_std
            daily_result = f'Total return: {total_return*100:.2f}, sharpe ratio: {sharpe_ratio:.2f}, daily return mean: {return_mean:.2f}, daily return std: {return_std:.2f}'
            debug_print(daily_result, None, 'CURRENT RESULT')

        cur_step += 1
        time.sleep(1)
    is_success = total_return > 0.1 # modify sucess condition
    return env_history, is_success

def run_trial(
        trial_log_path,
        world_log_path,
        trial_idx,
        env_configs: List[Dict[str, Any]],
        args,
    ) -> List[Dict[str, Any]]:
    use_memory = args.use_memory

    env = ETHTradingEnv(args)

    num_successes: int = 0
    num_additional_successes: int = 0
    num_envs: int = len(env_configs)

    for z, env_config in enumerate(env_configs):
        starting_state, reward, done, info = env.reset()

        if env_config["is_success"]:
            num_successes += 1

            with open(world_log_path, 'a') as wf:
                wf.write(f'Environment #{z} Trial #{trial_idx}: SUCCESS\n')
            with open(trial_log_path, 'a') as wf:
                wf.write(f'\n#####\n\nEnvironment #{z}: Success\n\n#####\n')
            continue

        final_env_history, is_success = eth_run(env, '', env_config["memory"] if use_memory else [], starting_state, args=args)

        # update env config
        if is_success:
            status_str: str = f'Environment #{z} Trial #{trial_idx}: SUCCESS'
            env_configs[z]['is_success'] = True
            num_successes += 1
            num_additional_successes += 1
        else:
            status_str: str = f'Environment #{z} Trial #{trial_idx}: FAIL'

        # log to world log
        with open(world_log_path, 'a') as f:
            f.write(status_str + '\n')

        # log env results to trial log
        with open(trial_log_path, 'a') as wf:
            wf.write(f'\n#####\n\nEnvironment #{z}:\n{str(final_env_history)}\n\nSTATUS: {"OK" if is_success else "FAIL"}\n\n#####\n')

    # close environment object
    env.close()

    # log trial results to trial and world logs
    log_str: str = f"""
-----
SUCCESS: {num_successes}
ADDITIONAL SUCCESS: {num_additional_successes}
FAIL: {num_envs - num_successes}
TOTAL: {num_envs}
ACCURACY: {round(num_successes / num_envs, 2)}
-----"""
    with open(trial_log_path, 'a') as wf:
        wf.write(log_str)
    with open(world_log_path, 'a') as wf:
        wf.write(log_str + '\n')

    return env_configs
