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
        cur_step += 1
        time.sleep(1)
    total_return = state['roi']
    returns = np.array(returns) * 100
    return_mean = np.mean(returns)
    return_std = np.std(returns)
    risk_free_rate = 0  # as same sociodojo
    sharpe_ratio = (return_mean - risk_free_rate) / return_std
    print(f'FINAL return: {total_return*100:.2f}, sharpe ratio: {sharpe_ratio:.2f}')
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

#         base_prompt = f"You are now stepping into the shoes of a seasoned Ether (ETH) trader, embarking on a virtual trading challenge. Your goal is straightforward but challenging: maximize your profits from trading ETH over a one-month simulation period, starting from January 1, 2024, to January 31, 2024. This simulation is designed to test your analytical skills, market intuition, and strategic decision-making. Here’s what you need to know:\
# 1. **Starting Capital:** You begin with a ${info['starting_cash']} cash reserve. Your mission is to grow this initial investment by wisely trading Ether. Success is measured by the total value of your cash and ETH holdings at the end of the simulation.\
# 2. **Daily Market Data & News:** Every day, you will receive vital market data including ETH's price movements, market cap, and total volume. Additionally, a curated list of news summaries will provide insights into the crypto market's sentiment and potential ETH price drivers.\
# 3. **Transaction Fees:** Each buy or sell transaction incurs a fee, calculated as a percentage of the transaction value. This simulates real-world trading conditions where fees can impact your trading strategy and profitability.\
# **Decision-Making Criteria:**\
# - **Analyzing Market Data with a Nuanced Approach:**\
#   - Explore various strategies, including the moving averages MA5 and MA20. A buy signal is indicated when MA5 crosses below MA20 and subsequently begins to rise, suggesting an upward trend in ETH’s price. Conversely, a sell signal is suggested when MA5 crosses above MA20 and starts to decline, indicating a potential price drop. It's crucial to utilize these signals as part of a broader analysis, combining them with other market indicators for a well-rounded decision.\
#   - Instead of defaulting to common responses, you're encouraged to analyze the market data deeply and consider the full spectrum of investment actions. For instance, subtle market movements might warrant precise adjustments in your position, such as investments or sales representing 15%, 35%, or even 85% of your assets, reflecting a granular approach to trading based on nuanced market interpretations.\
#   - An upward trend in ETH’s price not only suggests a buying opportunity but calls for a tailored decision on how much to invest. Rather than broad strokes like 50% or 100%, consider the exactitude of your confidence in the trend—might 22%, 47%, or 76% of your cash reserve be more appropriate? Similarly, in a downward trend, decide on the precise portion of your ETH holdings to sell—could 18%, 33%, or 88% better reflect your strategic outlook and risk assessment?\
# - **Interpreting News:**\
#   - Positive news could bolster confidence in ETH, suggesting a buy signal. The extent of your investment should reflect the perceived impact of the news.\
#   - Negative news might raise red flags, hinting at a sell action. The proportion of ETH you decide to sell should align with the news' potential negative impact on ETH’s price.\
# - **Neutral Signals:** In cases of mixed signals or uncertainty, maintaining your position (hold) or making minor adjustments might be prudent.\
# **Your Daily Decision:**\
# Every day, you will decide whether to 'buy', 'sell', or 'hold' based on your analysis. Your decision must be quantified as follows:\
# - **To Buy:** Specify a positive decimal fraction of your remaining cash to invest in ETH, reflecting your confidence and strategy (e.g., 0.25 for 25%).\
# - **To Sell:** Indicate a negative decimal fraction of your ETH holdings you wish to sell, capturing your strategic decision (e.g., -0.40 for 40%).\
# - **To Hold:** A decision to hold requires no action but is an active strategy based on your market analysis.\
# **Precision in Decision:** Ensure your decision is presented as a two-decimal value within the [-1, 1] range. This precision reflects the nuanced analysis and strategic thought behind your decision."
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
