import os
import json
import argparse

from eth_trial import run_trial
from generate_reflections import update_memory

from typing import Any, List, Dict

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='eth', help="[eth, btc, sol]")
    parser.add_argument("--model", type=str, default='gpt-3.5-turbo', help="[gpt-3.5-turbo, gpt-4o, gpt-4-turbo]")
    parser.add_argument("--to_print", type=int, default=1, help="Print debug info")

    parser.add_argument("--starting_date", type=str, default='2023-08-01', help="The starting date for the environment")
    parser.add_argument("--ending_date", type=str, default='2024-02-01', help="The ending date for the environment")
    parser.add_argument("--price_window", type=int, default=7, help="look back window size for prompt")
    parser.add_argument("--reflection_window", type=int, default=3, help="look back window size for prompt")
    parser.add_argument("--seed", type=int, default=6216, help="llm seed")
    parser.add_argument("--use_memory", action='store_true', help="Allow the Agent to use memory for reflexion")
    parser.add_argument("--use_tech", type=int, default=1, help="Prompt +tech_signal")  # part of onchain analyst
    parser.add_argument("--use_txnstat", type=int, default=1, help="Prompt +txn_stat")  # part of onchain analyst
    parser.add_argument("--use_news", type=int, default=1, help="Prompt +news")
    parser.add_argument("--use_reflection", type=int, default=1, help="Prompt +reflection")

    # from reflexion, not used
    parser.add_argument("--resume_dir", type=str, help="If resume, the logging directory", default="")
    parser.add_argument("--start_trial_num", type=int, help="If resume, the start trial num", default=0)
    parser.add_argument("--num_trials", type=int, default=1, help="The number of trials to run")
    parser.add_argument("--num_envs", type=int, default=1, help="The number of environments per trial")
    parser.add_argument("--run_name", type=str, default='eth_run', help="The name of the run")
    parser.add_argument("--is_resume", action='store_true', help="To resume run")
    return parser

def main(args) -> None:
    print(args)

    if args.is_resume:
        if not os.path.exists(args.resume_dir):
            raise ValueError(f"Resume directory `{args.resume_dir}` does not exist")
        logging_dir = args.resume_dir

        # load environment configs
        env_config_path: str = os.path.join(args.resume_dir, f'env_results_trial_{args.start_trial_num - 1}.json')
        if not os.path.exists(env_config_path):
            raise ValueError(f"Environment config file `{env_config_path}` does not exist")
        with open(env_config_path, 'r') as rf:
            env_configs: List[Dict[str, Any]] = json.load(rf)
    else:
        # Create the run directory
        if not os.path.exists(args.run_name):
            os.makedirs(args.run_name)
        logging_dir = args.run_name

        # initialize environment configs
        env_configs: List[Dict[str, Any]] = []
        for i in range(args.num_envs):
            env_configs += [{
                'name': f'env_{i}',
                'memory': [],
                'is_success': False,
                'skip': False
            }]
    
    world_log_path: str = os.path.join(logging_dir, 'world.log')

    # run trials
    trial_idx = args.start_trial_num
    while trial_idx < args.num_trials:
        with open(world_log_path, 'a') as wf:
            wf.write(f'\n\n***** Start Trial #{trial_idx} *****\n\n')

        # set paths to log files
        trial_log_path: str = os.path.join(args.run_name, f'trial_{trial_idx}.log')
        trial_env_configs_log_path: str = os.path.join(args.run_name, f'env_results_trial_{trial_idx}.json')
        if os.path.exists(trial_log_path):
            open(trial_log_path, 'w').close()
        if os.path.exists(trial_env_configs_log_path):
            open(trial_env_configs_log_path, 'w').close()

        # run trial
        run_trial(trial_log_path, world_log_path, trial_idx, env_configs, args=args)

        # update memory if needed
        if args.use_memory:
            env_configs: List[Dict[str, Any]] = update_memory(trial_log_path, env_configs)

        # log env configs for trial
        with open(trial_env_configs_log_path, 'w') as wf:
            json.dump(env_configs, wf, indent=4)

        # log world for trial
        with open(world_log_path, 'a') as wf:
            wf.write(f'\n\n***** End Trial #{trial_idx} *****\n\n')

        trial_idx += 1


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
