# # Ablation
# python -u run_agent.py --dataset btc --model gpt-4o --use_tech 1 --use_txnstat 0 --use_news 1 --use_reflection 1 &>run_agent-wo-txnstat.out 2>&1
# python -u run_agent.py --dataset btc --model gpt-4o --use_tech 0 --use_txnstat 1 --use_news 1 --use_reflection 1 &>run_agent-wo-tech.out 2>&1
# python -u run_agent.py --dataset btc --model gpt-4o --use_tech 1 --use_txnstat 1 --use_news 1 --use_reflection 0 &>run_agent-wo-reflection.out 2>&1
# python -u run_agent.py --dataset btc --model gpt-4o --use_tech 1 --use_txnstat 1 --use_news 0 --use_reflection 1 &>run_agent-wo-news.out 2>&1

# # Sensitivity
# python -u run_agent.py --model gpt-4o --dataset eth &>4o-eth-bear.out 2>&1
# python -u run_agent.py --model gpt-4o --dataset eth &>4o-eth-sideways.out 2>&1
# python -u run_agent.py --model gpt-4o --dataset eth &>4o-eth-bull.out 2>&1
# python -u run_agent.py --model gpt-4-turbo --dataset eth &>4turbo-eth-bear.out 2>&1
# python -u run_agent.py --model gpt-4-turbo --dataset eth &>4turbo-eth-sideways.out 2>&1
# python -u run_agent.py --model gpt-4-turbo --dataset eth &>4turbo-eth-bull.out 2>&1

# Main results 4o
python -u run_agent.py --model gpt-4o --dataset btc --starting_date 2023-04-12 --ending_date 2023-06-16 &>logs/btc-bear-4o.out 2>&1
python -u run_agent.py --model gpt-4o --dataset btc --starting_date 2023-06-17 --ending_date 2023-08-25 &>logs/btc-sideways-4o.out 2>&1
python -u run_agent.py --model gpt-4o --dataset btc --starting_date 2023-10-01 --ending_date 2023-12-01 &>logs/btc-bull-4o.out 2>&1

python -u run_agent.py --model gpt-4o --dataset eth --starting_date 2023-04-12 --ending_date 2023-06-16 &>logs/eth-bear-4o.out 2>&1
python -u run_agent.py --model gpt-4o --dataset eth --starting_date 2023-06-17 --ending_date 2023-08-25 &>logs/eth-sideways-4o.out 2>&1
python -u run_agent.py --model gpt-4o --dataset eth --starting_date 2023-10-01 --ending_date 2023-12-01 &>logs/eth-bull-4o.out 2>&1

python -u run_agent.py --model gpt-4o --dataset sol --starting_date 2023-04-12 --ending_date 2023-06-16 &>logs/sol-bear-4o.out 2>&1
python -u run_agent.py --model gpt-4o --dataset sol --starting_date 2023-06-17 --ending_date 2023-08-25 &>logs/sol-sideways-4o.out 2>&1
python -u run_agent.py --model gpt-4o --dataset sol --starting_date 2023-10-01 --ending_date 2023-12-01 &>logs/sol-bull-4o.out 2>&1

# # Main results 3.5
# python -u run_agent.py --dataset btc --starting_date 2023-04-12 --ending_date 2023-06-16 &>logs/btc-bear.out 2>&1
# python -u run_agent.py --dataset btc --starting_date 2023-06-17 --ending_date 2023-08-25 &>logs/btc-sideways.out 2>&1
# python -u run_agent.py --dataset btc --starting_date 2023-10-01 --ending_date 2023-12-01 &>logs/btc-bull.out 2>&1

# python -u run_agent.py --dataset eth --starting_date 2023-04-12 --ending_date 2023-06-16 &>logs/eth-bear.out 2>&1
# python -u run_agent.py --dataset eth --starting_date 2023-06-20 --ending_date 2023-08-31 &>logs/eth-sideways.out 2>&1
# python -u run_agent.py --dataset eth --starting_date 2023-10-01 --ending_date 2023-12-01 &>logs/eth-bull.out 2>&1

# python -u run_agent.py --dataset sol --starting_date 2023-04-12 --ending_date 2023-06-16 &>logs/sol-bear.out 2>&1
# python -u run_agent.py --dataset sol --starting_date 2023-07-08 --ending_date 2023-08-31 &>logs/sol-sideways.out 2>&1
# python -u run_agent.py --dataset sol --starting_date 2023-10-01 --ending_date 2023-12-01 &>logs/sol-bull.out 2>&1
