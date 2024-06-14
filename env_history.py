from typing import List, Dict


class EnvironmentHistory:
    def __init__(self, base_query: str, start_state, memory: List[str], history: List[Dict[str, str]], args) -> None:
        self.args = args
        self._cur_query: str = f'{_get_base_query(base_query, memory)}'
        self._history: List[Dict[str, str]] = history  # prompt, action, state, ...
        
        self.add('state', start_state)

    def add(self, label, value) -> None:
        self._history += [{
            'label': label,
            'value': value,
        }]

    def reset(self) -> None:
        self._history = []
    
    def get_prompt(self) -> str:
        price_window = self.args.price_window
        reflection_window = self.args.reflection_window
        use_tech = self.args.use_tech
        use_txnstat = self.args.use_txnstat
        delim = '\n"""\n'

        price_s = "You are an ETH cryptocurrency trading analyst. The recent price and auxiliary information is given in chronological order below:" + delim
        for i, item in enumerate(self._history[-price_window * 3:]):
            if item['label'] == 'state':
                state = item['value']
                state_log = f'Open price: {state["open"]:.2f}'
                if use_txnstat:
                    txnstat_dict = state['txnstat']
                    for k, v in txnstat_dict.items():
                        state_log += f', {k}: {v}'
                if use_tech:
                    tech_dict = state['technical']
                    for k, v in tech_dict.items():
                        state_log += f', {k}: {v}'
                price_s += state_log + '\n'
        # price_s += delim + 'Provide a brief report to analyze the recent information and estimate the market trend accordingly.'
        price_s += delim + 'Write one concise paragraph to analyze the recent information and estimate the market trend accordingly.'

        state = self._history[-1]['value']
        # news_s = f"You are an ETH cryptocurrency trading analyst. You are required to analyze the following news articles:{delim}{state['news']}{delim}Provide a brief report to analyze the news and estimate the market trend accordingly."
        news_s = f"You are an ETH cryptocurrency trading analyst. You are required to analyze the following news articles:{delim}{state['news']}{delim}Write one concise paragraph to analyze the news and estimate the market trend accordingly."

        reflection_s = 'You are an ETH cryptocurrency trading analyst. Your analysis and action history is given in chronological order:' + delim
        for i, item in enumerate(self._history[-reflection_window * 3:]):
            if item['label'] == 'trader_response':
                reflection_s += f'REASONING:\n{item["value"]}\n'
            elif item['label'] == 'action':
                reflection_s += f'ACTION:\n{item["value"]}\n'
            elif item['label'] == 'state':
                reflection_s += f'DAILY RETURN:\n{item["value"]["today_roi"]}\n'
        # reflection_s += delim + 'Reflect on your recent performance and instruct your future trades from a high level, e.g., identify what information is currently more important, and what to be next, like aggresive or conversative. Provide a brief reflection on your recent trading performance with a focus on the effective strategies and information that led to the most successful outcomes, and the ineffective strategies and information that led to loss of profit. Identify key trends and indicators in the current cryptocurrency market that are likely to influence future trades. Also assess whether a more aggressive or conservative trading approach is warranted.'
        reflection_s += delim + 'Reflect on your recent performance and instruct your future trades from a high level, e.g., identify what information is currently more important, and what to be next, like aggresive or conversative. Write one concise paragraph to reflect on your recent trading performance with a focus on the effective strategies and information that led to the most successful outcomes, and the ineffective strategies and information that led to loss of profit. Identify key trends and indicators in the current cryptocurrency market that are likely to influence future trades. Also assess whether a more aggressive or conservative trading approach is warranted.'

        base_prompt = 'You are an experienced ETH cryptocurrency trader and you are trying to maximize your overall profit by trading ETH. In each day, you will make an action to buy or sell ETH. You are assisted by a few analysts below and need to decide the final action.'
        # 'A small transaction fee is charged based on the transaction amount, so frequent large transactions are penalized.'
        # template_s = base_prompt + '\nON-CHAIN ANALYST REPORT: {}\n\n' + 'NEWS ANALYST REPORT: {}\n\n' + 'REFLECTION ANALYST REPORT: {}\n\n'
        template_s = f"{base_prompt}\n\nON-CHAIN ANALYST REPORT:{delim}{{}}{delim}\nNEWS ANALYST REPORT:{delim}{{}}{delim}\nREFLECTION ANALYST REPORT:{delim}{{}}{delim}\n"
        # template_s += 'Now, start your response with your brief reasoning over the given reports. Wisely select the useful information, conclude a clear market trend, pay more attention to long-term trends instead of short-term benefits, and avoid either being too conservative or changing strategies too rapidly. When the trend is upwards, it is profitable to buy ETH, and vice versa. Finally, suggest a 1-decimal float action in the range of [-1,1]. When the trend is bullish and upwards, you need to suggest a positive value to buy ETH. When the trend is bearish and downwards, you need to suggest a negative value to sell ETH. Your action is based on your prediction and confidence on the market trend, where a larger absolute value indicates a higher confidence on your trend prediction. Examples are: -0.3 (sell 30% ETH), 0.8 (buy with 80% cash), and so on. Make sure to directly give the value (could be negative) at the very end without any explanation.'
        template_s += 'Now, start your response with your brief reasoning over the given reports. Then, based on the synthesized reports, conclude a clear market trend, emphasizing long-term strategies over short-term gains. Finally, indicate your trading action as a 1-decimal float in the range of [-1,1], reflecting your confidence in the market trend and your strategic decision to manage risk appropriately.'

        return price_s, news_s, reflection_s, template_s

def _get_base_query(base_query: str, memory: List[str]) -> str:
    query = base_query

    # add memory if it exists
    if len(memory) > 0:
        query += '\n\nYour memory for the task below:'
        for i, m in enumerate(memory):
            query += f'\nTrial {i}:\n{m.strip()}'
    query += f"\nHere is the task:\n"
    return query
