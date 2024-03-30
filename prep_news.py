import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default='5')
parser.add_argument("--start_ymd", type=str, default='2023-8-1')  # inclusive
parser.add_argument("--end_ymd", type=str, default='2023-8-31')  # inclusive
args = parser.parse_args()
# args = parser.parse_args('--gpu 0 --start_ymd 2023-8-1 --end_ymd 2023-8-31'.split(' '))
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import json
import re
from datetime import datetime
from transformers import pipeline, set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
from datetime import date, timedelta


RAW_DIR = 'data/gnews_raw'  # 1216 articles in 2024-01
OUTPUT_DIR = 'data/gnews'
NEWS_TIME_FMT = "%a, %d %b %Y %H:%M:%S %Z"
MAX_TOKENS = 256
MIN_TOKENS = 128
WORD_PER_TOKEN = 0.75
CONTEXT_LENGTH = 4096
# DTYPE = torch.bfloat16  # half vram
DTYPE = torch.float16
# DTYPE = torch.float32  # full vram
# MODEL_ID = "gpt-3.5-turbo"  # openai api call
# MODEL_ID = "meta-llama/Llama-2-13b-chat-hf"  # fp32: 18G*3, 14min/news; fp16: 14G*2, 3.5min/news
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"  # fp32: 16G*2, 5.2min/news; fp16: 16G*1, 1min/news
# MODEL_ID = "gg-hf/gemma-7b-it"  # fp16: 20G*1, 2min/news
# MODEL_ID = "gg-hf/gemma-2b-it"  # fp16: 10G*1, 0.3min/news
# MODEL_ID = "facebook/bart-large-cnn"
# ID = 29  # DEBUG

if 'Llama' in MODEL_ID or 'gemma' in MODEL_ID:
    TASK = 'gen'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    generator = pipeline(task='text-generation', model=MODEL_ID, device_map="auto", max_new_tokens=MAX_TOKENS, torch_dtype=DTYPE)
elif 'bart' in MODEL_ID:
    TASK = 'sum'
    summarizer = pipeline("summarization", model=MODEL_ID, device="cuda")


def get_generation(raw_prompt):
    chat = [
        { "role": "user", "content": raw_prompt},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    prompt_tokenized = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)
    if len(prompt_tokenized) > CONTEXT_LENGTH - MAX_TOKENS:
        print(f"Prompt too long: {len(prompt_tokenized)} tokens. Skip news.")
        return None
    result = generator(prompt)
    response = result[0]['generated_text']
    if response.startswith(prompt):
        response = response[len(prompt):]
    return response


def get_summary(raw_prompt):
    result = summarizer(raw_prompt, max_length=MAX_TOKENS, min_length=MIN_TOKENS, do_sample=False)
    response = result[0]['summary_text']
    return response


# 5 W's and H: Who? What? When? Where? Why? How?
# news summary including 5 W's and H, sentiment => a structured news
# e.g. Where: America; What: Apple releases VisionPro; Sentiment: positive.
def format_news(item):
    if TASK == 'gen':
        # prompt = f'Summarize the following financial news in less than {int(MAX_TOKENS * WORD_PER_TOKEN)} words. Title: {item["title"]}. Content: {item["content"]}.'
        prompt = f'Summarize the following financial news in less than {MAX_TOKENS} tokens. Title: {item["title"]}. Content: {item["content"]}.'
        result = get_generation(prompt)
        if result is None:
            return None
        summary = {'title': item['title'], 'summary': result}
        return summary
    
    if TASK == 'sum':
        prompt = f'Title: {item["title"]}. Content: {item["content"]}.'
        result = get_summary(prompt)
        summary = {'title': item['title'], 'summary': result}
        return summary


def get_raw_file_names(start_ymd, end_ymd):
    start_date = datetime.strptime(start_ymd, '%Y-%m-%d')
    end_date = datetime.strptime(end_ymd, '%Y-%m-%d')
    delta = end_date - start_date
    raw_file_names = []
    for i in range(delta.days + 1):
        date = start_date + timedelta(days=i)
        year, month, day = date.year, date.month, date.day
        raw_file_name = f"{year}-{month}-{day}.json"
        raw_file_names.append(raw_file_name)
    # start_ymd = args.start_ymd
    # end_ymd = args.end_ymd
    # start_year, start_month, start_day = map(int, start_ymd.split('-'))
    # end_year, end_month, end_day = map(int, end_ymd.split('-'))
    # raw_file_names = []
    # for year in range(start_year, end_year + 1):
    #     for month in range(1, 13):
    #         for day in range(1, 32):
    #             if year == start_year and month < start_month:
    #                 continue
    #             if year == end_year and month > end_month:
    #                 continue
    #             if year == start_year and month == start_month and day < start_day:
    #                 continue
    #             if year == end_year and month == end_month and day > end_day:
    #                 continue
    #             if month in [4, 6, 9, 11] and day == 31:
    #                 continue
    #             if month == 2 and day > 28:  # missing leap year
    #                 continue
    #             raw_file_name = f"{year}-{month}-{day}.json"
    #             raw_file_names.append(raw_file_name)
    return raw_file_names


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    news_cnt = 0
    raw_file_names = get_raw_file_names(args.start_ymd, args.end_ymd)
    tick = time.time()
    for file_name in raw_file_names:
        file_path = os.path.join(RAW_DIR, file_name)
        with open(file_path, 'r') as f:
            data = json.load(f)
        date = file_path.split('/')[-1].split('.')[0]
        year, month, day = date.split('-')
        year, month, day = int(year), int(month), int(day)
        output_data = []
        for item in sorted(data, key=lambda x: x['id']):
            ID = item['id']
            time_str = item['time']
            title = item['title']
            content = item['content']
            parsed_time = datetime.strptime(time_str, NEWS_TIME_FMT)
            item_day = parsed_time.day
            if item_day != day:
                continue
            # if ID != 29: continue  # DEBUG

            news_cnt += 1
            tock = time.time()
            print(f"Processing news #{news_cnt}. File: {file_name}. ID: {ID}. Last time: {tock - tick:.2f}s.")
            tick = tock

            output_item = format_news(item)
            if output_item is not None:
                output_data.append(output_item)
            # break  # DEBUG

        output_file_name = f"{year}-{month:02d}-{day:02d}.json"
        output_file_path = os.path.join(OUTPUT_DIR, output_file_name)
        with open(output_file_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        # break  # DEBUG
