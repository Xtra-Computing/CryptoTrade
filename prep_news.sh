# nohup python -u prep_news.py --gpu 5 --start_ymd 2024-01-01 --end_ymd 2024-01-10 &>prep_news1.out 2>&1 &
# nohup python -u prep_news.py --gpu 6 --start_ymd 2024-01-11 --end_ymd 2024-01-20 &>prep_news2.out 2>&1 &
# nohup python -u prep_news.py --gpu 7 --start_ymd 2024-01-21 --end_ymd 2024-01-30 &>prep_news3.out 2>&1 &
nohup python -u prep_news.py --gpu 0 --start_ymd 2023-9-1 --end_ymd 2023-9-30 &>prep_news1.out 2>&1 &
nohup python -u prep_news.py --gpu 1 --start_ymd 2023-10-1 --end_ymd 2023-10-31 &>prep_news2.out 2>&1 &
nohup python -u prep_news.py --gpu 6 --start_ymd 2023-11-1 --end_ymd 2023-11-30 &>prep_news3.out 2>&1 &
nohup python -u prep_news.py --gpu 7 --start_ymd 2023-12-1 --end_ymd 2023-12-31 &>prep_news4.out 2>&1 &
