"""供 main.py check-data 使用，复用逐步学习脚本。"""
from .read_sample import main


def check_data(data_root):
    main(data_root=data_root)
