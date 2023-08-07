# coding: utf-8
# Project：windows.py
# Author：XHao
# Date ：2023-8-4 15:17
# Tool ：PyCharm

import json
from pathlib import Path
def gen_dialog(i):
    """单论对话制作"""
    single_dialog = {
        "prompt": "",
        "response": "",
        "history": []
    }
    keep = 1
    while keep == 1:
        question = input("问题：").strip()
        answer = input("回答：").strip()
        single_dialog["history"].append(question)
        single_dialog["history"].append(answer)
        keep = int(input("结束对话(0-结束，非0-继续)：").strip())
        if keep != 0:
            keep = 1
    end = single_dialog["history"].pop(-1)
    single_dialog["prompt"] = end[0]
    single_dialog["response"] = end[1]

    # 写入文件
    path = Path(f"datasets/train").resolve()
    path.mkdir(exist_ok=True, parents=True)
    path = path / f"{i}.json"
    with path.open("w", encoding="utf8") as file:
        json.dump(single_dialog, file, ensure_ascii=False)
    print(f"当前对话已保存至 {path}")


if __name__ == '__main__':
    gen_dialog(1)
