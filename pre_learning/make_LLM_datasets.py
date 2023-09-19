#coding:UTF-8
from pathlib import Path
import json
from copy import deepcopy

def gen_dialog(file_name, output_dir = None):
    """单论对话制作，一轮对话保存在一个json文件"""
    print(f"开启当前对话 {file_name}.json")
    path = Path(output_dir).resolve()
    # 替换下面的一行代码，防止文件重名
    # f_num = len(list(path.glob(f'{file_name}.json')))
    f_num = 0
    path = path / f"{file_name}.json" if f_num == 0 else path / f"{file_name}_{f_num+1}.json"

    datasets = []
    single_dialog = {
        "prompt":"",
        "response":"",
        "history":[]
    }
    # 写入文件
    with path.open("a+", encoding="utf8") as file:
        while True:
            prompt = single_dialog["prompt"]
            response = single_dialog["response"]
            if prompt != "" and response !="":
                single_dialog["history"].append([prompt, response])
            single_dialog["prompt"] = input("问题：").strip()
            single_dialog["response"] = input("回答：").strip()
            if single_dialog["prompt"] != "" and single_dialog["response"] !="":
                json.dump(single_dialog, file, ensure_ascii=False)
                file.write('\n')
            datasets.append(deepcopy(single_dialog))

            keep = int(input("操作：(0-结束程序， 1-继续当前对话，2-重置对话 )：").strip())
            if keep == 2:
                print(f"当前对话结束，开起新对话~~")
                single_dialog["history"] = []
            elif keep == 0:
                file.close()
                return datasets


if __name__ == '__main__':
    # 写入文件
    path = Path(f"datasets/train")
    path.mkdir(exist_ok=True, parents=True)
    print(gen_dialog("xhao", path))
