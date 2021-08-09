import os
import json

def build(path:str):
    dir_list = os.listdir(path)
    count = len(dir_list)
    result = {}
    for i in range (count):
        content = {}
        content['name'] = dir_list[i]
        content['uid'] = "class_" + format(i, '03d')
        result[i] = content
    json_content = json.dumps(result)
    with open('data.json', 'w') as outfile:
        json.dump(json_content, outfile)
    a=0


if __name__ == "__main__":
    build("C:\\animalsDataset")