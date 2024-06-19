import json

json_file = "E:/E盘/模式识别课设/label/frame_0001.json"

with open(json_file, "r") as f:
    data = json.load(f)

print(data["shapes"])
