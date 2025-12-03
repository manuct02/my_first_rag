import json

class DataLoader:

    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.data= json.load(f)
        
    def itir(self):
        for item in self.data:
            yield item["question"], item["expected_node_contains"]

