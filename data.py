import os
import pickle
import requests
import random, json
import numpy as np
import yaml
from datasets import load_dataset, Dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer
import pandas as pd

datsets =["sprout", "routerbench", "leaderboard"]


input_costs = [3.0, 0.5, 2.5, 0.15, 0.1, 0.2, 0.9, 0.2, 0.06, 0.06, 0.9, 3.5,0.6]
output_costs = [15.0, 1.5, 10.0, 0.6, 0.1, 0.2, 0.9, 0.2, 0.06, 0.06, 0.9, 3.5, 0.6] 
input_costs = [x / 1000000 for x in input_costs]
output_costs = [x / 1000000 for x in output_costs]

leader_cost = [
    0.8, 0.6, 1.2, 0.9, 1.2, 0.3,1.2, 0.9, 0.8, 0.3, 0.1, 0.3, 0.9, 0.2, 0.2, 0.2, 0.6, 0.9
]
leader_cost = [x / 1000000 for x in leader_cost]

class data():
    def __init__(self, name="",  models=None):
        if name not in datsets:
            raise Exception("dataset not found.")
        self.dir = "./data"
        self.models = models
        if name == datsets[1]:
            filename ="routerbench_0shot.pkl"
            local_path = f"{self.dir}/{filename}"

            with open(local_path, "rb") as f:
                self.dataset = pickle.load(f)

            self.dataset = Dataset.from_pandas(self.dataset)
            self.op = 1

    def split(self, e_num=10000, b_num=26497):
        if self.op == 1:
            if not os.path.exists(f"./routerbench_data/"):
                def compute_field(example):
                    example["dataset"] = example["eval_name"]
                    return example
                dataset = self.dataset.map(compute_field) 
                indices = list(range(len(dataset)))
                dataset = dataset.add_column("index", indices)
                dataset.save_to_disk(f"./routerbench_data/")
            else:
                dataset = load_from_disk(f"./routerbench_data/")

            random.seed(42)
            all_indices = list(range(len(dataset)))
            sample_indices = random.sample(all_indices, k=10000)
            
            rest_indices = list(set(all_indices) - set(sample_indices))
            self.test = dataset.select(rest_indices)
            if e_num < 10000:
                sample_indices = random.sample(sample_indices, k=e_num)
            if b_num < 26497:
                rest_indices = random.sample(rest_indices, k=b_num)


            sampled_dataset = dataset.select(sample_indices)
            rest_dataset = dataset.select(rest_indices)
            
            cost = []
            for model in self.models:
                cost.append(np.sum(sampled_dataset[f"{model}|total_cost"]))
            min_cost = np.array(cost).min()
        
        return sampled_dataset, rest_dataset, sample_indices, rest_indices, min_cost




