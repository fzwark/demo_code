from data import data
from ANN import ANN
from sentence_transformers import SentenceTransformer
from algo import Router
import yaml, os
import torch
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--eps', type=float, default=0.001)
parser.add_argument('--alpha', type=float, default=0.001)
parser.add_argument('--budget', type=float, default=0.5)
parser.add_argument('--N', type=int, default=10000)
parser.add_argument('--M', type=int, default=11)
parser.add_argument('--E', type=int, default=26497)
parser.add_argument('--segsize', type=int, default=256)
parser.add_argument('--dataset', type=int, default=1, choices=[0, 1, 2])
parser.add_argument('--ops', type=int, nargs='+', default=[1, 2, 3])
parser.add_argument('--split', type=str, default="weighted", choices=["weighted", "extreme", "random", "uniform", "cost", "perf"])
parser.add_argument('--embed', type=str, default="bge", choices=["bge", "gte", "sfr"])
parser.add_argument('--rdid', type=int, default=0)
parser.add_argument('--top_k', type=int, default=5)

args = parser.parse_args()

budget = args.budget
N = args.N
E = args.E
M = args.M
ops = args.ops
eps = args.eps
split= args.split
datset = args.dataset
segsize = args.segsize
embed = args.embed
alpha = args.alpha
rdid = args.rdid
top_k = args.top_k


with open("config.yaml", "r") as f:
    ans = yaml.safe_load(f)
models = ans [datset]

model_indices = []
if datset == 0:
    if M >= 13:
        M = 13
    elif M < 13:
        random.seed(42)
        for i in range(rdid + 1):
            model_indices = random.sample(range(13), k=M)
        new_models = []
        for indx, model in enumerate(models):
            if indx in model_indices:
                new_models.append(model)
        models = new_models

elif datset == 1:
    if M >= 11:
        M = 11 
    elif M < 11:
        random.seed(42)
        for i in range(rdid + 1):
            model_indices = random.sample(range(11), k=M)
        new_models = []
        for indx, model in enumerate(models):
            if indx in model_indices:
                new_models.append(model)
        models = new_models

elif datset == 2:
    if M >= 18:
        M = 18 
    elif M < 18:
        random.seed(42)
        for i in range(rdid + 1):
            model_indices = random.sample(range(18), k=M)
        new_models = []
        for indx, model in enumerate(models):
            if indx in model_indices:
                new_models.append(model)
        models = new_models
        

# prepare data 
dats =["sprout", "routerbench", "leaderboard"]
dat = data(name=dats[datset], models= models)
sampled, rested, sample_ind, rested_ind, min_cost = dat.split(e_num=N)


final_results = []
store_dir = "results/"
ans_name = f"{dats[datset]}_{alpha}_{eps}_{budget}_{N}_{E}_{split}_{embed}.json"
store_path = os.path.join(store_dir, ans_name)


budget = budget * min_cost

# load embedding
if not os.path.exists(f"embeddings/{embed}_embeddings{datset}.npy"):
    sampled_dataset = dat.dataset["prompt"]
    if embed == "bge":
        model = SentenceTransformer('BAAI/bge-base-en-v1.5', device='cuda')
    elif embed == "gte":
        model = SentenceTransformer('Alibaba-NLP/gte-Qwen2-1.5B-instruct', device='cuda')
    elif embed == "sfr":
        model = SentenceTransformer('Salesforce/SFR-Embedding-2_R', device='cuda')

    with torch.no_grad():
        embeds = model.encode(sampled_dataset, batch_size=64, convert_to_numpy=True, show_progress_bar=True)
    s_path = "embeddings" / f"{embed}_embeddings{datset}.npy"
    np.save(s_path, embeds)

else:
    embeds = np.load(f"embeddings/{embed}_embeddings{datset}.npy")


base_embeds = embeds[rested_ind]
sample_embeds = embeds[sample_ind]
embed_size = embeds.shape[1]
ann = ANN(method="hnsw", embed_size=embed_size,  sample_size=len(rested), top_k = top_k)
ann.add(base_embeds)

# generate weights for budget
B_list = []
weighted, co, perf, extreme, rd, uniform = budget_distribution(M, model_indices, datset)
if split == "weighted":
    weighted = np.array(weighted)
    B_list.append(weighted/sum(weighted)*budget)
elif split == "cost":
    co = np.array(co)
    B_list.append(co/sum(co)*budget)
elif split == "perf":
    perf = np.array(perf)
    B_list.append(perf/sum(perf)*budget)
elif split == "uniform":
    uniform = np.array(uniform)
    B_list.append(uniform/sum(uniform)*budget)
elif split == "extreme":
    for item in extreme:
        item = np.array(item)
        B_list.append(item/sum(item)*budget)
elif split == "random":
    for item in rd:
        item = np.array(item)
        B_list.append(item/sum(item)*budget)
    B_list = [B_list[rdid]]



for B in B_list:
    results = {}
    results["split"] = list(B)
    if 1 in ops:
        quality = 0.0
        cost = 0.0
        lost = 0.0
        router  = Router(ann=ann, eps=eps, M=M, length=N, base_data=rested, B=B, alpha=alpha, models=models)
        tg = np.zeros((M, N))
        for j, item in enumerate(sampled):
            index = item["index"]
            embed = embeds[index]
            ind = router.routing(query=embed)
            if ind < M:
                tg[ind, j] = item[f"{models[ind]}|total_cost"]
                if tg[ind, :].sum() > B[ind]:
                    tg[ind, j] = 0
                    lost += 1
                else:
                    quality += item[models[ind]]
                    cost += tg[ind, j]
            else:
                lost += 1

        results["ours"] = {
            "quality" : quality,
            "cost" : cost,
            "ratio" : quality /(cost ),
            "throughput": N - lost, 
            "acc" : quality/(N-lost),
        }
    print(f"Router finished, quality: {results['ours']['quality']}, cost: {results['ours']['cost']}, ratio: {results['ours']['ratio']}, throughput: {results['ours']['throughput']}")

    # greedy 
    if 2 in ops:
        quality, cost, lost = greedy(sampled, embeds=embeds, ann=ann, M=M, N=N, B=B, base_data=rested, op=4, models=models,)
        results["greedy_pref_results"] = {
            "quality" : quality,
            "cost" : cost,
            "ratio" : quality /(cost),
            "throughput": N - lost, 
        }
        quality, cost, lost = greedy(sampled, embeds=embeds, ann=ann, M=M, N=N, B=B, base_data=rested, op=2, models=models,)
        results["greedy_cost_results"] = {
            "quality" : quality,
            "cost" : cost,
            "ratio" : quality /(cost),
            "throughput": N - lost, 
        }


    # random
    if 3 in ops:
        quality, cost, lost = random_router(sampled, M=M, N=N, B=B, models=models)
        results["random_results"] = {
            "quality" : quality,
            "cost" : cost,
            "ratio" : quality /(cost),
            "throughput": N - lost, 
        }

    # KNN
    if 4 in ops:
        quality, cost, lost = knn(sampled, embeds=embeds, M=M, N=N, B=B, base_data=rested, base_embeds=base_embeds, op=1, models=models, top_k=top_k)
        results["knn_pref_results"] = {
            "quality" : quality,
            "cost" : cost,
            "ratio" : quality /(cost ),
            "throughput": N - lost, 
        }
        quality, cost, lost = knn(sampled, embeds=embeds, M=M, N=N, B=B, base_data=rested, base_embeds=base_embeds, op=2, models=models, top_k=top_k)
        results["knn_cost_results"] = {
            "quality" : quality,
            "cost" : cost,
            "ratio" : quality /(cost ),
            "throughput": N - lost, 
        }

    # batchsplit
    if 5 in ops:
        quality, cost, lost = segment_local(sampled, embeds, ann, M, N, B, models, rested, size=segsize)
        results["segment_optimize_results"] = {
            "quality" : quality,
            "cost" : cost,
            "ratio" : quality /(cost ),
            "throughput": N - lost, 
        }

    # robert-base 
    if 6 in ops:
        quality, cost, lost = roberta(sampled, M, N, B, models, datset, op=1)
        results["roberta_perf_results"] = {
            "quality" : quality,
            "cost" : cost,
            "ratio" : quality /(cost ),
            "throughput": N - lost, 
        }

        quality, cost, lost = roberta(sampled, M, N, B, models, datset, op=2)
        results["roberta_cost_results"] = {
            "quality" : quality,
            "cost" : cost,
            "ratio" : quality /(cost ),
            "throughput": N - lost, 
        }

    # est optimal
    if 7 in ops:
        _, pr_opt, cost, lost = est_optimal(sampled, embeds, ann, rested, M, N, B, models, show=False)
        results["offline_estimate_optimal_results"] = {
            "pr_opt" : pr_opt,
            "cost" : cost,
            "ratio" : pr_opt /(cost),
            "throughput": N - lost, 
        }

    # true optimal
    if 8 in ops:
        _, pr_opt, cost, lost = true_optimal(sampled, M=M, N=N, B=B, models=models, show=False)
        results["offline_true_optimal_results"] = {
            "pr_opt" : pr_opt,
            "cost" : cost,
            "ratio" : pr_opt /(cost),
            "throughput": N - lost, 
        }
    final_results.append(results)

with open(store_path, "w+") as f:
    json.dump(final_results, f)