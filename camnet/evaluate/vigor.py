import time
import torch
import numpy as np
from tqdm import tqdm
import gc
import copy
from ..trainer import predict
import torch.nn.functional as F
def evaluate(config,
             model,
             reference_dataloader,
             query_dataloader, 
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    
    
    print("\nExtract Features:")
    reference_features, reference_labels = predict(config, model, reference_dataloader) 
    query_features, query_labels = predict(config, model, query_dataloader)
    
    print("Compute Scores:")
    r1 =  calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=step_size, ranks=ranks) 
    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()
        
    return r1


def evaluate_triple(config,
             model_1,
             model_2,
             model_3,
             reference_dataloader1,
             reference_dataloader2,
             reference_dataloader3,
             query_dataloader1, 
             query_dataloader2, 
             query_dataloader3,
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    
    
    print("\nExtract Features:")
    reference_features1, reference_labels1 = predict(config, model_1, reference_dataloader1) 
    query_features1, query_labels1 = predict(config, model_1, query_dataloader1)
    reference_features2, reference_labels2 = predict(config, model_2, reference_dataloader2) 
    query_features2, query_labels2 = predict(config, model_2, query_dataloader2) 
    reference_features3, reference_labels3 = predict(config, model_3, reference_dataloader3) 
    query_features3, query_labels3 = predict(config, model_3, query_dataloader3) 
    print("Compute Scores:")
    r1 =  calculate_scores_triple(query_features1, reference_features1, reference_features2, query_features2, reference_features3, query_features3, query_labels1, reference_labels1, step_size=step_size, ranks=ranks) 
        
    # cleanup and free memory on GPU
    if cleanup:
        del reference_features1, reference_features2, reference_features3, reference_labels1, reference_labels2, reference_labels3, query_features1, query_features2, query_features3, query_labels1, query_labels2, query_labels3
        gc.collect()
        
    return r1

def calc_sim(config,
                        model,
                        reference_dataloader,
                        query_dataloader, 
                        ranks=[1, 5, 10],
                        step_size=1000,
                        cleanup=True):
    
    
    print("\nExtract Features:")
    reference_features, reference_labels = predict(config, model, reference_dataloader) 
    query_features, query_labels = predict(config, model, query_dataloader)
    
    print("Compute Scores Train:")
    r1 =  calculate_scores_train(query_features, reference_features, query_labels, reference_labels, step_size=step_size, ranks=ranks) 
    
    near_dict = calculate_nearest(query_features=query_features,
                                  reference_features=reference_features,
                                  query_labels=query_labels,
                                  reference_labels=reference_labels,
                                  neighbour_range=config.neighbour_range,
                                  step_size=step_size)
            
    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()
        
    return r1, near_dict



def calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=1000, ranks=[1,5,10]):

    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features)
    
    steps = Q // step_size + 1
    
    
    query_labels_np = query_labels.cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()
    
    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i
    
    
    similarity = []
    
    for i in range(steps):
        
        start = step_size * i
        
        end = start + step_size
          
        sim_tmp = query_features[start:end] @ reference_features.T
        
        similarity.append(sim_tmp.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)
    

    topk.append(R//100)
    
    results = np.zeros([len(topk)])
    
    hit_rate = 0.0
    
    bar = tqdm(range(Q))
    
    for i in bar:
        
        # similiarity value of gt reference
        gt_sim = similarity[i, ref2index[query_labels_np[i][0]]]
        
        # number of references with higher similiarity as gt
        higher_sim = similarity[i,:] > gt_sim
        
         
        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.
                        
        # mask for semi pos
        mask = torch.ones(R)
        for near_pos in query_labels_np[i][1:]:
            mask[ref2index[near_pos]] = 0
        
        # calculate hit rate
        hit = (higher_sim * mask).sum()
        if hit < 1:
            hit_rate += 1.0
                
    
    results = results/ Q * 100.
    hit_rate = hit_rate / Q * 100
    
    bar.close()
    
    # wait to close pbar
    time.sleep(0.1)
    
    string = []
    for i in range(len(topk)-1):
        
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
        
    string.append('Recall@top1: {:.4f}'.format(results[-1]))
    string.append('Hit_Rate: {:.4f}'.format(hit_rate))             
        
    print(' - '.join(string)) 

    return results[0]


def dirichlet_entropy_reliability(S):
    """
    计算 Dirichlet + 熵驱动的可靠性
    S: (batch, R) 相似度矩阵
    返回: (batch,) 每个 query 的 reliability
    """
    alpha = F.softplus(S) + 1.0         # Dirichlet concentration
    p = alpha / alpha.sum(dim=1, keepdim=True)  # Dirichlet mean
    H = -torch.sum(p * torch.log(p + 1e-8), dim=1) / np.log(S.size(1))  # 归一化熵
    reliability = 1.0 - H               # 低熵 → 高可靠性
    return reliability


def calculate_scores_triple(query_features1, reference_features1, reference_features2, 
                           query_features2, reference_features3, query_features3,
                           query_labels, reference_labels, step_size=1000, ranks=[1, 5, 10],
                                              T=1):
    device = query_features1.device
    # 获取参考特征总数并初始化topk指标
    R = len(reference_features1)
    topk = copy.deepcopy(ranks)
    topk.append(R // 100 if R > 0 else 0)  # 添加top 1%指标
    
    # 获取查询样本数量并处理边缘情况
    Q = len(query_features1)
    if Q == 0:
        return 0.0
    
    # 确保批次处理批次大小
    batch_size = min(step_size, Q)
    
    # 转换标签格式并创建参考标签映射
    query_labels_np = query_labels.cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()
    ref2index = {idx: i for i, idx in enumerate(reference_labels_np)}
    
    # 初始化评估指标
    # results = np.zeros([len(topk)])
    results = torch.zeros(len(topk), device=device)

    all_weights = torch.zeros((Q, 3), device=device)
    hit_rate = 0.0
    
    # 分批次处理查询特征
    bar = tqdm(range(0, Q, batch_size), desc="Calculating similarities")
    for start in bar:
        end = min(start + batch_size, Q)
        batch_size_actual = end - start
        
        # 计算三种特征的相似度并融合
        S1 = query_features1[start:end] @ reference_features1.T
        S2 = query_features2[start:end] @ reference_features2.T
        S3 = query_features3[start:end] @ reference_features3.T
        
        # reliability
        R1 = dirichlet_entropy_reliability(S1)
        R2 = dirichlet_entropy_reliability(S2)
        R3 = dirichlet_entropy_reliability(S3)

        R = torch.stack([R1, R2, R3], dim=1)

        # 数值稳定 softmax
        R_max, _ = torch.max(R / T, dim=1, keepdim=True)
        exp_R = torch.exp(R / T - R_max)
        weights = exp_R / exp_R.sum(dim=1, keepdim=True)
        all_weights[start:end, :] = weights

        # 融合相似度
        # S_fused = weights[:,0:1]*S1 + weights[:,1:2]*S2 + weights[:,2:3]*S3
        S_fused = S1 + S2 + S3
        # 融合相似度（可根据需要调整权重，这里使用简单相加）
        # batch_similarity = sim1 + sim2 + sim3
        
        # 处理批次中的每个查询
        for i in range(batch_size_actual):
            global_idx = start + i
            query_label = query_labels_np[global_idx]
            
            # 获取真实标签并检查有效性
            gt_label = query_label[0]
            if gt_label not in ref2index:
                continue
            gt_idx = ref2index[gt_label]
            gt_sim = S_fused[i, gt_idx]
            
            # 计算排名
            higher_sim = S_fused[i, :] > gt_sim
            ranking = higher_sim.sum().item()
            
            # 更新Recall@k结果
            for j, k in enumerate(topk):
                if ranking < k:
                    results[j] += 1.
            
            # 处理半正样本掩码
            mask = torch.ones(R, device=S_fused.device)
            for semi_pos_label in query_label[1:]:
                if semi_pos_label in ref2index:
                    mask[ref2index[semi_pos_label]] = 0
            
            # 计算命中率
            hit = (higher_sim * mask).sum().item()
            if hit < 1:
                hit_rate += 1.0
    
    # 计算百分比结果
    results = results / Q * 100. if Q > 0 else 0
    hit_rate = hit_rate / Q * 100. if Q > 0 else 0
    
    # 输出结果
    bar.close()
    time.sleep(0.1)
    
    string = []
    for i in range(len(topk)-1):
        string.append(f'Recall@{topk[i]}: {results[i]:.4f}')
    string.append(f'Recall@top1%: {results[-1]:.4f}')
    string.append(f'Hit_Rate: {hit_rate:.4f}')
    
    print(' - '.join(string))
    return results[0]



def calculate_scores_train(query_features, reference_features, query_labels, reference_labels, step_size=1000, ranks=[1,5,10]):

    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features)
    
    steps = Q // step_size + 1
    
    query_labels_np = query_labels[:,0].cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()
    
    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i
    
    similarity = []
    
    for i in range(steps):
        
        start = step_size * i
        
        end = start + step_size
          
        sim_tmp = query_features[start:end] @ reference_features.T
        
        similarity.append(sim_tmp.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)

    topk.append(R//100)
    
    results = np.zeros([len(topk)])
    
    bar = tqdm(range(Q))
    
    for i in bar:
        
        # similiarity value of gt reference
        gt_sim = similarity[i, ref2index[query_labels_np[i]]]
        
        # number of references with higher similiarity as gt
        higher_sim = similarity[i,:] > gt_sim
         
        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.
        
    results = results/ Q * 100.

    bar.close()
    
    # wait to close pbar
    time.sleep(0.1)
    
    string = []
    for i in range(len(topk)-1):
        
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
        
    string.append('Recall@top1: {:.4f}'.format(results[-1]))           
        
    print(' - '.join(string)) 

    return results[0]
   

def calculate_nearest(query_features, reference_features, query_labels, reference_labels, neighbour_range=64, step_size=1000):

    query_labels = query_labels[:,0]
    
    Q = len(query_features)
    
    steps = Q // step_size + 1
    
    similarity = []
    
    for i in range(steps):
        
        start = step_size * i
        
        end = start + step_size
          
        sim_tmp = query_features[start:end] @ reference_features.T
        
        similarity.append(sim_tmp.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)


    # there might be more ground views for same sat view
    topk_scores, topk_ids = torch.topk(similarity, k=neighbour_range+2, dim=1)


    topk_references = []
    
    for i in range(len(topk_ids)):
        topk_references.append(reference_labels[topk_ids[i,:]])
    
    topk_references = torch.stack(topk_references, dim=0)

     
    # mask for ids without gt hits
    mask = topk_references != query_labels.unsqueeze(1)
    
    
    topk_references = topk_references.cpu().numpy()
    mask = mask.cpu().numpy()
    

    # dict that only stores ids where similiarity higher than the lowes gt hit score
    nearest_dict = dict()
    
    for i in range(len(topk_references)):
        
        nearest = topk_references[i][mask[i]][:neighbour_range]
    
        nearest_dict[query_labels[i].item()] = list(nearest)
    

    return nearest_dict
