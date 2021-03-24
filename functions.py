import torch


def gather_bn_weights(model):
    size = []
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            size.append(m.weight.data.shape[0])

    bn_weights = torch.zeros(sum(size))
    index = 0
    idx = 0
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            bn_weights[index:(index + size[idx])] = m.weight.data.abs().clone()
            index += size[idx]
            idx += 1
    return bn_weights


def gather_bn_weights_(module_list, prune_idx):
    size_list = [module_list[idx][1].weight.data.shape[0] for idx in prune_idx]

    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in zip(prune_idx, size_list):
        bn_weights[index:(index + size)] = module_list[idx][1].weight.data.abs().clone()
        index += size

    return bn_weights
