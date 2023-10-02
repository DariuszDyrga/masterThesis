import torch
import torch.nn.functional as F


def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = 2
    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1).long(), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)
    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)


# def smooth_crossentropy(predictions, targets, smoothing=0.1):
#     one_hot_targets = torch.full_like(predictions, fill_value=smoothing).scatter_(dim=1, index=targets.unsqueeze(1).long(), value=1.0 - smoothing)
#     log_probs = F.log_softmax(predictions, dim=1)
#     loss = F.kl_div(input=log_probs, target=one_hot_targets, reduction='none').sum(-1)
#     return loss
