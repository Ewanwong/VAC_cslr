import torch
import torch.nn as nn
import torch.nn.functional as F


def get_kl_divergence_loss(criterion, prediction_logits, ref_logits, valid_len, device, temperature=8):
    # logits: bacth, len, num_classes
    prediction_logits = F.log_softmax(prediction_logits / temperature, dim=2).requires_grad_()
    ref_probs = F.softmax(ref_logits / temperature, dim=2).requires_grad_()
    loss = torch.tensor([0.0]).to(device)
    # only take valid length
    for i in range(len(valid_len)):
        loss += criterion(prediction_logits[i, :valid_len[i], :], ref_probs[i, :valid_len[i], :]).item()
    loss /= len(valid_len)
    return loss
