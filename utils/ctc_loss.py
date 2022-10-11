import torch
import torch.nn as nn


def get_ctc_loss(alignments, valid_len, outputs, valid_output_len, blank_id=0):
    # alignments: batch, max_len, num_classes
    # outputs: batch, max_output_len, num_classes

    ctc_loss = nn.CTCLoss(blank=blank_id, reduction='mean', zero_infinity=True)
    log_probs = alignments.permute(1, 0, 2).log_softmax(2).requires_grad_()  # max_len, batch, num_classes 是否要detach
    _, target = torch.max(outputs, dim=2)
    _, pred = torch.max(log_probs.permute(1, 0, 2), dim=2)
    # print(pred)
    loss = ctc_loss(log_probs, target, valid_len, valid_output_len)
    return loss

