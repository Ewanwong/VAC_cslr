import torch
import editdistance


def get_edit_distance(prediction, label, valid_len):
    # torch: len,
    label_length = valid_len

    edit_distance = editdistance.eval(prediction, label[:valid_len.item()])

    return edit_distance, label_length


def batch_evaluation(predictions, labels, valid_lengths):
    # predictions: list of tensor (1d)
    # outputs: batch, max_len, num_classes
    # valid_length: 1d tensor
    assert len(predictions) == labels.shape[0] and len(predictions) == valid_lengths.shape[0]
    batch = len(predictions)
    _, target = torch.max(labels, dim=2)
    total_distance = 0
    total_length = 0
    for i in range(batch):
        edit_distance, label_length = get_edit_distance(predictions[i].type(torch.int32).tolist(),
                                                        target[i, :].tolist(), valid_lengths[i])
        total_distance += edit_distance
        total_length += label_length
    wer = total_distance / total_length
    return wer, total_distance, total_length
