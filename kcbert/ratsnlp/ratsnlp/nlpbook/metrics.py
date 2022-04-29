import torch


def accuracy(preds, labels, ignore_index=None):
    with torch.no_grad():
        assert preds.shape[0] == len(labels)
        correct = torch.sum(preds == labels)
        total = torch.sum(torch.ones_like(labels))
        if ignore_index is not None:
            # except ignore index from pred == labels
            correct -= torch.sum(torch.logical_and(preds == ignore_index, preds == labels))
            #except ignore index from denominator
            total -= torch.sum(labels == ignore_index)
    return correct.to(dtype=torch.float) / total.to(dtype=torch.float)
