import torch
import torch.nn.functional as F
import numpy as np

epsilon = 1e-5
smooth = 1

def dice_array(y_true, y_pred):
    epsilon = np.array(1e-5)
    intersection = np.sum(np.logical_and(y_true, y_pred))
    return (2. * intersection + epsilon) / (np.sum(y_true) + np.sum(y_pred) + epsilon)


def dice_tensor(y_true, y_pred):
    tmp = torch.logical_and(y_true, y_pred)
    intersection = torch.sum(tmp.float())
    dice = (2. * intersection + epsilon) / (torch.sum(y_true.float()) + torch.sum(y_pred.float()) + epsilon)
    return dice


def dice_multi(y_true, y_pred):
    dice_value = 0.0
    n_labels = y_pred.size()[-1]
    prediction = torch.argmax(y_pred, -1)
    for i in range(n_labels):
        yi_true = y_true[..., i:i+1].bool()
        yi_pred = prediction.eq(i)
        dice_value += dice_tensor(yi_true, yi_pred)
    return dice_value / n_labels


def dice_multi_array(y_true, y_pred, labels):
    n_labels = len(labels)
    dice_value = np.zeros(n_labels, dtype=np.float32)
    for i in range(n_labels):
        yi_true = (y_true == labels[i])
        yi_pred = (y_pred == labels[i])
        dice_value[i] = dice_array(yi_true, yi_pred)
    return dice_value


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    intersection = torch.sum(torch.abs(y_true * y_pred))
    return (2. * intersection + smooth) / (torch.sum(y_true**2) + torch.sum(y_pred**2) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


def jaccard_distance_loss(y_true, y_pred, smooth=1e-5):
    intersection = torch.sum(torch.abs(y_true * y_pred))
    sum_ = torch.sum(torch.abs(y_true) + torch.abs(y_pred))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def tversky(y_true, y_pred):
    y_true_pos = y_true.view(-1)
    y_pred_pos = y_pred.view(-1)
    true_pos = torch.sum(y_true_pos * y_pred_pos)
    false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return (1 - pt_1)**gamma


def precision(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clamp(y_pred, 0, 1)))
    precision = true_positives / (possible_positives + epsilon)
    return precision


def recall(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
    predicted_positives = torch.sum(torch.round(torch.clamp(y_true, 0, 1)))
    recall = true_positives / (predicted_positives + epsilon)
    return recall


def f1_score(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clamp(y_pred, 0, 1)))
    precision = true_positives / (possible_positives + epsilon)
    true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
    predicted_positives = torch.sum(torch.round(torch.clamp(y_true, 0, 1)))
    recall = true_positives / (predicted_positives + epsilon)
    return 2 * ((precision * recall) / (precision + recall + epsilon))


def gen_dice_loss(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    BCE = F.binary_cross_entropy(y_pred, y_true)
    DICE_LOSS = dice_coefficient_loss(y_true, y_pred)

    return 0.5 * BCE + 0.5 * DICE_LOSS


def DiceBCELoss(y_true, y_pred):
    """
        y_true : (N, D, H, W, C)
        y_pred : (N, D, H, W, C)
    """
    n_labels = y_pred.size()[-1]
    loss = 0
    for c in range(n_labels):
        loss += gen_dice_loss(y_true[..., c], y_pred[..., c])
    return loss


def gen_dice_coefficient(y_true, y_pred):
    n_labels = y_pred.size()[-1]
    dice = 0
    for c in range(n_labels):
        dice += dice_coefficient(y_true[..., c], y_pred[..., c]) * 0.5
    return dice


def weighted_bce_loss(y_true, y_pred, weight):
    epsilon = torch.tensor(1e-7)
    y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = torch.log(y_pred / (1. - y_pred))

    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
        (torch.log(1. + torch.exp(-torch.abs(logit_y_pred))) +
         torch.maximum(-logit_y_pred, torch.tensor(0.)))
    return torch.sum(loss) / torch.sum(weight)


def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * torch.sum(w * intersection) + smooth) / \
        (torch.sum(w * m1) + torch.sum(w * m2) + smooth)
    loss = 1. - torch.sum(score)
    return loss


def weighted_bce_dice_loss(y_true, y_pred):
    y_true = y_true.float()
    y_pred = y_pred.float()
    averaged_mask = F.avg_pool3d(y_true, kernel_size=(11, 11), stride=(1, 1), padding='same')
    border = ((averaged_mask > 0.005) & (averaged_mask < 0.995)).float()
    weight = torch.ones_like(averaged_mask)
    w0 = torch.sum(weight)
    weight += border * 2