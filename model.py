import torch
import torch.nn as nn
from utils.decode import Decoder
from utils.ctc_loss import get_ctc_loss
from utils.kl_divergence_loss import get_kl_divergence_loss
from utils.evaluation import *
from reader import Reader
from tqdm import tqdm


class TemporalFusion(nn.Module):
    def __init__(self, conv_k, pool_k):
        super(TemporalFusion, self).__init__()
        self.conv = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=conv_k, padding='same')
        self.pooling = nn.MaxPool1d(kernel_size=pool_k, stride=pool_k, padding=int(pool_k / 2))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # c5-p2-c5-p2
        x = self.conv(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pooling(x)
        return x


class CSLR(nn.Module):
    def __init__(self, spatio_dim, num_classes, hidden_dim, decoder):
        super(CSLR, self).__init__()
        # backbone: resnet18
        self.conv2d = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.conv2d_fc = nn.Linear(1000, spatio_dim)
        self.conv1d = TemporalFusion(5, 2)
        self.conv1d_fc = nn.Linear(spatio_dim, num_classes)
        self.lstm = nn.LSTM(input_size=spatio_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim, num_classes)
        self.decoder = decoder

    def forward(self, videos, valid_lengths, phase):
        batch, max_len, C, H, W = videos.shape
        inputs = videos.reshape(batch * max_len, C, H, W)
        framewise_features = self.conv2d(inputs)
        framewise_features = self.conv2d_fc(framewise_features)

        # padding
        framewise_features = framewise_features.reshape(batch, max_len, -1)
        spatio_dim = framewise_features.shape[2]
        for i in range(batch):
            framewise_features[i, int(valid_lengths[i]):, :] = torch.zeros(
                ((max_len - valid_lengths[i]).item(), spatio_dim))
        framewise_features = framewise_features.permute(0, 2, 1)
        spatio_temporal = self.conv1d(framewise_features)
        spatio_temporal = spatio_temporal.permute(0, 2, 1)

        spatio_temporal_pred = self.conv1d_fc(spatio_temporal)

        # batch, len, dim

        # calculate valid length
        def v_len(l_in):
            return int((l_in + 2 * 1 - 2) / 2 + 1)

        valid_len = torch.Tensor([v_len(v_len(vlg)) for vlg in valid_lengths]).type(torch.int32)

        # mask for lstm
        packed_emb = nn.utils.rnn.pack_padded_sequence(spatio_temporal, valid_len, batch_first=True,
                                                       enforce_sorted=False)
        alignments, _ = self.lstm(packed_emb)
        alignments, _ = nn.utils.rnn.pad_packed_sequence(alignments, batch_first=True)

        alignments = self.fc(alignments)  # 有mask
        # batch, len , num_classes

        if phase == "predict":
            outputs = self.decoder.decode(alignments, valid_len)

            return outputs  # list of tensors

        if phase == "train":
            return alignments, spatio_temporal_pred, valid_len


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)


def train_model(model, optimizer, loss_type, mode, prefix, data_path, gloss_dict, epochs, batch, temperature, alpha,
                patience,
                save_path, device):
    model.train()
    model.to(device)

    # document losses:
    training_loss = []
    training_ctc_loss = []
    training_ve_loss = []
    training_va_loss = []

    dev_wer = []
    best_dev_wer = 100.0
    wait = 0

    training_data = Reader(prefix, data_path, mode, gloss_dict, batch)

    if loss_type == "VAC":
        criterion = nn.KLDivLoss(reduction='batchmean')
    for epoch in tqdm(range(epochs)):

        loss_epoch = []
        ctc_loss_epoch = []
        ve_loss_epoch = []
        va_loss_epoch = []

        model.train()

        for i in range(training_data.get_batch_numbers()):
            videos, valid_len, outputs, valid_output_len = next(training_data.iterate())
            # print(torch.max(outputs, dim=2)[1])
            videos, valid_len, outputs, valid_output_len = videos.to(device), valid_len.to(device), outputs.to(
                device), valid_output_len.to(device)
            alignments, spatio_temporal_pred, valid_len = model(videos, valid_len, 'train')
            if loss_type == "CTC":
                loss = get_ctc_loss(alignments, valid_len, outputs, valid_output_len)
                ctc_loss_epoch.append(loss.item())
            elif loss_type == 'VE':
                loss1 = get_ctc_loss(alignments, valid_len, outputs, valid_output_len)
                loss2 = get_ctc_loss(spatio_temporal_pred, valid_len, outputs, valid_output_len)
                loss = loss1 + loss2
                ctc_loss_epoch.append(loss1.item())
                ve_loss_epoch.append(loss2.item())
            elif loss_type == "VAC":
                loss1 = get_ctc_loss(alignments, valid_len, outputs, valid_output_len)
                loss2 = get_ctc_loss(spatio_temporal_pred, valid_len, outputs, valid_output_len)
                loss3 = get_kl_divergence_loss(criterion, spatio_temporal_pred, alignments, valid_len, device,
                                               temperature)
                loss = loss1 + loss2 + loss3 * alpha
                ctc_loss_epoch.append(loss1.item())
                ve_loss_epoch.append(loss2.item())
                va_loss_epoch.append(loss3.item())
            else:
                print("Loss type not valid. Choose from CTC/VE/VAC")
            # zero grad, backwards, step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.item())
        training_loss.append(sum(loss_epoch) / len(loss_epoch))
        training_ctc_loss.append(sum(ctc_loss_epoch) / len(ctc_loss_epoch))
        if loss_type == 'VE':
            training_ve_loss.append(sum(ve_loss_epoch) / len(ve_loss_epoch))
        if loss_type == 'VAC':
            training_ve_loss.append(sum(ve_loss_epoch) / len(ve_loss_epoch))
            training_va_loss.append(sum(va_loss_epoch) / len(va_loss_epoch))

        print("Epoch:" + str(epoch + 1))
        print("Average training loss: " + str(training_loss[-1]))
        print("Average ctc loss: " + str(training_ctc_loss[-1]))
        if loss_type == "VE":
            print("Average VE loss: " + str(training_ve_loss[-1]))
        if loss_type == "VAC":
            print("Average VE loss: " + str(training_ve_loss[-1]))
            print("Average VA loss: " + str(training_va_loss[-1]))

        optimizer.scheduler.step()  # epoch step

        # early stopping
        wer = evaluate(model, 'dev', prefix, data_path, gloss_dict, batch, device)
        print("Average dev wer: " + str(wer))
        if wer < best_dev_wer:
            best_dev_wer = wer
            dev_wer.append(wer)
            wait = 0
            # save the parameters of the current best model
            stat = model.state_dict()
        else:
            wait += 1
            if wait >= patience:
                dev_wer.append(wer)
                # save the best model
                torch.save(stat, save_path)
                print("stop at epoch " + str(epoch + 1))
                print(best_dev_wer)
                return training_loss, training_ctc_loss, training_ve_loss, training_va_loss, dev_wer
            else:
                dev_wer.append(wer)

        # evaluate on dev set, add to dev loss list
        # compare to previous / wait length

    torch.save(stat, save_path)
    print("stop at epoch " + str(epoch + 1))
    print(best_dev_wer)
    return training_loss, training_ctc_loss, training_ve_loss, training_va_loss, dev_wer


def evaluate(model, mode, prefix, data_path, gloss_dict, batch, device):
    model.to(device)
    model.eval()
    test_data = Reader(prefix, data_path, mode, gloss_dict, batch)

    total_distance, total_length = 0, 0
    for i in range(test_data.get_batch_numbers()):
        videos, valid_len, labels, valid_output_len = next(test_data.iterate())
        videos, valid_len, labels, valid_output_len = videos.to(device), valid_len.to(device), labels.to(
            device), valid_output_len.to(device)

        outputs = model(videos, valid_len, 'predict')
        wer, distance, length = batch_evaluation(outputs, labels, valid_output_len)
        total_length += length
        total_distance += distance
    return (total_distance / total_length).cpu().item()
