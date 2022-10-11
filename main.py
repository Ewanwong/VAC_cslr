import torch.cuda
import pickle
from model import *
from utils.decode import Decoder
from utils.optimizer import Optimizer
import argparse


# print(torch.cuda.is_available())

def main(loss_type, patience, save_path, stat_path, device):
    decoder = Decoder(1296, gloss_dict='data/gloss_dict.pkl', search_mode='beam')
    model = CSLR(1024, 1296, 512, decoder)
    optim_dict = {
        'base_lr': 1e-4,
        'step': [40, 60],
        'weight_decay': 1e-4
    }
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    optimizer = Optimizer(model, optim_dict)
    prefix = 'phoenix2014_data/features/fullFrame-224x224px'
    training_loss, training_ctc_loss, training_ve_loss, training_va_loss, dev_wer = train_model(model=model,
                                                                                                optimizer=optimizer,
                                                                                                loss_type=loss_type,
                                                                                                mode='train',
                                                                                                prefix=prefix,
                                                                                                data_path='data/data.pkl',
                                                                                                gloss_dict='gloss_dict.pkl',
                                                                                                epochs=80,
                                                                                                batch=2,
                                                                                                temperature=8,
                                                                                                alpha=25,
                                                                                                patience=patience,
                                                                                                save_path=save_path,
                                                                                                device=device)

    stat = {"training_loss": training_loss,
            "training_ctc_loss": training_ctc_loss,
            "training_ve_loss": training_ve_loss,
            "training_va_loss": training_va_loss,
            "dev_wer": dev_wer}
    with open(stat_path, 'wb') as f:
        pickle.dump(stat, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", type=str, default='VAC', help="CTC/VE/VAC")
    parser.add_argument("--patience", default=10)
    parser.add_argument("--save_path", default='./model.pt')
    parser.add_argument("--stat_path", default='./training_stat.pkl')
    parser.add_argument("--device", default='cuda')

    args = parser.parse_args()

    main(args.loss, args.patience, args.save_path, args.stat_path, args.device)
