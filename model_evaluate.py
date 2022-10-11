from model import *
import argparse
import random


def model_evaluate(model_path, data, device):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    prefix = 'phoenix2014_data/features/fullFrame-224x224px'
    decoder = Decoder(1296, gloss_dict='data/gloss_dict.pkl', search_mode='beam')
    model = CSLR(1024, 1296, 512, decoder)

    load_model(model, model_path)
    model.to(device)
    random.seed(1)  # keep evaluated performance static
    print(evaluate(model, data, prefix, 'data/data.pkl', 'data/gloss_dict.pkl', 2, device))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", default='./model.pt')
    parser.add_argument("--data", default='test', help='dev/test')
    parser.add_argument("--device", default='cuda')
    args = parser.parse_args()

    model_evaluate(args.model_path, args.data, args.device)
