import pickle
import matplotlib.pyplot as plt


def plot_training_stat(model_name, stat_type):
    # stat_type: training_loss, training_ctc_loss, training_ve_loss, training_va_loss, dev_wer
    path = f'data/{model_name}_model/{model_name}_training_stat.pkl'
    with open(path, 'rb') as f:
        stats = pickle.load(f)

    stat = stats[stat_type]
    stop_epoch = stats['dev_wer'].index(min(stats['dev_wer']))
    # print(min(stats['dev_wer']))
    stat = stat[:stop_epoch+1]

    x = [i+1 for i in range(len(stat))]
    plt.plot(x, stat)
    plt.xlabel("Epochs")
    plt.ylabel(stat_type)
    plt.title(f'{model_name}_model')
    if len(x) <= 20:
        plt.xticks(range(0, max(x)+1, 1))
    plt.show()


if __name__ == "__main__":
    plot_training_stat('ve', 'dev_wer')