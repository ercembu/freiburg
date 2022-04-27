from lib.benchmark_api import Metric, NB201BenchmarkAPI, Dataset
from lib.plot import plot


if __name__ == '__main__':
    # Initialized the API
    api = NB201BenchmarkAPI('./benchmark/nb201_cifar10_full_training.pickle', Dataset.CIFAR10)

    # Randomly selected architecture
    # Try playing around with it!
    arch = '|nor_conv_1x1~0|+|none~0|nor_conv_1x1~1|+|none~0|nor_conv_3x3~1|nor_conv_1x1~2|'

    # Query the API for the learning curves.
    train_losses = api.query(arch, Metric.TRAIN_LOSS, last_epoch_only=False)
    train_accs = api.query(arch, Metric.TRAIN_ACCURACY, last_epoch_only=False)
    val_losses = api.query(arch, Metric.VAL_LOSS, last_epoch_only=False)
    val_accs = api.query(arch, Metric.VAL_ACCURACY, last_epoch_only=False)

    plot(train_accs, train_losses, val_accs, val_losses)
