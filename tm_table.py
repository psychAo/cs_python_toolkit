import time
import torch
import numpy as np
from terminaltables import AsciiTable
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

if __name__ == '__main__':
    # constant
    EPOCH = 20
    BATCH_SIZE = 35

    # random data
    images = np.random.uniform(low=0, high=1, size=[500, 32, 32, 3])
    labels = np.random.randint(low=0, high=5, size=500)

    images = torch.tensor(images)
    labels = torch.tensor(labels)

    # data loader
    train_set = TensorDataset(images, labels)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    # 'training'
    for epoch in range(EPOCH):
        for i, batch in enumerate(train_loader):
            images, labels = batch
            # random loss for every iteration
            loss = np.abs(np.random.randn())
            # other metrics for every iteration
            metric0 = np.abs(np.random.randn())
            metric1 = np.abs(np.random.randn())
            metric2 = np.abs(np.random.randn())
            metric3 = np.abs(np.random.randn())
            time.sleep(1)  # simulate training time
            # print information
            table_data = [
                ['Metric', 'Value'], ['loss', np.round(loss, 6)],
                ['metric0', np.round(metric0, 6)], ['metric1', np.round(metric1, 6)],
                ['metric2', np.round(metric2, 6)], ['metric3', np.round(metric3, 6)]
            ]
            print('--- Epoch', epoch+1, '/', EPOCH, ',',
                  'Batch', i+1, '/', len(train_loader), '---')
            print(AsciiTable(table_data).table, '\n')

        # random accuracy for every epoch
        accuracy = np.random.uniform(low=0, high=1)
        print("Accuracy=", np.round(accuracy, 6), '\n')

