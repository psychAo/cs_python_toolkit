from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time

if __name__ == '__main__':
    learning_rate = [1, 2, 3]

    for lr in learning_rate:
        writer = SummaryWriter()
        for i in range(1000):
            writer.add_scalar(tag='loss', scalar_value=np.random.uniform()+lr, global_step=i)
            writer.add_scalar(tag='accuracy', scalar_value=np.random.randn()+lr, global_step=i)
        time.sleep(5)  # avoid different events files being stored in same file folder
        writer.close()

# run
# tensorboard --logdir <absolute_path>/runs
# in terminal or cmd
