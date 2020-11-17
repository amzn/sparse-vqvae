from models.vqvae import Encoder
from models.model_utils import get_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn as nn

import os
print(os.path.abspath("."))

train_dataset, _ = get_dataset('cifar10', '../../../data', 32)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)


i = 0
for batch in tqdm(train_loader, desc='Extracting unlearned encodings'):
    encoder = Encoder(in_channel=3, channel=128, n_res_block=2, n_res_channel=32, stride=2)
    quantize_conv = nn.Conv2d(128, 64, 1)
    enc = encoder(batch[0])
    quant = quantize_conv(enc).permute(0, 2, 3, 1)
    np.save('unlearned_encodings/unlearned_cifar10_{}'.format(i), quant.detach().numpy())

    del encoder
    del quantize_conv
    i += 1
    a = 5