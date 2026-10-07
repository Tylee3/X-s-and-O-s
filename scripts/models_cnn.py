"""
models_cnn.py -- Classifier 3: a small convolutional network for the 64x64 playground input.

Why a CNN: the MLP treats pixel (5,5) and pixel (5,6) as unrelated inputs; it has no idea
what "next to" means and must memorise every position a stroke can be in.  A convolution
slides ONE small filter over the whole image, computing the same weighted sum everywhere.
It learns "diagonal stroke" once and finds it anywhere: fewer weights, position tolerance.

Shapes (batch dimension omitted):
    input        1 x 64 x 64     playground tensor, ink dark, [-1,1]
    normalize    1 x 64 x 64     BackgroundNormalize (ink bright, background 0), then
                                 CentreScale (symbol centred, zoomed to a standard size)
    conv1+pool   16 x 32 x 32    16 filters of 3x3; max over 2x2 blocks halves the size
    conv2+pool   32 x 16 x 16
    conv3+pool   64 x 8 x 8      one cell now summarises an 8x8 patch of the original
    flatten      4096
    dropout      (training only)
    fc1          64
    fc2          2               logits (O, X)
"""
import torch.nn as nn

from models import Normalize, SIZE


class CNN(nn.Module):
    def __init__(self, size: int = SIZE, channels=(16, 32, 64), fc: int = 64, dropout: float = 0.3, centre: bool = True, power: int = 4):
        super().__init__()
        self.norm = Normalize(centre, power)
        blocks, c_in = [], 1
        for c_out in channels:
            blocks += [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        side = size // (2 ** len(channels))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(c_in * side * side, fc), nn.ReLU(),
            nn.Linear(fc, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(self.norm(x)))
