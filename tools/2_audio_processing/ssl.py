import numpy as np
import torch
from speechbrain_lite import *
from scipy.io import  loadmat
import  matplotlib.pyplot as plt

def inti_mics():
    """
    Initialize the microphone array coordinates in Euclidean coordinates.

    Augments
    --------
    rotation : int
        Use a rotation matrix to rotate the microphone array
        coordinates counterclockwise.

    Returns
    -------
    mics : Tensor
        Return the coordinates of (x, y, z).
    """
    mics = torch.zeros((6, 3))
    angle = torch.Tensor([0,60,120,180,-120,-180])/180*np.pi
    rou = 70.85/1000/2
    for i in range(len(mics)):
        mics[i,:] = torch.Tensor([rou*torch.cos(angle[i]),rou*torch.sin(angle[i]),0])
    return mics


def doa_detection(waveform, mics=None):
    """
    Using the SRP-PHAT to determine the direction of angles.
    Augments
    --------
    waveform : torch.Tensor
        the input shape is [batch, time_step, channel], and
        the number of channels should be at least 4.
    """
    if mics is None:
        mics = inti_mics()

    stft = STFT(sample_rate=16000)
    cov = Covariance()

    Xs = stft(waveform)
    XXs = cov(Xs)

    srpphat = SrpPhat(mics=mics, sample_rate=16000, speed_sound=346)
    doas =  srpphat(XXs)
    doas = doas[:, 0, :]

    doas[:, 2] = doas[:, 2].abs()
    doas[:, 1] = -doas[:, 1]

    r = (doas[:, 0]**2 + doas[:, 1]**2).sqrt()
    azi = torch.atan2(doas[:, 1], doas[:, 0])/np.pi*180
    ele = torch.atan2(doas[:, 2], r)/np.pi*180
    return torch.column_stack((azi, ele)),doas