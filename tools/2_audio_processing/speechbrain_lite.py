# This file is based on modifications to the SpeechBrain.
# https://speechbrain.readthedocs.io/en/latest/API/speechbrain.processing.multi_mic.html#
import torch
import numpy as np
from scipy.io.wavfile import read
def doas2taus(doas, mics, fs, c=343.0):
    taus = (fs / c) * torch.matmul(doas.to(mics.device), mics.transpose(0, 1))
    return taus
def steering(taus, n_fft):
    pi = 3.141592653589793
    frame_size = int((n_fft - 1) * 2)

    # Computing the different parts of the steering vector
    omegas = 2 * pi * torch.arange(0, n_fft, device=taus.device) / frame_size
    omegas = omegas.repeat(taus.shape + (1,))
    taus = taus.unsqueeze(len(taus.shape)).repeat(
        (1,) * len(taus.shape) + (n_fft,)
    )

    # Assembling the steering vector
    a_re = torch.cos(-omegas * taus)
    a_im = torch.sin(-omegas * taus)
    a = torch.stack((a_re, a_im), len(a_re.shape))
    a = a.transpose(len(a.shape) - 3, len(a.shape) - 1).transpose(
        len(a.shape) - 3, len(a.shape) - 2
    )
    return a
def sphere(levels_count=4):
    """
    (n_points, 3)
    levels_count = 1, then the sphere will have 42 points
    levels_count = 2, then the sphere will have 162 points
    levels_count = 3, then the sphere will have 642 points
    levels_count = 4, then the sphere will have 2562 points
    levels_count = 5, then the sphere will have 10242 points
    """

    # points at level 0

    h = (5.0 ** 0.5) / 5.0
    r = (2.0 / 5.0) * (5.0 ** 0.5)
    pi = 3.141592654

    pts = torch.zeros((12, 3), dtype=torch.float)
    pts[0, :] = torch.FloatTensor([0, 0, 1])
    pts[11, :] = torch.FloatTensor([0, 0, -1])
    pts[range(1, 6), 0] = r * torch.sin(2.0 * pi * torch.arange(0, 5) / 5.0)
    pts[range(1, 6), 1] = r * torch.cos(2.0 * pi * torch.arange(0, 5) / 5.0)
    pts[range(1, 6), 2] = h
    pts[range(6, 11), 0] = (
        -1.0 * r * torch.sin(2.0 * pi * torch.arange(0, 5) / 5.0)
    )
    pts[range(6, 11), 1] = (
        -1.0 * r * torch.cos(2.0 * pi * torch.arange(0, 5) / 5.0)
    )
    pts[range(6, 11), 2] = -1.0 * h

    # Generate triangles at level 0

    trs = torch.zeros((20, 3), dtype=torch.long)

    trs[0, :] = torch.LongTensor([0, 2, 1])
    trs[1, :] = torch.LongTensor([0, 3, 2])
    trs[2, :] = torch.LongTensor([0, 4, 3])
    trs[3, :] = torch.LongTensor([0, 5, 4])
    trs[4, :] = torch.LongTensor([0, 1, 5])

    trs[5, :] = torch.LongTensor([9, 1, 2])
    trs[6, :] = torch.LongTensor([10, 2, 3])
    trs[7, :] = torch.LongTensor([6, 3, 4])
    trs[8, :] = torch.LongTensor([7, 4, 5])
    trs[9, :] = torch.LongTensor([8, 5, 1])

    trs[10, :] = torch.LongTensor([4, 7, 6])
    trs[11, :] = torch.LongTensor([5, 8, 7])
    trs[12, :] = torch.LongTensor([1, 9, 8])
    trs[13, :] = torch.LongTensor([2, 10, 9])
    trs[14, :] = torch.LongTensor([3, 6, 10])

    trs[15, :] = torch.LongTensor([11, 6, 7])
    trs[16, :] = torch.LongTensor([11, 7, 8])
    trs[17, :] = torch.LongTensor([11, 8, 9])
    trs[18, :] = torch.LongTensor([11, 9, 10])
    trs[19, :] = torch.LongTensor([11, 10, 6])

    # Generate next levels

    for levels_index in range(0, levels_count):

        #      0
        #     / \
        #    A---B
        #   / \ / \
        #  1---C---2

        trs_count = trs.shape[0]
        subtrs_count = trs_count * 4

        subtrs = torch.zeros((subtrs_count, 6), dtype=torch.long)

        subtrs[0 * trs_count + torch.arange(0, trs_count), 0] = trs[:, 0]
        subtrs[0 * trs_count + torch.arange(0, trs_count), 1] = trs[:, 0]
        subtrs[0 * trs_count + torch.arange(0, trs_count), 2] = trs[:, 0]
        subtrs[0 * trs_count + torch.arange(0, trs_count), 3] = trs[:, 1]
        subtrs[0 * trs_count + torch.arange(0, trs_count), 4] = trs[:, 2]
        subtrs[0 * trs_count + torch.arange(0, trs_count), 5] = trs[:, 0]

        subtrs[1 * trs_count + torch.arange(0, trs_count), 0] = trs[:, 0]
        subtrs[1 * trs_count + torch.arange(0, trs_count), 1] = trs[:, 1]
        subtrs[1 * trs_count + torch.arange(0, trs_count), 2] = trs[:, 1]
        subtrs[1 * trs_count + torch.arange(0, trs_count), 3] = trs[:, 1]
        subtrs[1 * trs_count + torch.arange(0, trs_count), 4] = trs[:, 1]
        subtrs[1 * trs_count + torch.arange(0, trs_count), 5] = trs[:, 2]

        subtrs[2 * trs_count + torch.arange(0, trs_count), 0] = trs[:, 2]
        subtrs[2 * trs_count + torch.arange(0, trs_count), 1] = trs[:, 0]
        subtrs[2 * trs_count + torch.arange(0, trs_count), 2] = trs[:, 1]
        subtrs[2 * trs_count + torch.arange(0, trs_count), 3] = trs[:, 2]
        subtrs[2 * trs_count + torch.arange(0, trs_count), 4] = trs[:, 2]
        subtrs[2 * trs_count + torch.arange(0, trs_count), 5] = trs[:, 2]

        subtrs[3 * trs_count + torch.arange(0, trs_count), 0] = trs[:, 0]
        subtrs[3 * trs_count + torch.arange(0, trs_count), 1] = trs[:, 1]
        subtrs[3 * trs_count + torch.arange(0, trs_count), 2] = trs[:, 1]
        subtrs[3 * trs_count + torch.arange(0, trs_count), 3] = trs[:, 2]
        subtrs[3 * trs_count + torch.arange(0, trs_count), 4] = trs[:, 2]
        subtrs[3 * trs_count + torch.arange(0, trs_count), 5] = trs[:, 0]

        subtrs_flatten = torch.cat(
            (subtrs[:, [0, 1]], subtrs[:, [2, 3]], subtrs[:, [4, 5]]), axis=0
        )
        subtrs_sorted, _ = torch.sort(subtrs_flatten, axis=1)

        index_max = torch.max(subtrs_sorted)

        subtrs_scalar = (
            subtrs_sorted[:, 0] * (index_max + 1) + subtrs_sorted[:, 1]
        )

        unique_scalar, unique_indices = torch.unique(
            subtrs_scalar, return_inverse=True
        )

        unique_values = torch.zeros(
            (unique_scalar.shape[0], 2), dtype=unique_scalar.dtype
        )

        unique_values[:, 0] = torch.div(
            unique_scalar, index_max + 1, rounding_mode="floor"
        )
        unique_values[:, 1] = unique_scalar - unique_values[:, 0] * (
            index_max + 1
        )

        trs = torch.transpose(torch.reshape(unique_indices, (3, -1)), 0, 1)

        pts = pts[unique_values[:, 0], :] + pts[unique_values[:, 1], :]
        pts /= torch.repeat_interleave(
            torch.unsqueeze(torch.sum(pts ** 2, axis=1) ** 0.5, 1), 3, 1
        )

    return pts
class Covariance(torch.nn.Module):
    def __init__(self, average=True):
        super().__init__()
        self.average = average
    def forward(self, Xs):
        XXs = Covariance.cov(Xs=Xs, average=self.average)
        return XXs
    def cov(Xs, average=True):
        n_mics = Xs.shape[4]

        # the real and imaginary parts
        Xs_re = Xs[..., 0, :].unsqueeze(4)
        Xs_im = Xs[..., 1, :].unsqueeze(4)

        # covariance
        Rxx_re = torch.matmul(Xs_re, Xs_re.transpose(3, 4)) + torch.matmul(
            Xs_im, Xs_im.transpose(3, 4)
        )

        Rxx_im = torch.matmul(Xs_re, Xs_im.transpose(3, 4)) - torch.matmul(
            Xs_im, Xs_re.transpose(3, 4)
        )

        #  the upper triangular part of the covariance matrices
        idx = torch.triu_indices(n_mics, n_mics)

        XXs_re = Rxx_re[..., idx[0], idx[1]]
        XXs_im = Rxx_im[..., idx[0], idx[1]]

        XXs = torch.stack((XXs_re, XXs_im), 3)

        if average is True:
            n_time_frames = XXs.shape[1]
            XXs = torch.mean(XXs, 1, keepdim=True)
            XXs = XXs.repeat(1, n_time_frames, 1, 1, 1)
        return XXs
class SrpPhat(torch.nn.Module):
    def __init__(
        self,
        mics,
        sample_rate=16000,
        speed_sound=343.0,
        eps=1e-20,
    ):
        super().__init__()
        self.doas = sphere()
        self.taus = doas2taus(self.doas, mics=mics, fs=sample_rate, c=speed_sound)
        self.eps = eps
    def forward(self, XXs):
        """
        (batch, time_steps, 3).
        XXs : tensor
            The covariance matrices of the input signal.
            (batch, time_steps, n_fft/2 + 1, 2, n_mics + n_pairs).
        """
        n_fft = XXs.shape[2]
        # steering vector
        As = steering(self.taus.to(XXs.device), n_fft)
        doas = SrpPhat.srp_phat(XXs=XXs, As=As, doas=self.doas, eps=self.eps)
        return doas
    def srp_phat(XXs, As, doas, eps=1e-20):
        """
        (batch, time_steps, 3)
        XXs : The covariance matrices of the input signal.
            (batch, time_steps, n_fft/2 + 1, 2, n_mics + n_pairs).
        As : steering vector of all the potential directions of arrival.
            (n_doas, n_fft/2 + 1, 2, n_mics).
        doas : All the possible directions
            (n_doas, 3).
        """
        As = As.to(XXs.device)
        doas = doas.to(XXs.device)
        n_mics = As.shape[3]

        # the pairs of microphones
        idx = torch.triu_indices(n_mics, n_mics)

        # the demixing vector from the steering vector
        As_1_re = As[:, :, 0, idx[0, :]]
        As_1_im = As[:, :, 1, idx[0, :]]
        As_2_re = As[:, :, 0, idx[1, :]]
        As_2_im = As[:, :, 1, idx[1, :]]
        Ws_re = As_1_re * As_2_re + As_1_im * As_2_im
        Ws_im = As_1_re * As_2_im - As_1_im * As_2_re
        Ws_re = Ws_re.reshape(Ws_re.shape[0], -1)
        Ws_im = Ws_im.reshape(Ws_im.shape[0], -1)

        # Get unique covariance values to reduce the number of computations
        XXs_val, XXs_idx = torch.unique(XXs, return_inverse=True, dim=1)

        # phase transform
        XXs_re = XXs_val[:, :, :, 0, :]
        XXs_im = XXs_val[:, :, :, 1, :]
        XXs_re = XXs_re.reshape((XXs_re.shape[0], XXs_re.shape[1], -1))
        XXs_im = XXs_im.reshape((XXs_im.shape[0], XXs_im.shape[1], -1))
        XXs_abs = torch.sqrt(XXs_re ** 2 + XXs_im ** 2) + eps
        XXs_re_norm = XXs_re / XXs_abs
        XXs_im_norm = XXs_im / XXs_abs

        # Project on the demixing vectors, and keep only real part
        Ys_A = torch.matmul(XXs_re_norm, Ws_re.transpose(0, 1))
        Ys_B = torch.matmul(XXs_im_norm, Ws_im.transpose(0, 1))
        Ys = Ys_A - Ys_B                  # torch.Size([1, 1, 10242])

        # Get maximum points
        _, doas_idx = torch.max(Ys, dim=2)

        # Repeat for each frame
        doas = (doas[doas_idx, :])[:, XXs_idx, :]
        return doas
class STFT(torch.nn.Module):
    """
    (batch, time, channels).
    """
    def __init__(
        self,
        sample_rate,
        win_length=25,
        hop_length=10,
        n_fft=400,
        window_fn=torch.hamming_window,
        normalized_stft=False,
        center=True,
        pad_mode="constant",
        onesided=True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.normalized_stft = normalized_stft
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided

        # Convert win_length and hop_length from ms to samples
        self.win_length = int(
            round((self.sample_rate / 1000.0) * self.win_length)
        )
        self.hop_length = int(
            round((self.sample_rate / 1000.0) * self.hop_length)
        )
        self.window = window_fn(self.win_length)

    def forward(self, x):
        # multi-channel stft
        or_shape = x.shape
        if len(or_shape) == 3:
            x = x.transpose(1, 2)
            x = x.reshape(or_shape[0] * or_shape[2], or_shape[1])
        stft = torch.stft(
            x,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.window.to(x.device),
            self.center,
            self.pad_mode,
            self.normalized_stft,
            self.onesided,
            return_complex=True,
        )
        stft = torch.view_as_real(stft)
        # Retrieving the original dimensionality (batch,time, channels)
        if len(or_shape) == 3:
            stft = stft.reshape(
                or_shape[0],
                or_shape[2],
                stft.shape[1],
                stft.shape[2],
                stft.shape[3],
            )
            stft = stft.permute(0, 3, 2, 4, 1)
        else:
            # (batch, time, channels)
            stft = stft.transpose(2, 1)
        return stft