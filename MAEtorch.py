import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from torchaudio.functional import griffinlim
import librosa

def calculate_heatmap(x, sr=22050, device='cuda'):
    B, C, seq_len = x.size()

    n_fft = 1024
    hop_length = 160
    win_length = 1024
    n_mels = 128
    window_size = 12

    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        f_min=20,
        f_max=11025,
        n_mels=n_mels,
        power=2.0,
        pad_mode="constant",
        norm='slaney',
        mel_scale="slaney",
    ).to(device)

    mel_freqs = torch.tensor(librosa.mel_frequencies(n_mels=n_mels, fmin=20, fmax=11025), device=device)
    bark_bands = 13 * torch.atan(0.76 * mel_freqs / 1000) + 3.5 * torch.atan((mel_freqs / 7500) ** 2)

    kernel_2d = torch.ones((2 * window_size + 1, 1), device=device)
    kernel = torch.flip(kernel_2d, dims=[0, 1]).unsqueeze(0).unsqueeze(0)
    pad_h = kernel_2d.size(0) // 2
    pad_w = kernel_2d.size(1) // 2

    freq_diff = bark_bands[:, None] - bark_bands[None, :]
    freq_diff = freq_diff.to(dtype=torch.float32)

    high_mask = torch.zeros_like(freq_diff, device=device)
    low_mask = torch.zeros_like(freq_diff, device=device)
    high_mask[freq_diff < 0] = torch.pow(10, -27 * freq_diff[freq_diff < 0].abs() / 20.0)
    low_mask[freq_diff > 0] = torch.pow(10, -5 * freq_diff[freq_diff > 0].abs() / 20.0)
    low_mask[freq_diff == 0] = 0.0

    decay_1d = torch.tensor([10 ** (-0.02 * dt) for dt in range(1, 6)], device=device, dtype=torch.float32)

    inverse_mel_scale = torchaudio.transforms.InverseMelScale(
        n_mels=n_mels,
        sample_rate=sr,
        n_stft=n_fft // 2 + 1,
        f_min=20,
        f_max=11025,
        norm='slaney',
        mel_scale="slaney"
    ).to(device)

    hann_window = torch.hann_window(win_length, device=device)

    batch_outputs = []
    for batch_idx in range(B):
        x_batch = x[batch_idx]

        mel_spec = mel_transform(x_batch).squeeze(0)

        mel_spec_4d = mel_spec.unsqueeze(0).unsqueeze(0)
        mel_spec_padded = F.pad(mel_spec_4d, pad=(pad_w, pad_w, pad_h, pad_h), mode='reflect')
        critical_band_energy = F.conv2d(mel_spec_padded, kernel).squeeze(0).squeeze(0)

        M_f_high = torch.matmul(high_mask, critical_band_energy)
        M_f_low = torch.matmul(low_mask, critical_band_energy)
        M_f = M_f_high + M_f_low

        channels = M_f.shape[0]
        decay_kernel = decay_1d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1)
        M_f_batch = M_f.unsqueeze(0)
        M_t = F.conv1d(M_f_batch, decay_kernel, padding=2, groups=channels).squeeze(0)

        M = M_f + M_t
        masking_heatmap = M / (M.max() + 1e-8) if M.max() > 0 else torch.zeros_like(M)
        masking_threshold = 10 ** (-3 * masking_heatmap)
        target_mel = mel_spec * masking_threshold

        target_stft_mag = inverse_mel_scale(target_mel)

        y = griffinlim(
            specgram=target_stft_mag,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=hann_window,
            power=2.0,
            n_iter=16,
            momentum=0.99,
            length=x_batch.shape[-1],
            rand_init=None
        )

        if y.shape[-1] != seq_len:
            if y.shape[-1] > seq_len:
                y = y[..., :seq_len]
            else:
                pad_len = seq_len - y.shape[-1]
                y = F.pad(y, (0, pad_len), mode='constant', value=0)

        y = y.unsqueeze(0)

        batch_outputs.append(y)

    return torch.stack(batch_outputs, dim=0)