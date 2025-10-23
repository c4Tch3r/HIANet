import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedSpectrogramLoss(nn.Module):
    def __init__(self, fs=16000, main_n_fft=2048, device=torch.device("cuda")):
        super().__init__()
        self.fs = fs
        self.main_n_fft = main_n_fft
        self.device = device
        
        self.scales = [
            {'n_fft': 2048, 'hop': 512, 'mel_weights': self._gen_mel_weights(2048)},
            {'n_fft': 1024, 'hop': 256, 'mel_weights': self._gen_mel_weights(1024)},
            {'n_fft': 512, 'hop': 128, 'mel_weights': self._gen_mel_weights(512)}
        ]

        for idx, scale in enumerate(self.scales):
            self.register_buffer(f"mel_weights_{idx}", scale['mel_weights'])

    def _gen_mel_weights(self, n_fft):
        freqs = torch.fft.rfftfreq(n_fft, 1/self.fs)
        center_freq = min(4000, self.fs//2 - 500)
        bandwidth = 3000 if n_fft >= 1024 else 1500
        
        weights = torch.exp(-(freqs - center_freq)**2 / (2*(bandwidth/2.3548)**2))
        return weights.view(1, -1, 1)  # 形状 [1, Freq, 1]

    def _log_spectral_loss(self, x, y):
        return F.l1_loss(torch.log(x + 1e-7), torch.log(y + 1e-7))

    def _compute_spectral_loss(self, orig, proc, scale):
        orig_spec = torch.stft(orig, scale['n_fft'], hop_length=scale['hop'],
                             window=torch.hann_window(scale['n_fft']).to(self.device),
                             return_complex=True).abs()
        proc_spec = torch.stft(proc, scale['n_fft'], hop_length=scale['hop'],
                             window=torch.hann_window(scale['n_fft']).to(self.device),
                             return_complex=True).abs()
        
        log_loss = self._log_spectral_loss(orig_spec, proc_spec)
        
        mel_weights = getattr(self, f"mel_weights_{self.scales.index(scale)}")
        weighted_loss = (F.mse_loss(orig_spec, proc_spec, reduction='none') 
                       * mel_weights).mean()
        
        return log_loss + weighted_loss

    def forward(self, original_audio, processed_audio):
        assert original_audio.shape == processed_audio.shape, "输入形状必须相同"
        assert original_audio.dim() in [1,2], "输入应为1D或2D张量"
        
        total_loss = 0.0
        for scale in self.scales:
            if original_audio.size(-1) < scale['n_fft']:
                continue 
            total_loss += self._compute_spectral_loss(original_audio, processed_audio, scale)
        
        main_scale = self.scales[0]
        main_loss = self._compute_spectral_loss(original_audio, processed_audio, main_scale)
        
        return (total_loss + main_loss) / (len(self.scales) + 1)