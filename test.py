import os

import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
import math
import csv

from test_loader import loader
from IAINet import *
from AAINet import *
import modules.Unet_common as common
from scipy.signal import stft
from MAEtorch import calculate_heatmap

from pydtw import SoftDTW
from auraloss.time import SNRLoss
from pytorch_msssim import ssim
import PerceptualSimilarity.models

import audio_config as ac

os.makedirs("./ours_test", exist_ok=True)
os.makedirs(os.path.join("./ours_test", "cover"), exist_ok=True)
os.makedirs(os.path.join("./ours_test", "steg"), exist_ok=True)
CSV_SAVE_PATH = "./ours_test/test_metrics.csv"


def load(name):
    state_dicts = torch.load(name)
    network_R_state_dict = {k: v for k, v in state_dicts['netR'].items() if 'tmp_var' not in k}
    network_G_state_dict = {k: v for k, v in state_dicts['netG'].items() if 'tmp_var' not in k}
    network_B_state_dict = {k: v for k, v in state_dicts['netB'].items() if 'tmp_var' not in k}
    network_A_state_dict = {k: v for k, v in state_dicts['netA'].items() if 'tmp_var' not in k}
    netR.load_state_dict(network_R_state_dict)
    netG.load_state_dict(network_G_state_dict)
    netB.load_state_dict(network_B_state_dict)
    netA.load_state_dict(network_A_state_dict)


def audio_to_spectrogram_batch(audio, fs=16000, nperseg=2048, noverlap=1536, nfft=2048):
    batches, _ = audio.shape
    nperseg = nperseg if nperseg is not None else 256
    noverlap = noverlap if noverlap is not None else nperseg // 8
    
    spectrograms = []
    for i in range(batches):
        f, t, Zxx = stft(audio[i], fs=fs, window='hann',
                       nperseg=nperseg, noverlap=noverlap,
                       nfft=nfft, return_onesided=True)
        spectrograms.append(np.abs(Zxx))
    
    return f, t, np.stack(spectrograms, axis=0)


def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = ac.init_scale * torch.randn(param.data.shape).cuda()
            if split[-2] == 'conv5':
                param.data.fill_(0.)


def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()
    return noise


def computePSNR(origin, pred):
    origin = np.array(origin).astype(np.float32)
    pred = np.array(pred).astype(np.float32)
    mse = np.mean((origin - pred) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0**2 / mse)


def computeSNR(signal, noise):
    loss_fn = SNRLoss().cuda()
    snr = -loss_fn(signal, noise)
    return snr.item()


test_loader = loader()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lpips_model = PerceptualSimilarity.models.PerceptualLoss(
    model='net-lin', net='alex', use_gpu=True, gpu_ids=ac.device_ids
)

netR = IAINet().cuda()
netG = IAINet().cuda()
netB = IAINet().cuda()
netA = AAINet().cuda()

init_model(netR)
init_model(netG)
init_model(netB)
init_model(netA)

netR = torch.nn.DataParallel(netR, device_ids=ac.device_ids)
netG = torch.nn.DataParallel(netG, device_ids=ac.device_ids)
netB = torch.nn.DataParallel(netB, device_ids=ac.device_ids)
netA = torch.nn.DataParallel(netA, device_ids=ac.device_ids)

load("<BEST_MODEL_PATH>")

dwt = common.DWT()
iwt = common.IWT()
softDTW = SoftDTW(gamma=1.0, normalize=True)
best_losses = float('inf')

netR.eval()
netG.eval()
netB.eval()
netA.eval()

snr_C = []
mse_C = []
psnr_S = []
ssim_S = []
lpips_S = []

snr_M = []
mse_M = []

with torch.no_grad():
    with open(CSV_SAVE_PATH, 'w', newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            'Index',
            'Audio MSE',
            'Audio SNR',
            'Image PSNR',
            'Image SSIM',
            'Image LPIPS'
        ])
    

    for batch_idx, data in enumerate(test_loader):
        secrets, covers, sr = data[0].to(device), data[1].to(device), data[2]
        signal_length = covers.size(-1)

        torchaudio.save(
            f"./ours_test/cover/cover_{batch_idx:05d}.wav", 
            covers.squeeze(1).cpu().detach(), 
            sr[0],
            encoding="PCM_S",
            bits_per_sample=16
        )

        masked_covers = calculate_heatmap(covers)
        covers_R, covers_G, covers_B = torch.chunk(masked_covers, 3, dim=2)
        
        secrets_R = secrets[:, 0, :, :].unsqueeze(1)
        secrets_G = secrets[:, 1, :, :].unsqueeze(1)
        secrets_B = secrets[:, 2, :, :].unsqueeze(1)

        processed_covers_R = covers_R.reshape(ac.batchsize_val, ac.channels_audio, ac.test_image_size, ac.test_image_size)
        processed_covers_G = covers_G.reshape(ac.batchsize_val, ac.channels_audio, ac.test_image_size, ac.test_image_size)
        processed_covers_B = covers_B.reshape(ac.batchsize_val, ac.channels_audio, ac.test_image_size, ac.test_image_size)

        cover_input_R = dwt(processed_covers_R)
        secret_R_input = dwt(secrets_R)
        input_img_R = torch.cat((cover_input_R, secret_R_input), 1)
        output1 = netR(input_img_R)
        output_steg_1 = output1.narrow(1, 0, 4)
        output_z_1 = output1.narrow(1, 4, 4)

        cover_input_G = dwt(processed_covers_G)
        secret_G_input = dwt(secrets_G)
        input_img_G = torch.cat((cover_input_G, secret_G_input), 1)
        output2 = netG(input_img_G)
        output_steg_2 = output2.narrow(1, 0, 4)
        output_z_2 = output2.narrow(1, 4, 4)

        cover_input_B = dwt(processed_covers_B)
        secrets_B_input = dwt(secrets_B)
        input_img_B = torch.cat((cover_input_B, secrets_B_input), 1)
        output3 = netB(input_img_B)
        output_steg_3 = output3.narrow(1, 0, 4)
        output_z_3 = output3.narrow(1, 4, 4)

        containers_R = iwt(output_steg_1).reshape(ac.batchsize_val, 1, ac.test_image_size ** 2 * ac.channels_audio)
        containers_G = iwt(output_steg_2).reshape(ac.batchsize_val, 1, ac.test_image_size ** 2 * ac.channels_audio)
        containers_B = iwt(output_steg_3).reshape(ac.batchsize_val, 1, ac.test_image_size ** 2 * ac.channels_audio)
        img_containers = torch.cat((containers_R, containers_G, containers_B), dim=2)

        snr_m = computeSNR(masked_covers, img_containers)
        mse_m = F.mse_loss(masked_covers, img_containers).item()
        snr_M.append(snr_m)
        mse_M.append(mse_m)

        covers_input = dwt(covers.reshape(ac.batchsize_val, 1, ac.test_image_size * 3, ac.test_image_size))
        masked_input = dwt(img_containers.reshape(ac.batchsize_val, 1, ac.test_image_size * 3, ac.test_image_size))
        last_input = torch.cat((covers_input, masked_input), 1)
        last_output = netA(last_input)
        steg_containers = last_output.narrow(1, 0, 4)
        auxiliary_z = last_output.narrow(1, 4, 4)

        containers = iwt(steg_containers).reshape(ac.batchsize_val, 1, ac.test_image_size ** 2 * 3)
        restored_covers = containers.squeeze(1).cpu().detach()

        snr_c = computeSNR(covers, containers)
        mse_c = F.mse_loss(covers, containers).item()
        snr_C.append(snr_c)
        mse_C.append(mse_c)

        torchaudio.save(
            f"./ours_test/steg/output_{batch_idx:05d}.wav", 
            restored_covers, 
            sr[0],
            encoding="PCM_S",
            bits_per_sample=16
        )

        new_sound, _ = torchaudio.load(f"./ours_test/steg/output_{batch_idx:05d}.wav")
        new_sound = new_sound.reshape(ac.batchsize_val, 1, ac.test_image_size * 3, ac.test_image_size).cuda()

        steg_containers = dwt(new_sound)
        output_z_first = gauss_noise(auxiliary_z.shape)
        output_rev_first = torch.cat((steg_containers, output_z_first), 1)
        output_image_first = netA(output_rev_first, True)
        rev_steg_first = output_image_first.narrow(1, 0, 4)
        rev_secret_first = output_image_first.narrow(1, 4, 4)

        rev_secret_first = iwt(rev_secret_first).reshape(ac.batchsize_val, 1, ac.test_image_size ** 2 * 3)
        rev_output_steg_1, rev_output_steg_2, rev_output_steg_3 = torch.chunk(rev_secret_first, 3, dim=2)
        
        rev_output_steg_1 = dwt(rev_output_steg_1.reshape(ac.batchsize_val, 1, ac.test_image_size, ac.test_image_size))
        rev_output_steg_2 = dwt(rev_output_steg_2.reshape(ac.batchsize_val, 1, ac.test_image_size, ac.test_image_size))
        rev_output_steg_3 = dwt(rev_output_steg_3.reshape(ac.batchsize_val, 1, ac.test_image_size, ac.test_image_size))

        output_z_guass_3 = gauss_noise(output_z_3.shape)
        output_rev_3 = torch.cat((rev_output_steg_3, output_z_guass_3), 1)
        output_image_3 = netB(output_rev_3, True)
        rev_secret_B = output_image_3.narrow(1, 4, 4)
        output_B = iwt(rev_secret_B)

        output_z_gauss_2 = gauss_noise(output_z_2.shape)
        output_rev_2 = torch.cat((rev_output_steg_2, output_z_gauss_2), 1)
        output_image_2 = netG(output_rev_2, True)
        rev_secret_G = output_image_2.narrow(1, 4, 4)
        output_G = iwt(rev_secret_G)

        output_z_gauss_1 = gauss_noise(output_z_1.shape)
        output_rev_1 = torch.cat((rev_output_steg_1, output_z_gauss_1), 1)
        output_image_1 = netR(output_rev_1, True)
        rev_secret_R = output_image_1.narrow(1, 4, 4)
        output_R = iwt(rev_secret_R)

        secret_rev = torch.cat((output_R, output_G, output_B), 1)

        ssim_temp_s = ssim(secrets, secret_rev, data_range=1.0)
        ssim_S.append(ssim_temp_s.item())
        
        lpips_temp_s = lpips_model.forward(secrets, secret_rev)
        lpips_S.append(lpips_temp_s.item())
        
        secrets_np = secrets.cpu().detach().numpy().squeeze() * 255
        secrets_np = np.clip(secrets_np, 0, 255)
        secret_rev_np = secret_rev.cpu().detach().numpy().squeeze() * 255
        secret_rev_np = np.clip(secret_rev_np, 0, 255)
        psnr_temp_s = computePSNR(secrets_np, secret_rev_np)
        psnr_S.append(psnr_temp_s)

        with open(CSV_SAVE_PATH, 'a', newline='', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([
                batch_idx, 
                f"{mse_c:.6e}", 
                f"{snr_c:.4f}",
                f"{psnr_temp_s:.4f}",
                f"{ssim_temp_s.item():.4f}",
                f"{lpips_temp_s.item():.6f}"
            ])
    
    print(np.mean(snr_M), np.mean(mse_M))

    avg_mse_C = np.mean(mse_C) if mse_C else 0.0
    avg_snr_C = np.mean(snr_C) if snr_C else 0.0
    avg_psnr_S = np.mean(psnr_S) if psnr_S else 0.0
    avg_ssim_S = np.mean(ssim_S) if ssim_S else 0.0
    avg_lpips_S = np.mean(lpips_S) if lpips_S else 0.0

    with open(CSV_SAVE_PATH, 'a', newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['', '', '', '', '', ''])
        csv_writer.writerow([
            'Average values',
            f"{avg_mse_C:.6e}",
            f"{avg_snr_C:.4f}",
            f"{avg_psnr_S:.4f}",
            f"{avg_ssim_S:.4f}",
            f"{avg_lpips_S:.6f}"
        ])

    print(f"MSE: {avg_mse_C:.6e}")
    print(f"SNR: {avg_snr_C:.4f} dB")
    print(f"PSNR: {avg_psnr_S:.4f} dB")
    print(f"SSIM: {avg_ssim_S:.4f}")
    print(f"LPIPS: {avg_lpips_S:.6f}")

    print(f"CSV SAVE PATH: {os.path.abspath(CSV_SAVE_PATH)}")