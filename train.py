import os

import torch
import torchvision
import torchaudio
import torch.nn.functional as F
import numpy as np
import math
import viz

from train_loader import loader
from IAINet import *
from AAINet import *
import modules.Unet_common as common

from scipy.signal import stft
from MAEtorch import calculate_heatmap

from Specloss import EnhancedSpectrogramLoss
from pydtw import SoftDTW
from auraloss.time import SNRLoss
from VGGLoss import VGGLoss

import audio_config as ac


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
    
    magnitude_spectrogram = np.stack(spectrograms, axis=0)
    return f, t, magnitude_spectrogram

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

def guide_loss(output, bicubic_image):
    loss_fn = torch.nn.MSELoss(reduce=True, reduction='mean')
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)

def snr_loss(output, bicubic_image):
    loss_fn = SNRLoss()
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)

def perceptual_loss(output, bicubic_image):
    loss_fn = VGGLoss().to(device)
    loss = loss_fn(output, bicubic_image)
    return loss


def reconstruction_loss(rev_input, input):
    loss_fn = torch.nn.MSELoss(reduce=True, reduction='mean')
    loss = loss_fn(rev_input, input)
    return loss.to(device)


def spectrogram_loss(ori_spec, final_spec):
    loss = F.l1_loss(ori_spec, final_spec)
    return loss.to(device)


def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)

def computeSNR(signal, noise):
    loss_fn = SNRLoss().cuda()
    snr = -loss_fn(signal, noise)
    return snr.item()

def audio_to_spectrogram(audio, fs=16000, nperseg=2048, noverlap=1536, nfft=2048):
    f, t, Zxx = stft(audio, fs=fs, window='hann', 
                     nperseg=nperseg, noverlap=noverlap,
                     nfft=nfft, return_onesided=True)
    magnitude_spectrogram = np.abs(Zxx)
    return f, t, magnitude_spectrogram

def apply_log_scaling(spec, amin=1e-10, top_db=80.0):
    spec = np.maximum(spec, amin)
    spec_db = 20 * np.log10(spec)
    spec_db = np.maximum(spec_db, -top_db)
    return spec_db

def calculate_similarity_loss(spec_pred, spec_target):
    spec_pred_db = apply_log_scaling(spec_pred)
    spec_target_db = apply_log_scaling(spec_target)
    loss = np.mean((spec_pred_db - spec_target_db) ** 2)
    return loss

train_loader, val_loader = loader()
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

params_trainable = (list(filter(lambda p: p.requires_grad, netR.parameters()))) + \
                   (list(filter(lambda p: p.requires_grad, netG.parameters()))) + \
                   (list(filter(lambda p: p.requires_grad, netB.parameters()))) + \
                   (list(filter(lambda p: p.requires_grad, netA.parameters())))
optim = torch.optim.Adam(params_trainable, lr=ac.lr, betas=ac.betas, eps=1e-6, weight_decay=ac.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, ac.weight_step, gamma=ac.gamma)

spec_loss = EnhancedSpectrogramLoss().to(device)
if ac.pretrained:
    load("<PRETRAINED_MODEL_PATH>")

dwt = common.DWT()
iwt = common.IWT()

softDTW = SoftDTW(gamma=1.0, normalize=True)
best_losses = float('inf')

try:
    os.makedirs("./audios", exist_ok=True)
    os.makedirs(os.path.join(ac.MODEL_PATH, "checkpoints"), exist_ok=True)
    for epoch in range(ac.epochs):
        epoch = epoch + ac.trained_epochs + 1
        loss_history = []
        #################
        #     train:    #
        #################
        for i, data in enumerate(train_loader):

            secrets, covers, sr = data[0].to(device), data[1].to(device), data[2]
            signal_length = covers.size(-1)

            masked_covers = calculate_heatmap(covers)
            covers_R, covers_G, covers_B = torch.chunk(masked_covers, 3, dim=2)
            secrets_R = secrets[:, 0, :, :].unsqueeze(1)
            secrets_G = secrets[:, 1, :, :].unsqueeze(1)
            secrets_B = secrets[:, 2, :, :].unsqueeze(1)

            processed_covers_R = covers_R.reshape(ac.batch_size, ac.channels_audio, ac.image_size, ac.image_size)
            processed_covers_G = covers_G.reshape(ac.batch_size, ac.channels_audio, ac.image_size, ac.image_size)
            processed_covers_B = covers_B.reshape(ac.batch_size, ac.channels_audio, ac.image_size, ac.image_size)
    
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

            containers_R = iwt(output_steg_1).reshape(ac.batch_size, 1, ac.image_size ** 2 * ac.channels_audio)
            containers_G = iwt(output_steg_2).reshape(ac.batch_size, 1, ac.image_size ** 2 * ac.channels_audio)
            containers_B = iwt(output_steg_3).reshape(ac.batch_size, 1, ac.image_size ** 2 * ac.channels_audio)
            img_containers = torch.cat((containers_R, containers_G, containers_B), dim=2)

            covers_input = dwt(covers.reshape(ac.batch_size, 1, ac.image_size * 3, ac.image_size))
            masked_input = dwt(img_containers.reshape(ac.batch_size, 1, ac.image_size * 3, ac.image_size))
            last_input = torch.cat((covers_input, masked_input), 1)
            last_output = netA(last_input)
            steg_containers = last_output.narrow(1, 0, 4)
            auxiliary_z = last_output.narrow(1, 4, 4)

            containers = iwt(steg_containers).reshape(ac.batch_size, 1, ac.image_size ** 2 * 3)
            restored_covers = containers.squeeze(1).cpu().detach()

            output_z_first = gauss_noise(auxiliary_z.shape)
            output_rev_first = torch.cat((steg_containers, output_z_first), 1)
            output_image_first = netA(output_rev_first, True)
            rev_steg_first = output_image_first.narrow(1, 0, 4)
            rev_secret_first = output_image_first.narrow(1, 4, 4)

            rev_secret_first = iwt(rev_secret_first).reshape(ac.batch_size, 1, ac.image_size ** 2 * 3)
            rev_output_steg_1, rev_output_steg_2, rev_output_steg_3 = torch.chunk(rev_secret_first, 3, dim=2)
            rev_output_steg_1 = dwt(rev_output_steg_1.reshape(ac.batch_size, 1, ac.image_size, ac.image_size))
            rev_output_steg_2 = dwt(rev_output_steg_2.reshape(ac.batch_size, 1, ac.image_size, ac.image_size))
            rev_output_steg_3 = dwt(rev_output_steg_3.reshape(ac.batch_size, 1, ac.image_size, ac.image_size))

            output_z_guass_3 = gauss_noise(output_z_3.shape)
            output_rev_3 = torch.cat((rev_output_steg_3, output_z_guass_3), 1)
            output_image_3 = netB(output_rev_3, True)
            rev_steg_2 = output_image_3.narrow(1, 0, 4)
            rev_secret_B = output_image_3.narrow(1, 4, 4)
            output_B = iwt(rev_secret_B)
 
            output_z_gauss_2 = gauss_noise(output_z_2.shape)
            output_rev_2 = torch.cat((rev_output_steg_2, output_z_gauss_2), 1)
            output_image_2 = netG(output_rev_2, True)
            rev_steg_1 = output_image_2.narrow(1, 0, 4)
            rev_secret_G = output_image_2.narrow(1, 4, 4)
            output_G = iwt(rev_secret_G)

            output_z_gauss_1 = gauss_noise(output_z_1.shape)
            output_rev_1 = torch.cat((rev_output_steg_1, output_z_gauss_1), 1)
            output_image_1 = netR(output_rev_1, True)
            rev_steg = output_image_1.narrow(1, 0, 4)
            rev_secret_R = output_image_1.narrow(1, 4, 4)
            output_R = iwt(rev_secret_R)

            secret_rev = torch.cat((output_R, output_G, output_B), 1)

            g_loss = guide_loss(covers, containers)
            r_loss = reconstruction_loss(secrets, secret_rev)

            _, _, ori_spectrogram = audio_to_spectrogram_batch(covers.squeeze(1).cpu().detach().numpy(), fs=22050)
            _, _, processed_spectrogram = audio_to_spectrogram_batch(restored_covers, fs=22050)

            spec1_log = apply_log_scaling(ori_spectrogram)
            spec2_log = apply_log_scaling(processed_spectrogram)

            s_loss = spec_loss(covers.squeeze(1), restored_covers.cuda())
            dtw_loss = softDTW(covers.squeeze(1).cpu(), restored_covers.cpu()).cuda()

            total_loss = ac.lambda_guide * g_loss + ac.lambda_reconstruction * r_loss + ac.lambda_spectrogram * s_loss + ac.lambda_dtw * dtw_loss
            total_loss.backward()
            optim.step()
            optim.zero_grad()

            loss_history.append([total_loss.item(), 0.])
        
        epoch_losses = np.mean(np.array(loss_history), axis=0)
        epoch_losses[1] = np.log10(optim.param_groups[0]['lr'])

        #################
        #      val:     #
        #################

        if epoch >= 0 and epoch % ac.val_freq == 0:
            snr_C = []
            psnr_S = []
            for i, data in enumerate(val_loader):

                secrets, covers, sr = data[0].to(device), data[1].to(device), data[2]
                signal_length = covers.size(-1)

                masked_covers = calculate_heatmap(covers)
                covers_R, covers_G, covers_B = torch.chunk(masked_covers, 3, dim=2)
                secrets_R = secrets[:, 0, :, :].unsqueeze(1)
                secrets_G = secrets[:, 1, :, :].unsqueeze(1)
                secrets_B = secrets[:, 2, :, :].unsqueeze(1)

                processed_covers_R = covers_R.reshape(ac.batchsize_val, ac.channels_audio, ac.image_size, ac.image_size)
                processed_covers_G = covers_G.reshape(ac.batchsize_val, ac.channels_audio, ac.image_size, ac.image_size)
                processed_covers_B = covers_B.reshape(ac.batchsize_val, ac.channels_audio, ac.image_size, ac.image_size)

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

                containers_R = iwt(output_steg_1).reshape(ac.batchsize_val, 1, ac.image_size ** 2 * ac.channels_audio)
                containers_G = iwt(output_steg_2).reshape(ac.batchsize_val, 1, ac.image_size ** 2 * ac.channels_audio)
                containers_B = iwt(output_steg_3).reshape(ac.batchsize_val, 1, ac.image_size ** 2 * ac.channels_audio)
                img_containers = torch.cat((containers_R, containers_G, containers_B), dim=2)

                covers_input = dwt(covers.reshape(ac.batchsize_val, 1, ac.image_size * 3, ac.image_size))
                masked_input = dwt(img_containers.reshape(ac.batchsize_val, 1, ac.image_size * 3, ac.image_size))
                last_input = torch.cat((covers_input, masked_input), 1)
                last_output = netA(last_input)
                steg_containers = last_output.narrow(1, 0, 4)
                auxiliary_z = last_output.narrow(1, 4, 4)

                containers = iwt(steg_containers).reshape(ac.batchsize_val, 1, ac.image_size ** 2 * 3)
                restored_covers = containers.squeeze(1).cpu().detach()

                snr_c = computeSNR(covers, containers)
                snr_C.append(snr_c)

                torchaudio.save(
                    "./audios/output_{}.wav".format(i), 
                    restored_covers, 
                    sr[0],
                    encoding="PCM_S",
                    bits_per_sample=16
                )

                ######### Reveal ##########

                new_sound, sr = torchaudio.load("./audios/output_{}.wav".format(i))
                new_sound = new_sound.reshape(ac.batchsize_val, 1, ac.image_size * 3, ac.image_size).cuda()

                steg_containers = dwt(new_sound)
                output_z_first = gauss_noise(auxiliary_z.shape)
                output_rev_first = torch.cat((steg_containers, output_z_first), 1)
                output_image_first = netA(output_rev_first, True)
                rev_steg_first = output_image_first.narrow(1, 0, 4)
                rev_secret_first = output_image_first.narrow(1, 4, 4)

                rev_secret_first = iwt(rev_secret_first).reshape(ac.batchsize_val, 1, ac.image_size ** 2 * 3)
                rev_output_steg_1, rev_output_steg_2, rev_output_steg_3 = torch.chunk(rev_secret_first, 3, dim=2)
                rev_output_steg_1 = dwt(rev_output_steg_1.reshape(ac.batchsize_val, 1, ac.image_size, ac.image_size))
                rev_output_steg_2 = dwt(rev_output_steg_2.reshape(ac.batchsize_val, 1, ac.image_size, ac.image_size))
                rev_output_steg_3 = dwt(rev_output_steg_3.reshape(ac.batchsize_val, 1, ac.image_size, ac.image_size))

                output_z_guass_3 = gauss_noise(output_z_3.shape)
                output_rev_3 = torch.cat((rev_output_steg_3, output_z_guass_3), 1)
                output_image_3 = netB(output_rev_3, True)
                rev_steg_2 = output_image_3.narrow(1, 0, 4)
                rev_secret_B = output_image_3.narrow(1, 4, 4)
                output_B = iwt(rev_secret_B)

                output_z_gauss_2 = gauss_noise(output_z_2.shape)
                output_rev_2 = torch.cat((rev_output_steg_2, output_z_gauss_2), 1)
                output_image_2 = netG(output_rev_2, True)
                rev_steg_1 = output_image_2.narrow(1, 0, 4)
                rev_secret_G = output_image_2.narrow(1, 4, 4)
                output_G = iwt(rev_secret_G)

                output_z_gauss_1 = gauss_noise(output_z_1.shape)
                output_rev_1 = torch.cat((rev_output_steg_1, output_z_gauss_1), 1)
                output_image_1 = netR(output_rev_1, True)
                rev_steg = output_image_1.narrow(1, 0, 4)
                rev_secret_R = output_image_1.narrow(1, 4, 4)
                output_R = iwt(rev_secret_R)

                secret_rev = torch.cat((output_R, output_G, output_B), 1)
                torchvision.utils.save_image(secret_rev, "./audios4/secret_{}.png".format(i))

                secrets = secrets.cpu().detach().numpy().squeeze() * 255
                np.clip(secrets, 0, 255)
                secret_rev = secret_rev.cpu().detach().numpy().squeeze() * 255
                np.clip(secret_rev, 0, 255)
                psnr_s = computePSNR(secrets, secret_rev)
                psnr_S.append(psnr_s)
            
            print("SNR C: {}, PSNR S: {}".format(np.mean(snr_C), np.mean(psnr_S)))

        viz.show_loss(epoch_losses)
        if epoch > 0 and (epoch % ac.save_freq) == 0:
            torch.save({'opt': optim.state_dict(),
                        'netR': netR.state_dict(),
                        'netG': netG.state_dict(),
                        'netB': netB.state_dict(),
                        'netA': netA.state_dict(),
                        }, ac.MODEL_PATH + 'checkpoints/model_checkpoint_%.5i' % epoch + '.pt')
        weight_scheduler.step()

        if epoch_losses[0] < best_losses:
            torch.save({'opt': optim.state_dict(),
                        'netR': netR.state_dict(),
                        'netG': netG.state_dict(),
                        'netB': netB.state_dict(),
                        'netA': netA.state_dict(),
                        }, ac.MODEL_PATH + 'checkpoints/model_best.pt')
            best_losses = epoch_losses[0]
except:
    if ac.checkpoint_on_error:
        torch.save({'opt': optim.state_dict(),
                    'netR': netR.state_dict(),
                    'netG': netG.state_dict(),
                    'netB': netB.state_dict(),
                    'netA': netA.state_dict(),
                    }, ac.MODEL_PATH + 'checkpoints/model_ABORT' + '.pt')
    raise
finally:
    viz.signal_stop()