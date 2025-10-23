image_size = 128
test_image_size = 256
stages = 5
frame_length = 2 ** 10
frame_step = 2 ** 5 - 2
limit = 8674

n_fft = 1024
hop_length = 160

epochs = 2000
trained_epochs = 0
pretrained = False

lambda_guide = 10
lambda_reconstruction = 5
lambda_spectrogram = 1
lambda_dtw = 1e-2
init_scale = 0.01

val_freq = 10
save_freq = 10

clamp = 2.0
channels_audio = 1
channels_img = 1
device_ids = [0, 1]
log10_lr = -5
lr = 10 ** log10_lr
betas = (0.5, 0.999)
weight_step = 1000
weight_decay = 1e-5
gamma = 0.5

MODEL_PATH = "<MODEL_PATH>"
checkpoint_on_error = True

batch_size = 4
batchsize_val = 1
shuffle_val = False

loss_display_cutoff = 2.0
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = False