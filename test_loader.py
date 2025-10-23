import torch
import pathlib
import torchaudio
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch_stft import STFT
import audio_config as ac

TEST_AUDIO_FOLDER = "<TEST_AUDIO_FOLDER>"
TEST_DATA_FOLDER = "<TEST_DATA_FOLDER>"

class AudioProcessor():
    def __init__(self, transform, target_length=None):
        self.target_length = target_length
        self._limit = ac.limit
        self._frame_length = ac.frame_length if transform == 'cosine' else 2 ** 11 - 1
        self._frame_step = ac.frame_step if transform == 'cosine' else 132
        self._transform = transform
        
        if self._transform == 'fourier':
            self.stft = STFT(
                filter_length=self._frame_length, 
                hop_length=self._frame_step, 
                win_length=self._frame_length,
                window='hann'
            )   

    def forward(self, audio_path):
        sound, sr = torchaudio.load(audio_path)
        
        if self.target_length is not None and sound.size(1) > self.target_length:
            sound = sound[:, :self.target_length]
                
        return sound, sr
    
    def _process_fourier(self, sound):
        magnitude, phase = self.stft.transform(sound)
        return magnitude, phase

class StegoDataset(Dataset):
    def __init__(
        self,
        image_root: str,
        audio_root: str,
        folder: str,
        rgb: bool = True,
        transform: str = 'cosine',
        image_extension: str = "png",
        audio_extension: str = "wav",
        audio_target_length: int = None,
        image_size: int = ac.test_image_size
    ):
        self._image_data_path = pathlib.Path(image_root)
        self._audio_data_path = pathlib.Path(audio_root)
        self._MAX_LIMIT = 10000 if folder == 'train' else 900
        self._MAX_AUDIO_LIMIT = 17584 if folder == 'train' else 946
        self._colorspace = 'RGB' if rgb else 'L'
        self._transform = transform
        self.audio_target_length = audio_target_length
        self.image_size = image_size

        print(f'IMAGE DATA LOCATED AT: {self._image_data_path}')
        print(f'AUDIO DATA LOCATED AT: {self._audio_data_path}')

        self.image_files = glob.glob(f'{self._image_data_path}/*.{image_extension}')
        if len(self.image_files) > self._MAX_LIMIT:
            self.image_files = self.image_files[:self._MAX_LIMIT]
            
        all_audio_files = glob.glob(f'{self._audio_data_path}/*.{audio_extension}')
        self.audio_files = self._filter_audio_by_length(all_audio_files, audio_target_length)
        
        if len(self.audio_files) > self._MAX_AUDIO_LIMIT:
            self.audio_files = self.audio_files[:self._MAX_AUDIO_LIMIT]
            
        self.length = len(self.audio_files)
        self.num_images = len(self.image_files)
        
        self._AUDIO_PROCESSOR = AudioProcessor(
            transform=self._transform, 
            target_length=audio_target_length
        )
        
        self.image_transforms = transforms.Compose([
            transforms.CenterCrop((512, 512)),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        print(f'Loaded {self.num_images} images and {self.length} audio files')
        print(f'Dataset size: {self.length} samples')

    def _filter_audio_by_length(self, audio_files, min_length):
        if min_length is None:
            return audio_files
            
        filtered_files = []
        discarded_count = 0
        
        print(f"开始过滤音频文件，最小长度要求: {min_length}")
        
        for audio_path in audio_files:
            try:
                info = torchaudio.info(audio_path)
                audio_length = info.num_frames
                
                if audio_length >= min_length:
                    filtered_files.append(audio_path)
                else:
                    discarded_count += 1
                    
            except Exception as e:
                print(f"读取音频文件 {audio_path} 时出错: {e}")
                discarded_count += 1
                
        print(f"音频过滤完成: 保留 {len(filtered_files)} 个文件, 舍弃 {discarded_count} 个文件")
        return filtered_files

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_idx = index % self.num_images
        audio_idx = index
        
        img_path = self.image_files[img_idx]
        audio_path = self.audio_files[audio_idx]

        img = Image.open(img_path).convert(self._colorspace)
        img = self.image_transforms(img)
        
        sound, sr = self._AUDIO_PROCESSOR.forward(audio_path)
        
        if self._transform == 'cosine':
            return (img, sound, sr)
        elif self._transform == 'fourier':
            magnitude, phase = self._AUDIO_PROCESSOR._process_fourier(sound)
            return (img, magnitude, phase, sr)
        else: 
            raise Exception(f'Transform not implemented: {self._transform}')

def collate_fn(batch):
    elements = list(zip(*batch))
    
    if isinstance(elements[0][0], torch.Tensor):
        images = torch.stack(elements[0])
    else:
        images = torch.tensor(elements[0])
    
    if isinstance(elements[1][0], torch.Tensor):
        audios = torch.stack(elements[1])
    else:
        audios = torch.tensor(elements[1])
    
    sample_rates = elements[2] if len(elements) > 2 else None
    
    if len(elements) == 3:
        return (images, audios, sample_rates)
    else:
        magnitudes = torch.stack(elements[2])
        phases = torch.stack(elements[3])
        return (images, magnitudes, phases, sample_rates)

def loader(rgb=True, transform='cosine'):
    audio_target_length = ac.test_image_size ** 2 * ac.channels_audio * 3
    image_size = ac.test_image_size
    

    test_dataset = StegoDataset(
        image_root=TEST_DATA_FOLDER,
        audio_root=TEST_AUDIO_FOLDER,
        folder='val',
        rgb=rgb,
        transform=transform,
        audio_target_length=audio_target_length,
        image_size=image_size
    )


    test_dataloader = DataLoader(
        test_dataset,
        batch_size=ac.batchsize_val,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )

    print('Data loaded ++')
    return test_dataloader