import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import zipfile
import numpy as np
from PIL import Image
from io import BytesIO

class ZipReader(object):
    def __init__(self):
        super(ZipReader, self).__init__()
        self.id_context = dict()

    def read(self, zip_file, file_name):
        key_name = zip_file
        if key_name in self.id_context:
            with self.id_context[key_name].open(file_name) as f:
                tmp = f.read()
            return tmp
        else:
            file_handle = zipfile.ZipFile(zip_file, 'r')
            self.id_context[key_name] = file_handle
            return self.id_context[key_name].read(file_name)


class Kinetics(Dataset):

    def __init__(self, folder, video_list, audio_zip_name):

        self.folder = folder
        self.audio_zip_name = audio_zip_name
        self.data = []

        video_map = {}
        f = open(video_list)
        for line in f:
            tmp = line.strip().split('/')
            cat = tmp[0]
            name = tmp[1]
            jpgname = tmp[2]
            prefix = cat + '/' + name
            if not prefix in video_map:
                video_map[prefix] = []
            video_map[prefix].append(line.strip())

        for key in video_map:
            if len(video_map[key]) >= 32:
                video_map[key] = sorted(video_map[key])
                self.data.append({"filename": video_map[key], 'width': 240, 'height': 240, 'audioname': key + '.npy'})

        self.zipreader = ZipReader()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        data = self.data[idx]
        filename = data['filename']
        audioname = data['audioname']

        start = 0
        frame_indices = np.arange(32) +start
        video_zip_name = self.folder + '/' + filename[0].split('/')[0] + '.zip'
        video_image = [Image.open(BytesIO(self.zipreader.read(video_zip_name, filename[i]))) for i in frame_indices]
        w, h = video_image[0].size
        if w < h:
            neww = 112
            newh = int(112. / w * h)
            ow = 0
            oh = int((newh - neww) / 2)
        else:
            newh = 112
            neww = int(112. / h * w)
            oh = 0
            ow = int((neww - newh) / 2)
        video_image = [img.resize((neww, newh)) for img in video_image]
        video_image = [img.crop((ow, oh, ow+112, oh+112)) for img in video_image]

        video_image = [np.asarray(img) for img in video_image]
        video_image = [img / 255. - np.array([0.485, 0.456, 0.406]) / np.array([0.229, 0.224, 0.225]) for img in video_image]
        video_image = [img.transpose((2, 0, 1)) for img in video_image]
        video_image_torch = [torch.from_numpy(item).to(dtype=torch.float32) for item in video_image]

        audio_clip = np.load(BytesIO(self.zipreader.read(self.audio_zip_name, 'kinetics_audio/' + audioname)))
        audio_tot = audio_clip.shape[0]
        audio_start = int(start * 1.0 / len(filename) * audio_tot)
        audio_end = audio_start + int(32. / len(filename) * audio_tot) 
        audio_clip = audio_clip[audio_start:audio_end]
        
        clip = torch.stack(video_image_torch, 0).permute(1, 0, 2, 3)
        audio_clip = torch.from_numpy(audio_clip)
        audio_clip = audio_clip.unsqueeze(0).unsqueeze(0)
        audio_clip = F.interpolate(audio_clip, (96, 80))
        audio_clip = audio_clip.view(96, 80)

        return clip, audio_clip
