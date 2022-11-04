import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

TYPES = ('*.png', '*.jpg', '*.jpeg', '*.bmp')

#torch.set_default_tensor_type(torch.FloatTensor)

class ct_dataset(Dataset):
    def __init__(self, mode, load_mode, input_path, target_path, patch_n=None, patch_size=None, transform=None):
        assert mode in ['train', 'test'], "mode is 'train' or 'test'"
        assert load_mode in [0,1], "load_mode is 0 or 1"

        #input_path = sorted(glob(os.path.join(saved_path, '*_input.npy')))
        #target_path = sorted(glob(os.path.join(saved_path, '*_target.npy')))
        self.input_path = []
        self.target_path = []
        for ext in TYPES:
            self.input_path.extend(glob.glob(os.path.join(input_path, ext)))
            #print(glob.glob(os.path.join(input_path, ext)))
            self.target_path.extend(glob.glob(os.path.join(target_path, ext)))
        self.load_mode = load_mode
        self.patch_n = patch_n
        self.patch_size = patch_size
        self.transform = transform
        #print(input_path)


    def __len__(self):
        return len(self.target_path)

    def __getitem__(self, idx):
        input_img_path, target_img_path = self.input_path[idx], self.target_path[idx]
        input_img = np.array(Image.open(input_img_path))/255
        target_img = np.array(Image.open(target_img_path))/255
       

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        if self.patch_size:
            input_patches, target_patches = get_patch(input_img,
                                                      target_img,
                                                      self.patch_n,
                                                      self.patch_size)
            return (input_patches, target_patches)
        else:
            return (input_img, target_img, input_img_path)


def get_patch(full_input_img, full_target_img, patch_n, patch_size):
    assert full_input_img.shape == full_target_img.shape
    #print(full_target_img.shape)
    patch_input_imgs = []
    patch_target_imgs = []
    h, w = full_input_img.shape
    new_h, new_w = patch_size, patch_size
    for _ in range(patch_n):
        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)
        patch_input_img = full_input_img[top:top+new_h, left:left+new_w]
        patch_target_img = full_target_img[top:top+new_h, left:left+new_w]
        patch_input_imgs.append(patch_input_img)
        patch_target_imgs.append(patch_target_img)
    return np.array(patch_input_imgs), np.array(patch_target_imgs)


def get_loader(mode='train', load_mode=0,
               input_path=None, target_path=None,
               patch_n=None, patch_size=None,
               transform=None, batch_size=32, num_workers=0):
    dataset_ = ct_dataset(mode, load_mode, input_path, target_path, patch_n, patch_size, transform)
    data_loader = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return data_loader
