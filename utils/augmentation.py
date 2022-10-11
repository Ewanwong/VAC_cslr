import cv2
import pdb
import PIL
import copy
import scipy.misc
import torch
import random
import numbers
import numpy as np


class ToTensor(object):
    def __call__(self, video):
        if isinstance(video, list):
            video = np.array(video)
            video = torch.from_numpy(video.transpose((0, 3, 1, 2))).float()
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video.transpose((0, 3, 1, 2)))
            # length, channel, w, h
        return video


class TemporalRescale(object):
    def __init__(self, temp_scaling=0.2):
        self.min_len = 32
        self.max_len = 230
        self.L = 1.0 - temp_scaling
        self.U = 1.0 + temp_scaling

    def __call__(self, clip):
        vid_len = len(clip)
        new_len = int(vid_len * (self.L + (self.U - self.L) * np.random.random()))
        if new_len < self.min_len:
            new_len = self.min_len
        if new_len > self.max_len:
            new_len = self.max_len
        if (new_len - 4) % 4 != 0:
            new_len += 4 - (new_len - 4) % 4
        if new_len <= vid_len:
            index = sorted(random.sample(range(vid_len), new_len))
        else:
            index = sorted(random.choices(range(vid_len), k=new_len))
        return clip[index]


class ImageAugmentation:
    def __init__(self, trans=0.1, color_dev=0.1, distortion=True):

        self.trans = trans
        self.color_dev = color_dev
        self.distortion = distortion

    def __call__(self, video):
        # img width*height*channel
        if isinstance(video, list):
            video_len = len(video)
            W, H, C = video[0].shape
        elif isinstance(video, np.ndarray):
            video_len, W, H, C = video.shape

        if self.distortion:
            ran_noise = np.random.random((4, 2))
            ran_color = np.random.randn(3, )
        else:
            ran_noise = np.ones((4, 2)) * 0.5
            ran_color = np.zeros(3, )

        # perspective translation
        dst = np.float32([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]) * np.float32([W, H])
        noise = self.trans * ran_noise * np.float32([[1., 1.], [-1., 1.], [1., -1.], [-1., -1.]]) * [W, H]
        src = np.float32(dst + noise)

        mat = cv2.getPerspectiveTransform(src, dst)
        for i in range(video_len):
            video[i] = cv2.warpPerspective(video[i], mat, (W, H))

        # TODO: add color deviation
        # deviation = np.dot(self.eigen_vector, (self.color_dev * ran_color * self.eigen_value)) * 255.
        # video += deviation[None, None, None, :]

        return video


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label, num_classes):
        for t in self.transforms:
            image = t(image)

        one_hot_labels = []
        for gloss in label:
            one_hot_label = torch.zeros((num_classes,))
            one_hot_label[gloss] = 1.0
            one_hot_labels.append(one_hot_label)
        return image, torch.stack(one_hot_labels, dim=0)


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):
        try:
            im_h, im_w, im_c = clip[0].shape
        except ValueError:
            print(clip[0].shape)
        new_h, new_w = self.size
        new_h = im_h if new_h >= im_h else new_h
        new_w = im_w if new_w >= im_w else new_w
        top = int(round((im_h - new_h) / 2.))
        left = int(round((im_w - new_w) / 2.))
        return [img[top:top + new_h, left:left + new_w] for img in clip]
