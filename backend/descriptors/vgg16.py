"""VGG16 deep features (convolutional activations of a network pretrained on ImageNet).

VGG16 (Simonyan & Zisserman, 2014) stacks 13 3x3 convolution layers in 5 blocks,
each block ending with a 2x2 max-pooling. We keep only the convolutional part
(`features`, up to and including block 5's pooling; the fully connected
classifier is dropped) with the torchvision ImageNet weights (IMAGENET1K_V1).

Pipeline (identical offline and online):
  1. BGR -> RGB, resize the whole image to 224 x 224 (no crop, so no content is lost),
  2. scale to [0, 1] and standardize with the ImageNet mean/std,
  3. forward through the conv blocks -> 512 feature maps of 7 x 7,
  4. global average pooling over the 7 x 7 positions -> 512 values,
  5. L2 normalization (unit length), compared with the L2 distance.
On unit vectors, L2 distance is a monotone function of cosine similarity
(||a - b||^2 = 2 - 2 cos(a, b)), so this ranks by cosine similarity.
Unlike the hand-crafted descriptors, these activations encode semantic content
(objects, scene types) learned from ImageNet labels.
"""
import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img_bgr, input_size=224):
    """BGR uint8 image -> normalized CHW float32 array of shape (3, size, size)."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    interp = cv2.INTER_AREA if min(h, w) > input_size else cv2.INTER_LINEAR
    rgb = cv2.resize(rgb, (input_size, input_size), interpolation=interp)
    x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


class VGG16Extractor:
    def __init__(self, input_size=224, threads=None):
        import torch  # imported lazily: only needed when VGG16 is used
        import torchvision

        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.input_size = input_size
        model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        self.features = model.features.eval()

    def extract_preprocessed(self, batch):
        """batch: (n, 3, size, size) float32 -> (n, 512) float32, L2-normalized."""
        with self.torch.inference_mode():
            maps = self.features(self.torch.from_numpy(batch))  # (n, 512, 7, 7)
            vec = maps.mean(dim=(2, 3))  # global average pooling
            vec = self.torch.nn.functional.normalize(vec, p=2, dim=1)
        return vec.numpy().astype(np.float32)

    def extract(self, images_bgr):
        batch = np.stack([preprocess(img, self.input_size) for img in images_bgr])
        return self.extract_preprocessed(batch)
