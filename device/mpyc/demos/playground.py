import os
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np

transform = transforms.Compose([transforms.Resize((100,100)), transforms.ToTensor()])
image = Image.open(os.path.join('data_faces', '1.pgm'))
image = image.convert("L")
image = transform(image)
image = image.to(torch.device("cpu"))
image = image[None, ...]
image = np.array(image)[0][0]
print(image)
