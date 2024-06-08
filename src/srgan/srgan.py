import torch
import requests
import os
import numpy as np

from torch import nn
from typing import Optional, Union

from . import model
from .utils import load_pretrained_state_dict
from .imgproc import tensor_to_image, image_to_tensor


def download_from_url(url: str, path: str) -> None:
    response = requests.get(url)
    with open(path, 'wb') as f:
        f.write(response.content)


class SRGAN:
    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # download model weights if not already downloaded from a link to ./pretrained
        self._model_url = "https://huggingface.co/goodfellowliu/SRGAN-PyTorch/resolve/main/SRGAN_x4-SRGAN_ImageNet.pth.tar"
        self._model_weights_path = "./pretrained/SRGAN_x4-SRGAN_ImageNet.pth.tar"

        if not os.path.exists(self._model_url):
            os.makedirs("./pretrained", exist_ok=True)
            download_from_url(self._model_url, self._model_weights_path)
        
        self.model_arch_name = 'srresnet_x4'
        self.model = None
        self._build_model()

    def _build_model(self) -> None:
        # Initialize the super-resolution model
        sr_model = model.__dict__[self.model_arch_name]()

        # Load model weights
        sr_model = load_pretrained_state_dict(sr_model, False, self._model_weights_path)
    
        # Start the verification mode of the model.
        sr_model.eval()
        sr_model.half()

        sr_model = sr_model.to(self.device)
        self.model = sr_model

    def __call__(self, img: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        # img needs to be in the range [0, 1]
        if isinstance(img, np.ndarray):
            img = image_to_tensor(img, False, True).to(self.device).unsqueeze_(0)
        elif isinstance(img, torch.Tensor):
            img = img.half().to(self.device)
        else:
            raise ValueError("img should be a numpy array or a torch tensor")
        
        with torch.no_grad():
            sr_img = self.model(img)
        return sr_img
