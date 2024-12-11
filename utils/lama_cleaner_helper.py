import io
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List
from urllib.parse import urlparse
import cv2
from PIL import Image, ImageOps, PngImagePlugin
import numpy as np
import torch
import logging
from torch.hub import download_url_to_file, get_dir
from utils.device_utils import get_device

logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
   handlers=[
       logging.FileHandler('lama_cleaner.log'),
       logging.StreamHandler()
   ]
)
logger = logging.getLogger(__name__)

class ModelError(Exception):
   """모델 관련 에러"""
   pass

class ValidationError(Exception):
   """검증 관련 에러"""
   pass

@dataclass 
class ModelConfig:
   """모델 설정"""
   url: str
   md5: Optional[str] = None
   device: str = get_device()
   model_dir: Optional[Path] = None

   def __post_init__(self):
       if self.model_dir is None:
           hub_dir = Path(get_dir())
           self.model_dir = hub_dir / "checkpoints"
           self.model_dir.mkdir(parents=True, exist_ok=True)

def compute_md5(file_path: Union[str, Path]) -> str:
   path = Path(file_path)
   if not path.exists():
       raise FileNotFoundError(f"File not found: {path}")
       
   md5 = hashlib.md5()
   try:
       with path.open("rb") as f:
           for chunk in iter(lambda: f.read(128 * md5.block_size), b""):
               md5.update(chunk)
       return md5.hexdigest()
   except Exception as e:
       logger.error(f"MD5 계산 실패: {str(e)}")
       raise

def get_cache_path_by_url(url: str, config: ModelConfig) -> Path:
   parts = urlparse(url)
   return config.model_dir / Path(parts.path).name

def validate_model(file_path: Path, expected_md5: Optional[str]) -> None:
   if not expected_md5:
       return
   actual_md5 = compute_md5(file_path)
   if actual_md5 != expected_md5:
       try:
           file_path.unlink(missing_ok=True)
           raise ValidationError(
               f"Model validation failed\n"
               f"Expected MD5: {expected_md5}\n"
               f"Actual MD5: {actual_md5}"
           )
       except Exception as e:
           logger.error(f"모델 삭제 실패: {str(e)}")
           raise

def download_model(url: str, config: ModelConfig) -> Path:
   cache_path = get_cache_path_by_url(url, config)
   
   if not cache_path.exists():
       logger.info(f"Downloading model from {url} to {cache_path}")
       try:
           download_url_to_file(url, str(cache_path), None, progress=True)
       except Exception as e:
           logger.error(f"모델 다운로드 실패: {str(e)}")
           raise ModelError(f"Failed to download model: {str(e)}")
       
   try:
       validate_model(cache_path, config.md5)
       logger.info(f"Model validated successfully: {cache_path}")
       return cache_path
   except ValidationError as e:
       raise ModelError(str(e)) from e

def load_jit_model(url_or_path: Union[str, Path], 
                   device: str = get_device(),
                   model_md5: Optional[str] = None
                   ) -> torch.jit.ScriptModule:
   try:
       config = ModelConfig(url=str(url_or_path), md5=model_md5, device=device)

       if Path(url_or_path).exists():
           model_path = Path(url_or_path)
       else:
           model_path = download_model(url_or_path, config)
           
       logger.info(f"Loading model from: {model_path}")
       model = torch.jit.load(str(model_path), map_location="cpu").to(device)
       model.eval()
       return model
       
   except Exception as e:
       logger.error(f"Model loading failed: {str(e)}")
       raise ModelError(f"Failed to load model: {str(e)}")

def norm_img(np_img: np.ndarray) -> np.ndarray:
   try:
       if not isinstance(np_img, np.ndarray):
           raise ValueError("Input must be numpy array")
           
       if len(np_img.shape) == 2:
           np_img = np_img[:, :, np.newaxis]
           
       if np_img.shape[2] not in [1, 3, 4]:
           raise ValueError(f"Invalid number of channels: {np_img.shape[2]}")
           
       np_img = np.transpose(np_img, (2, 0, 1))
       np_img = np_img.astype("float32") / 255.0
       
       return np_img
       
   except Exception as e:
       logger.error(f"Image normalization failed: {str(e)}")
       raise ValueError(f"Failed to normalize image: {str(e)}")

@torch.no_grad()
def process_image(
   model: torch.jit.ScriptModule,
   image: torch.Tensor,
   mask: torch.Tensor,
) -> torch.Tensor:
   try:
       device = get_device()
       if not torch.is_tensor(image) or not torch.is_tensor(mask):
           raise ValueError("Inputs must be torch tensors")
           
       image = image.to(device)
       mask = mask.to(device)
       
       output = model(image, mask)
       return output.cpu()
       
   except Exception as e:
       logger.error(f"Image processing failed: {str(e)}")
       raise ModelError(f"Failed to process image: {str(e)}")

if __name__ == "__main__":
   try:
       test_url = "https://example.com/model.pth"
       test_config = ModelConfig(url=test_url)
       model = load_jit_model(test_url)
       logger.info("Test completed successfully")
   except Exception as e:
       logger.error(f"Test failed: {str(e)}")