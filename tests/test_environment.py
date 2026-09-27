import torch
import cv2
import mediapipe
import fastapi
import numpy as np

def test_cuda_available():
    assert torch.cuda.is_available(), "CUDA must be available for RTX 5060 Ti"
    assert torch.cuda.device_count() >= 1

def test_dependencies_importable():
    import scipy
    import av
    import sentencepiece
    import websockets
    assert True
