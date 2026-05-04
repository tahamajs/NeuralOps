# jobscheduler/logger.py
import base64
import json
import requests
from typing import Dict, Any, Optional, Union, List
from io import BytesIO
import matplotlib.pyplot as plt
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

class CustomRunLogger:
    """
    A flexible logger for tracking arbitrary metrics during training.
    Can log scalars, images, tables, figures, and text.
    """
    
    def __init__(self, run_id: str, api_base_url: str = None):
        self.run_id = run_id
        self.api_base_url = api_base_url or settings.API_BASE_URL
        self._session = None
    
    @property
    def session(self):
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                'Authorization': f'Token {settings.ML_TOKEN}',
                'Content-Type': 'application/json'
            })
        return self._session
    
    def log_scalar(self, name: str, value: float, step: int, epoch: Optional[float] = None):
        """Log a scalar value (loss, accuracy, etc.)"""
        self._send_metric({
            'name': name,
            'type': 'scalar',
            'value': value,
            'step': step,
            'epoch': epoch
        })
    
    def log_image(self, name: str, image, step: int, caption: str = None):
        """Log an image (PIL Image, numpy array, or file path)"""
        from PIL import Image
        import numpy as np
        
        # Convert to base64 or upload to media storage
        if isinstance(image, Image.Image):
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            image_data = f"data:image/png;base64,{img_str}"
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            pil_img = Image.fromarray(image)
            buffered = BytesIO()
            pil_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            image_data = f"data:image/png;base64,{img_str}"
        elif isinstance(image, str):
            # Assume it's a file path
            with open(image, 'rb') as f:
                img_str = base64.b64encode(f.read()).decode()
                image_data = f"data:image/png;base64,{img_str}"
        else:
            raise ValueError("Unsupported image type")
        
        self._send_metric({
            'name': name,
            'type': 'image',
            'value': image_data,
            'step': step,
            'epoch': None,
            'caption': caption
        })
    
    def log_figure(self, name: str, figure: plt.Figure, step: int):
        """Log a matplotlib figure"""
        buffered = BytesIO()
        figure.savefig(buffered, format='png', bbox_inches='tight')
        img_str = base64.b64encode(buffered.getvalue()).decode()
        image_data = f"data:image/png;base64,{img_str}"
        
        self._send_metric({
            'name': name,
            'type': 'image',
            'value': image_data,
            'step': step,
            'epoch': None
        })
        plt.close(figure)
    
    def log_table(self, name: str, data: List[Dict], step: int):
        """Log a table (list of dictionaries)"""
        self._send_metric({
            'name': name,
            'type': 'table',
            'value': data,
            'step': step,
            'epoch': None
        })
    
    def log_text(self, name: str, text: str, step: int):
        """Log arbitrary text"""
        self._send_metric({
            'name': name,
            'type': 'text',
            'value': text,
            'step': step,
            'epoch': None
        })
    
    def _send_metric(self, metric: Dict[str, Any]):
        """Send metric to the backend API"""
        try:
            url = f"{self.api_base_url}/api/runs/{self.run_id}/metrics/"
            response = self.session.post(url, json=metric)
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to log metric: {e}")