# jobscheduler/logger.py
import base64
import io
import requests
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
from PIL import Image
from django.conf import settings

class CustomRunLogger:
    """
    A flexible logger for tracking arbitrary metrics during training.
    Sends data to the backend API.
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
            'name': name, 'type': 'scalar', 'value': value, 'step': step, 'epoch': epoch
        })

    def log_image(self, name: str, image, step: int, caption: str = None):
        """Log an image (PIL Image, numpy array, or file path) – converts to base64."""
        buffered = io.BytesIO()
        if isinstance(image, Image.Image):
            image.save(buffered, format="PNG")
        elif isinstance(image, str):
            Image.open(image).save(buffered, format="PNG")
        else:
            # Assume numpy array
            from PIL import Image as PILImage
            PILImage.fromarray(image).save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        self._send_metric({
            'name': name, 'type': 'image',
            'value': f"data:image/png;base64,{img_str}",
            'step': step, 'epoch': None, 'caption': caption
        })

    def log_figure(self, name: str, figure: plt.Figure, step: int):
        """Log a matplotlib figure."""
        buf = io.BytesIO()
        figure.savefig(buf, format='png', bbox_inches='tight')
        img_str = base64.b64encode(buf.getvalue()).decode()
        self._send_metric({
            'name': name, 'type': 'figure',
            'value': f"data:image/png;base64,{img_str}",
            'step': step, 'epoch': None
        })
        plt.close(figure)

    def log_table(self, name: str, data: List[Dict], step: int):
        """Log a table (list of dictionaries)."""
        self._send_metric({
            'name': name, 'type': 'table', 'value': data, 'step': step, 'epoch': None
        })

    def _send_metric(self, metric: Dict[str, Any]):
        try:
            url = f"{self.api_base_url}/api/runs/{self.run_id}/metrics/"
            response = self.session.post(url, json=metric)
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to log metric: {e}")



            # jobscheduler/logger.py
import base64
import json
import requests
import logging
from typing import Dict, Any, Optional, Union, List
from io import BytesIO
import matplotlib.pyplot as plt
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class CustomRunLogger:
    """
    A production‑ready logger for tracking arbitrary metrics during training.
    Supports scalars, images, matplotlib figures, tables, and text.
    Images are sent as base64 data URIs by default; can be configured to upload to media storage.
    """
    
    def __init__(self, run_id: str, api_base_url: str = None, upload_images: bool = False):
        """
        Args:
            run_id: UUID of the training run (Job)
            api_base_url: Base URL for the Django API (default: settings.API_BASE_URL)
            upload_images: If True, images are saved to media storage and only the URL is sent.
        """
        self.run_id = run_id
        self.api_base_url = api_base_url or settings.API_BASE_URL
        self.upload_images = upload_images
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
        """
        Log an image (PIL Image, numpy array, or file path).
        If upload_images=True, the image is saved to media storage and its URL is sent.
        Otherwise, it is sent as a base64 data URI.
        """
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif isinstance(image, str):
                image = Image.open(image)
            
            if self.upload_images:
                # Save to media storage
                buffer = BytesIO()
                image.save(buffer, format='PNG')
                filename = f"metrics/{self.run_id}/{name}_step_{step}.png"
                saved_path = default_storage.save(filename, ContentFile(buffer.getvalue()))
                image_data = default_storage.url(saved_path)
                value = image_data
            else:
                # Convert to base64
                buffer = BytesIO()
                image.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                value = f"data:image/png;base64,{img_str}"
            
            self._send_metric({
                'name': name,
                'type': 'image',
                'value': value,
                'step': step,
                'epoch': None,
                'caption': caption
            })
        except Exception as e:
            logger.error(f"Failed to log image: {e}")
    
    def log_figure(self, name: str, figure: plt.Figure, step: int):
        """Log a matplotlib figure (converted to image)."""
        try:
            buffer = BytesIO()
            figure.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            pil_image = Image.open(buffer)
            # Reuse log_image logic
            self.log_image(name, pil_image, step, caption="Matplotlib figure")
            plt.close(figure)
        except Exception as e:
            logger.error(f"Failed to log figure: {e}")
    
    def log_table(self, name: str, data: List[Dict], step: int):
        """Log a table (list of dictionaries)."""
        self._send_metric({
            'name': name,
            'type': 'table',
            'value': data,
            'step': step,
            'epoch': None
        })
    
    def log_text(self, name: str, text: str, step: int):
        """Log arbitrary text (e.g., hyperparameters, notes)."""
        self._send_metric({
            'name': name,
            'type': 'text',
            'value': text,
            'step': step,
            'epoch': None
        })
    
    def _send_metric(self, metric: Dict[str, Any]):
        """Send metric to the backend API with error handling."""
        try:
            url = f"{self.api_base_url}/api/runs/{self.run_id}/metrics/"
            response = self.session.post(url, json=metric, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to log metric: {e}")


# jobscheduler/logger.py
import base64
import json
import requests
import logging
from typing import Dict, Any, Optional, Union, List
from io import BytesIO
import matplotlib.pyplot as plt
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class CustomRunLogger:
    """
    A production‑ready logger for tracking arbitrary metrics during training.
    Supports scalars, images, matplotlib figures, tables, and text.
    Images are sent as base64 data URIs by default; can be configured to upload to media storage.
    """
    
    def __init__(self, run_id: str, api_base_url: str = None, upload_images: bool = False):
        """
        Args:
            run_id: UUID of the training run (Job)
            api_base_url: Base URL for the Django API (default: settings.API_BASE_URL)
            upload_images: If True, images are saved to media storage and only the URL is sent.
        """
        self.run_id = run_id
        self.api_base_url = api_base_url or settings.API_BASE_URL
        self.upload_images = upload_images
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
        """
        Log an image (PIL Image, numpy array, or file path).
        If upload_images=True, the image is saved to media storage and its URL is sent.
        Otherwise, it is sent as a base64 data URI.
        """
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif isinstance(image, str):
                image = Image.open(image)
            
            if self.upload_images:
                # Save to media storage
                buffer = BytesIO()
                image.save(buffer, format='PNG')
                filename = f"metrics/{self.run_id}/{name}_step_{step}.png"
                saved_path = default_storage.save(filename, ContentFile(buffer.getvalue()))
                image_data = default_storage.url(saved_path)
                value = image_data
            else:
                # Convert to base64
                buffer = BytesIO()
                image.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                value = f"data:image/png;base64,{img_str}"
            
            self._send_metric({
                'name': name,
                'type': 'image',
                'value': value,
                'step': step,
                'epoch': None,
                'caption': caption
            })
        except Exception as e:
            logger.error(f"Failed to log image: {e}")
    
    def log_figure(self, name: str, figure: plt.Figure, step: int):
        """Log a matplotlib figure (converted to image)."""
        try:
            buffer = BytesIO()
            figure.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            pil_image = Image.open(buffer)
            # Reuse log_image logic
            self.log_image(name, pil_image, step, caption="Matplotlib figure")
            plt.close(figure)
        except Exception as e:
            logger.error(f"Failed to log figure: {e}")
    
    def log_table(self, name: str, data: List[Dict], step: int):
        """Log a table (list of dictionaries)."""
        self._send_metric({
            'name': name,
            'type': 'table',
            'value': data,
            'step': step,
            'epoch': None
        })
    
    def log_text(self, name: str, text: str, step: int):
        """Log arbitrary text (e.g., hyperparameters, notes)."""
        self._send_metric({
            'name': name,
            'type': 'text',
            'value': text,
            'step': step,
            'epoch': None
        })
    
    def _send_metric(self, metric: Dict[str, Any]):
        """Send metric to the backend API with error handling."""
        try:
            url = f"{self.api_base_url}/api/runs/{self.run_id}/metrics/"
            response = self.session.post(url, json=metric, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to log metric: {e}")