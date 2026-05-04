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