import base64
import logging
from io import BytesIO
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import requests
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from PIL import Image

logger = logging.getLogger(__name__)


class CustomRunLogger:
    def __init__(
        self, run_id: str, api_base_url: str = None, upload_images: bool = False
    ):
        self.run_id = run_id
        self.api_base_url = api_base_url or settings.API_BASE_URL
        self.upload_images = upload_images
        self._session = None

    @property
    def session(self):
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(
                {
                    "Authorization": f"Token {settings.ML_TOKEN}",
                    "Content-Type": "application/json",
                }
            )
        return self._session

    def log_scalar(
        self, name: str, value: float, step: int, epoch: Optional[float] = None
    ):
        self._send_metric(
            {
                "name": name,
                "type": "scalar",
                "value": value,
                "step": step,
                "epoch": epoch,
            }
        )

    def log_image(self, name: str, image, step: int, caption: str = None):
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif isinstance(image, str):
                image = Image.open(image)

            if self.upload_images:
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                filename = f"metrics/{self.run_id}/{name}_step_{step}.png"
                saved_path = default_storage.save(
                    filename, ContentFile(buffer.getvalue())
                )
                value = default_storage.url(saved_path)
            else:
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode()
                value = f"data:image/png;base64,{img_str}"

            self._send_metric(
                {
                    "name": name,
                    "type": "image",
                    "value": value,
                    "step": step,
                    "epoch": None,
                    "caption": caption,
                }
            )
        except Exception as e:
            logger.error(f"Failed to log image: {e}")

    def log_figure(self, name: str, figure: plt.Figure, step: int):
        try:
            buffer = BytesIO()
            figure.savefig(buffer, format="png", bbox_inches="tight")
            buffer.seek(0)
            self.log_image(name, Image.open(buffer), step, caption="Matplotlib figure")
            plt.close(figure)
        except Exception as e:
            logger.error(f"Failed to log figure: {e}")

    def log_table(self, name: str, data: List[Dict], step: int):
        self._send_metric(
            {"name": name, "type": "table", "value": data, "step": step, "epoch": None}
        )

    def log_text(self, name: str, text: str, step: int):
        self._send_metric(
            {"name": name, "type": "text", "value": text, "step": step, "epoch": None}
        )

    def _send_metric(self, metric: Dict[str, Any]):
        try:
            url = f"{self.api_base_url}/api/runs/{self.run_id}/metrics/"
            response = self.session.post(url, json=metric, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to log metric: {e}")
