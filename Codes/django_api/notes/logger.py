# jobscheduler/logger.py
import base64
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import requests
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

logger = logging.getLogger(__name__)


class CustomRunLogger:
    """
    A flexible, robust logger for tracking arbitrary metrics during training.
    Supports scalars, images, figures, tables, and text.
    Features:
        - Automatic retries on failure (with exponential backoff)
        - Optional async sending (fire-and-forget)
        - Optional upload to media storage (instead of base64)
        - Comprehensive error logging
    """

    def __init__(
        self,
        run_id: str,
        api_base_url: str = None,
        async_mode: bool = True,
        upload_images: bool = False,
        max_retries: int = 3,
    ):
        """
        Args:
            run_id: UUID of the training run (Job)
            api_base_url: Base URL of the Django API (default: settings.API_BASE_URL)
            async_mode: If True, send metrics in a background thread (non‑blocking)
            upload_images: If True, save images to media storage and send URL instead of base64
            max_retries: Number of retry attempts on failure
        """
        self.run_id = run_id
        self.api_base_url = api_base_url or settings.API_BASE_URL
        self.async_mode = async_mode
        self.upload_images = upload_images
        self.max_retries = max_retries
        self._executor = ThreadPoolExecutor(max_workers=2) if async_mode else None
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
        """Log a scalar value (loss, accuracy, etc.)"""
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
        """
        Log an image (PIL Image, numpy array, or file path).
        If upload_images=True, the image is saved to media storage and its URL is sent.
        Otherwise, it is sent as a base64 data URI.
        """
        try:
            import numpy as np
            from PIL import Image

            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif isinstance(image, str):
                image = Image.open(image)

            if self.upload_images:
                # Save to media storage (async if configured)
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                filename = f"metrics/{self.run_id}/{name}_step_{step}.png"
                saved_path = default_storage.save(
                    filename, ContentFile(buffer.getvalue())
                )
                value = default_storage.url(saved_path)
                metric_type = "image_url"
            else:
                # Convert to base64
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode()
                value = f"data:image/png;base64,{img_str}"
                metric_type = "image"

            self._send_metric(
                {
                    "name": name,
                    "type": metric_type,
                    "value": value,
                    "step": step,
                    "epoch": None,
                    "caption": caption,
                }
            )
        except Exception as e:
            logger.exception(f"Failed to log image '{name}': {e}")

    def log_figure(self, name: str, figure: plt.Figure, step: int):
        """Log a matplotlib figure (converted to image)."""
        try:
            buffer = BytesIO()
            figure.savefig(buffer, format="png", bbox_inches="tight")
            buffer.seek(0)
            from PIL import Image

            pil_image = Image.open(buffer)
            self.log_image(name, pil_image, step, caption="Matplotlib figure")
            plt.close(figure)
        except Exception as e:
            logger.exception(f"Failed to log figure '{name}': {e}")

    def log_table(self, name: str, data: List[Dict], step: int):
        """Log a table (list of dictionaries)."""
        self._send_metric(
            {"name": name, "type": "table", "value": data, "step": step, "epoch": None}
        )

    def log_text(self, name: str, text: str, step: int):
        """Log arbitrary text."""
        self._send_metric(
            {"name": name, "type": "text", "value": text, "step": step, "epoch": None}
        )

    def _send_metric(self, metric: Dict[str, Any]):
        """Send metric to the backend API with retries and optional async mode."""

        def _send():
            url = f"{self.api_base_url}/api/runs/{self.run_id}/metrics/"
            retries = 0
            while retries < self.max_retries:
                try:
                    response = self.session.post(url, json=metric, timeout=10)
                    response.raise_for_status()
                    return
                except requests.exceptions.RequestException as e:
                    retries += 1
                    wait = 2**retries  # exponential backoff
                    logger.warning(
                        f"Metric send failed (attempt {retries}/{self.max_retries}): {e}. Retrying in {wait}s."
                    )
                    if retries < self.max_retries:
                        time.sleep(wait)
                    else:
                        logger.error(
                            f"Failed to log metric after {self.max_retries} attempts: {metric}"
                        )

        if self.async_mode and self._executor:
            self._executor.submit(_send)
        else:
            _send()

    def close(self):
        """Shut down the background executor (call when training ends)."""
        if self._executor:
            self._executor.shutdown(wait=True)
