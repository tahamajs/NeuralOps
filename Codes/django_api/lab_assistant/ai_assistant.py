"""AI-Assisted Labeling Engine — uses a model to pre-label data."""
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class AIAssistant:
    """
    Uses a pre-trained model to generate pre-annotations for data tasks.
    Supports: text classification, NER, sentiment analysis.
    """

    def __init__(self, model_id: str = 'bert-base-uncased'):
        self.model_id = model_id
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
                # Determine task type from model_id
                if 'sentiment' in self.model_id or 'emotion' in self.model_id:
                    self._pipeline = pipeline('sentiment-analysis', model=self.model_id)
                elif 'ner' in self.model_id:
                    self._pipeline = pipeline('ner', model=self.model_id)
                else:
                    self._pipeline = pipeline('text-classification', model=self.model_id)
            except Exception as e:
                logger.error(f"Failed to load model {self.model_id}: {e}")
                self._pipeline = None
        return self._pipeline

    def predict(self, text: str, labels: Optional[List[str]] = None) -> dict:
        """Run inference on a single text and return predictions."""
        if not self.pipeline:
            return {'predictions': [], 'confidence': 0.0}

        try:
            if labels:
                # If labels are provided, use zero-shot classification
                from transformers import pipeline as zs_pipeline
                classifier = zs_pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
                result = classifier(text, labels)
                return {
                    'predictions': [
                        {'label': lbl, 'score': round(scr, 4)}
                        for lbl, scr in zip(result['labels'], result['scores'])
                    ],
                    'confidence': round(result['scores'][0], 4) if result['scores'] else 0.0,
                }
            else:
                result = self.pipeline(text)
                if isinstance(result, list) and len(result) > 0:
                    first = result[0]
                    return {
                        'predictions': [{'label': first.get('label', first.get('entity_group', '')), 'score': round(first.get('score', 0), 4)}],
                        'confidence': round(first.get('score', 0), 4),
                    }
        except Exception as e:
            logger.error(f"AI prediction failed: {e}")

        return {'predictions': [], 'confidence': 0.0}

    def pre_label_task(self, task) -> bool:
        """Pre-label a single DataTask and store predictions."""
        if not self.pipeline:
            return False

        text = task.data.get('text', '')
        if not text:
            return False

        labels = task.project.labels
        result = self.predict(text, labels if labels else None)

        task.ai_predictions = result['predictions']
        task.ai_confidence = result['confidence']
        task.save(update_fields=['ai_predictions', 'ai_confidence'])
        return True