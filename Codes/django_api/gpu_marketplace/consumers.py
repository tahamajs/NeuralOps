import json
import logging

from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer
from django.utils import timezone

from .models import ComputeJob, GPUListing, JobEvent

logger = logging.getLogger(__name__)


class ProviderConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for GPU provider agents.
    Allows agents to connect, register their listing, stream metrics/snapshots,
    and receive compute jobs via Channels group communication.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.listing_id = None
        self.group_name = None

    async def connect(self):
        await self.accept()
        logger.info("Provider Agent websocket connected")

    async def disconnect(self, close_code):
        if self.group_name:
            # Leave group
            await self.channel_layer.group_discard(self.group_name, self.channel_name)
            # Mark listing as offline
            await self.update_listing_status(self.listing_id, "offline")
            logger.info(f"Provider Agent listing {self.listing_id} disconnected")

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
        except json.JSONDecodeError:
            return

        msg_type = data.get("type")

        if msg_type == "register":
            self.listing_id = data.get("listing_id")
            if self.listing_id:
                self.group_name = f"provider_listing_{self.listing_id}"
                # Join listing-specific group
                await self.channel_layer.group_add(self.group_name, self.channel_name)
                # Mark listing as online
                await self.update_listing_status(self.listing_id, "online")
                logger.info(f"Provider Agent registered listing: {self.listing_id}")

        elif msg_type == "system_stats":
            # Save or log system stats (can be extended to update models)
            pass

        elif msg_type == "gpu_snapshot":
            # Live GPU metrics snapshot
            pass

        elif msg_type == "job_update":
            job_id = data.get("job_id")
            status = data.get("status")
            update_data = data.get("data") or {}
            if job_id and status:
                await self.update_job_status(
                    job_id, status, update_data.get("container_id")
                )

    # Handler for messages sent from celery/matching engine to group
    async def send_job(self, event):
        await self.send(text_data=json.dumps({"type": "new_job", "job": event["job"]}))

    async def cancel_job(self, event):
        await self.send(
            text_data=json.dumps({"type": "cancel_job", "job_id": event["job_id"]})
        )

    @database_sync_to_async
    def update_listing_status(self, listing_id, status):
        try:
            listing = GPUListing.objects.get(id=listing_id)
            listing.status = status
            listing.save(update_fields=["status"])
        except GPUListing.DoesNotExist:
            pass

    @database_sync_to_async
    def update_job_status(self, job_id, status, container_id=None):
        try:
            job = ComputeJob.objects.get(id=job_id)
            job.status = status
            if status == "running":
                job.started_at = timezone.now()
            elif status in ["completed", "failed"]:
                job.completed_at = timezone.now()

            update_fields = ["status"]
            if status == "running":
                update_fields.append("started_at")
            elif status in ["completed", "failed"]:
                update_fields.append("completed_at")

            job.save(update_fields=update_fields)
            JobEvent.objects.create(job=job, event_type=status)
        except ComputeJob.DoesNotExist:
            pass
