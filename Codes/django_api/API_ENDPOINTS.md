# API Endpoints Documentation

Base URL: [http://localhost:8000/api](http://localhost:8000/api)

## Authentication

### POST /auth/register/

Register a new user.

Request:

```json
{
  "username": "string",
  "password": "string",
  "email": "string"
}
```

Response (201):

```json
{
  "token": "string",
  "username": "string",
  "email": "string",
  "user_id": "uuid"
}
```

### POST /auth/login/

Login and obtain token.

Request:

```json
{
  "username": "string",
  "password": "string"
}
```

Response (200):

```json
{
  "token": "string",
  "username": "string",
  "email": "string",
  "user_id": "uuid"
}
```

### POST /auth/logout/

Logout user.

Requires token.

### GET /auth/user/

Get current user profile.

### PATCH /auth/profile/

Update profile.

Request:

```json
{
  "first_name": "string",
  "last_name": "string",
  "email": "string"
}
```

### POST /auth/change-password/

Change password.

Request:

```json
{
  "old_password": "string",
  "new_password": "string"
}
```

### POST /auth/refresh-token/

Refresh token.

---

## Configuration

### GET /config/attentions/

Get all available attention mechanisms.

### GET /config/compressors/

Get all compression methods.

### GET /config/models/

Get supported models.

### GET /config/tasks/

Get supported tasks.

### GET /config/methods/

Get supported training methods.

### GET /config/presets/

Get supported presets.

---

## System

### GET /system/status/

System CPU, memory, disk, uptime.

### GET /system/gpu/

GPU information (CUDA or MPS).

### GET /health/

Simple health check.

### GET /version/

API version and Django version.

---

## Experiments

### GET /experiments/

List experiments.

### POST /experiments/

Create experiment.

Request:

```json
{
  "name": "string",
  "description": "string"
}
```

### GET /experiments/<id>/

Get experiment details.

### PATCH /experiments/<id>/

Update experiment.

### DELETE /experiments/<id>/

Delete experiment.

---

## Jobs

### POST /train/

Start training job.

### POST /evaluate/

Start evaluation job.

### POST /compress/

Start compression job.

### GET /runs/<id>/

Get run details.

### POST /runs/<id>/stop/

Stop a running job.

---

## Authentication Header

Use the following header for protected endpoints:

```text
Authorization: Bearer <token>
```
