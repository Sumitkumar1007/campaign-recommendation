# AIML Campaign Recommendation API Guide for Digital Team

## Overview

This document is for the Digital application team to integrate with the AIML recommendation service.

Base URL for testing:

```text
http://10.10.0.10:8040
```

Note:

- This URL is only for testing.
- Actual production/final URL will be shared later.

Implemented endpoints:

- `POST /api/v1/auth`
- `POST /api/v1/training`
- `POST /api/v1/inference`
- `GET /api/status`
- `GET /api/health`
- `GET /api/v1/transactions/{transactionId}`

## Integration Flow

1. Call `POST /api/v1/auth` to get bearer token.
2. Ensure Digital application has already inserted the required `transactionId` row into `digital_collections.ai_configurations`.
3. Call either:
   - `POST /api/v1/training`
   - `POST /api/v1/inference`
4. API returns `202 ACCEPTED` for async processing.
5. Poll `GET /api/v1/transactions/{transactionId}` to get final job status.
6. Optionally call `GET /api/status` for current model health/status.

## Authentication

All endpoints except `/api/v1/auth` and `/api/health` require:

```text
Authorization: Bearer <access_token>
```

Current deployment token expiry:

- `expires_in = 28800` seconds
- `8 hours`

Note:

- `expires_in` is controlled by server configuration.

## Common Headers

For all `POST` APIs:

```text
Content-Type: application/json
```

For protected APIs:

```text
Authorization: Bearer <access_token>
```

## Endpoint Summary

| Endpoint | Method | Auth Required | Purpose |
| --- | --- | --- | --- |
| `/api/v1/auth` | `POST` | No | Get bearer token |
| `/api/v1/training` | `POST` | Yes | Trigger model training |
| `/api/v1/inference` | `POST` | Yes | Trigger inference and recommendation generation |
| `/api/status` | `GET` | Yes | Get current model status |
| `/api/health` | `GET` | No | Health check |
| `/api/v1/transactions/{transactionId}` | `GET` | Yes | Check transaction/job status |

---

## 1. Authentication API

### Endpoint

- Method: `POST`
- URL: `/api/v1/auth`

### Request

```json
{
  "username": "aiml",
  "password": "aiml"
}
```

### Success Response

HTTP `200 OK`

```json
{
  "access_token": "<token>",
  "expires_in": 28800
}
```

### Failure Response

HTTP `401 Unauthorized`

```json
{
  "message": "Authentication failed."
}
```

---

## 2. Training API

### Endpoint

- Method: `POST`
- URL: `/api/v1/training`

### Purpose

Triggers manual model training using historical data.

### Pre-condition

`transactionId` must already exist in `digital_collections.ai_configurations`.

### Request

```json
{
  "transactionId": "TRN202606080001",
  "months": 3
}
```

### Success Response

HTTP `202 Accepted`

```json
{
  "transactionId": "TRN202606080001",
  "status": "ACCEPTED",
  "modelVersion": "v1.1.4",
  "message": "Request accepted for processing."
}
```

### Notes

- `modelVersion` in training response is the next model version that will be assigned if training completes successfully.
- This is an async API. Final completion must be checked separately.

### Failure Responses

#### Missing transactionId

HTTP `400 Bad Request`

```json
{
  "transactionId": "",
  "status": "FAILED",
  "message": "transactionId is required."
}
```

#### Missing months

HTTP `400 Bad Request`

```json
{
  "transactionId": "TRN202606080001",
  "status": "FAILED",
  "message": "months is required."
}
```

#### Unexpected field

HTTP `400 Bad Request`

```json
{
  "transactionId": "TRN202606080001",
  "status": "FAILED",
  "message": "Unexpected fields: model"
}
```

#### transactionId not present in `ai_configurations`

HTTP `404 Not Found`

```json
{
  "transactionId": "TRN202606080001",
  "status": "FAILED",
  "message": "transactionId TRN202606080001 not found in ai_configurations."
}
```

#### Same transaction already running

HTTP `409 Conflict`

```json
{
  "transactionId": "TRN202606080001",
  "status": "FAILED",
  "message": "Job already running for TRN202606080001."
}
```

#### Unexpected server error

HTTP `500 Internal Server Error`

```json
{
  "message": "<server error message>"
}
```

---

## 3. Inference API

### Endpoint

- Method: `POST`
- URL: `/api/v1/inference`

### Purpose

Triggers inference to generate recommendation outputs and current drift/accuracy snapshot.

### Pre-condition

`transactionId` must already exist in `digital_collections.ai_configurations`.

### Request

```json
{
  "transactionId": "TRN202606080002"
}
```

### Success Response

HTTP `202 Accepted`

```json
{
  "transactionId": "TRN202606080002",
  "status": "ACCEPTED",
  "driftPercentage": 4.8,
  "currentAccuracy": 78.5,
  "modelVersion": "v1.1.3",
  "message": "Request accepted for processing."
}
```

### Notes

- `modelVersion` in inference response is the currently active trained model version.
- `currentAccuracy` and `driftPercentage` are the latest available metrics snapshot at the time the request is accepted.
- This is an async API. Final completion must be checked separately.

### Failure Responses

#### Missing transactionId

HTTP `400 Bad Request`

```json
{
  "transactionId": "",
  "status": "FAILED",
  "message": "transactionId is required."
}
```

#### Unexpected field

HTTP `400 Bad Request`

```json
{
  "transactionId": "TRN202606080002",
  "status": "FAILED",
  "message": "Unexpected fields: sourceMonth"
}
```

#### transactionId not present in `ai_configurations`

HTTP `404 Not Found`

```json
{
  "transactionId": "TRN202606080002",
  "status": "FAILED",
  "message": "transactionId TRN202606080002 not found in ai_configurations."
}
```

#### Same transaction already running

HTTP `409 Conflict`

```json
{
  "transactionId": "TRN202606080002",
  "status": "FAILED",
  "message": "Job already running for TRN202606080002."
}
```

#### Unexpected server error

HTTP `500 Internal Server Error`

```json
{
  "message": "<server error message>"
}
```

---

## 4. Transaction Status API

### Endpoint

- Method: `GET`
- URL: `/api/v1/transactions/{transactionId}`

### Purpose

Returns current status of a submitted training/inference transaction.

### Example

```text
GET /api/v1/transactions/TRN202606080002
```

### Success Response

HTTP `200 OK`

```json
{
  "transactionId": "TRN202606080002",
  "type": "INFERENCE",
  "status": "COMPLETED",
  "message": "Metrics generated successfully",
  "modelVersion": "v1.1.3",
  "accuracy": "78.5",
  "drift": "4.8",
  "createdOn": "2026-06-11T05:20:00",
  "modifiedOn": "2026-06-11T05:22:10",
  "trainingWindow": "10 days"
}
```

### Possible `status` values

- `ACCEPTED`
- `COMPLETED`
- `FAILED`

### Failure Responses

#### Missing/invalid bearer token

HTTP `401 Unauthorized`

```json
{
  "message": "Missing bearer token."
}
```

or

```json
{
  "message": "Invalid token signature."
}
```

or

```json
{
  "message": "Token expired."
}
```

#### transactionId not found

HTTP `404 Not Found`

```json
{
  "message": "Transaction TRN202606080002 not found."
}
```

---

## 5. Model Status API

### Endpoint

- Method: `GET`
- URL: `/api/status`

### Purpose

Returns latest training and inference status along with current model health.

### Success Response

HTTP `200 OK`

```json
{
  "modelVersion": "v1.1.3",
  "lastTrainingDate": "2026-06-10T09:30:00",
  "lastInferenceDate": "2026-06-10T09:45:00",
  "currentAccuracy": 78.5,
  "currentDrift": 4.8,
  "modelStatus": "Healthy",
  "latestTrainingStatus": "COMPLETED",
  "latestInferenceStatus": "COMPLETED"
}
```

### `modelStatus` meaning

- `Healthy` when accuracy is `>= 70.0`
- `Not Healthy` when accuracy is `< 70.0`
- `Unknown` when accuracy is not available

### Failure Responses

#### Missing/invalid bearer token

HTTP `401 Unauthorized`

```json
{
  "message": "Missing bearer token."
}
```

#### Unexpected server error

HTTP `500 Internal Server Error`

```json
{
  "message": "<server error message>"
}
```

---

## 6. Health API

### Endpoint

- Method: `GET`
- URL: `/api/health`

### Purpose

Simple health endpoint for service and DB connectivity checks.

### Success Response

HTTP `200 OK`

```json
{
  "status": "ok",
  "service": "aiml-integration-api",
  "dbStatus": "ok",
  "dbError": null,
  "modelVersion": "v1.1.3"
}
```

Possible `status` values:

- `ok`
- `degraded`

Possible `dbStatus` values:

- `ok`
- `error`
- `not-configured`

---

## Common Error Codes

| HTTP Code | Meaning | Typical APIs |
| --- | --- | --- |
| `200` | Request succeeded | auth, health, status, transaction lookup |
| `202` | Async request accepted | training, inference |
| `400` | Bad request / validation error | training, inference |
| `401` | Authentication/authorization failure | all protected APIs |
| `404` | transactionId not found / route not found | training, inference, transaction lookup |
| `405` | Wrong HTTP method | all endpoints |
| `409` | Same transaction already running | training, inference |
| `500` | Unexpected server error | any endpoint |

## Error Response Formats

### Protected API auth error

```json
{
  "message": "Missing bearer token."
}
```

### Training/Inference validation error

```json
{
  "transactionId": "TRN202606080002",
  "status": "FAILED",
  "message": "<Actual Failure Reason>"
}
```

### Generic server error

```json
{
  "message": "<server error message>"
}
```

## Database Update Behavior

Digital team should be aware of the following behavior:

- Digital application inserts `transactionId` row first in `digital_collections.ai_configurations`.
- AIML API does not create the transaction row.
- AIML API updates the existing row during processing.

Important columns updated by AIML:

- `type`
- `status`
- `message`
- `model_version`
- `accuracy`
- `drift`
- `processing_time_ms`
- `training_window`
- `modified_by`
- `modified_on`

## Polling Recommendation

For `training` and `inference`:

1. Submit request.
2. If response is `202`, wait a few seconds.
3. Poll `GET /api/v1/transactions/{transactionId}` until:
   - `status = COMPLETED`
   - or `status = FAILED`

