# AIML Integration API Specification

## Authentication API

**Endpoint**

- Method: `POST`
- URL: `/api/v1/auth`

**Request**

```json
{
  "username": "abc",
  "password": "password"
}
```

**Success Response**

```json
{
  "access_token": "<token>",
  "expires_in": 3600
}
```

**Failure Response**

```json
{
  "message": "Authentication failed."
}
```

## Training API

**Endpoint**

- Method: `POST`
- URL: `/api/v1/training`
- Headers: `Authorization: Bearer <access_token>`

**Request**

```json
{
  "transactionId": "TRN202606080001",
  "months": 3
}
```

**Accepted Response**

```json
{
  "transactionId": "TRN202606080001",
  "status": "ACCEPTED",
  "modelVersion": "v1.2",
  "message": "Request accepted for processing."
}
```

**Failure Response**

```json
{
  "transactionId": "TRN202606080001",
  "status": "FAILED",
  "message": "<Actual Failure Reason>"
}
```

**DB Behavior**

- `transactionId` must already exist in `digital_collections.ai_configurations`.
- API updates that existing row to `ACCEPTED`, then later to `COMPLETED` or `FAILED`.

## Inference API

**Endpoint**

- Method: `POST`
- URL: `/api/v1/inference`
- Headers: `Authorization: Bearer <access_token>`

**Request**

```json
{
  "transactionId": "TRN202606080002"
}
```

**Accepted Response**

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

**Failure Response**

```json
{
  "transactionId": "TRN202606080002",
  "status": "FAILED",
  "message": "<Actual Failure Reason>"
}
```

**DB Behavior**

- `transactionId` must already exist in `digital_collections.ai_configurations`.
- API updates that existing row to `ACCEPTED`, then later to `COMPLETED` or `FAILED`.
- Drift + accuracy persist in `digital_collections.ai_configurations`.`drift` and `digital_collections.ai_configurations`.`accuracy`.

## Model Status API

**Endpoint**

- Method: `GET`
- URL: `/api/status`

**Response**

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

Rules:

- `lastTrainingDate` = latest completed `TRAINING` `modified_on` in `ai_configurations`
- `lastInferenceDate` = latest completed `INFERENCE` `modified_on` in `ai_configurations`
- `currentAccuracy` = latest completed inference `accuracy`
- `currentDrift` = latest completed inference `drift`

## Health API

**Endpoint**

- Method: `GET`
- URL: `/api/health`

## Inference Workbook Export

If `API_EXPORT_AFTER_INFERENCE=true`, inference completion writes 2 files to configured `API_SFTP_EXPORT_PATH`:

- `MD_UB_DATASET_DDMMYYYY_01.xlsx`
- `MD_UB_SCHEDULER_DDMMYYYY_01.xlsx`

Dataset columns:

- `Name`
- `Query`

Scheduler columns:

- `Name`
- `Mode`
- `Date`
- `Time`
- `Dataset Name`
- `Vendor`
- `Active`
- `Reason`
