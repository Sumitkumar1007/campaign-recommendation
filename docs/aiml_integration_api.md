# AIML Integration API

Implemented endpoints:

- `POST /api/v1/auth`
- `POST /api/v1/training`
- `POST /api/v1/inference`
- `GET /api/status`
- `GET /api/health`
- `GET /api/v1/transactions/{transactionId}`

Run service:

```bash
./venv/bin/python scripts/run_api_service.py --host 0.0.0.0 --port 8081
```

Required env:

```bash
API_AUTH_USERNAME=aiml
API_AUTH_PASSWORD=replace_with_secret
API_AUTH_SECRET=replace_with_long_random_secret
API_TOKEN_TTL_SECONDS=3600
```

Optional env:

```bash
API_HOST=0.0.0.0
API_PORT=8081
AI_CONFIG_TABLE=ai_configurations
API_MODEL_BASE_VERSION=v1.1.0
API_EXPORT_AFTER_INFERENCE=false
API_EXPORT_WRITE=false
API_SFTP_EXPORT_PATH=/path/to/sftp/drop
```

Systemd unit file:

- [deploy/systemd/recommendation.service](../deploy/systemd/recommendation.service)

Install and enable:

```bash
sudo cp deploy/systemd/recommendation.service /etc/systemd/system/recommendation.service
sudo systemctl daemon-reload
sudo systemctl enable --now recommendation.service
sudo systemctl status recommendation.service
```

Notes:

- `POST /api/v1/training` and `POST /api/v1/inference` require `transactionId` to already exist in `digital_collections.ai_configurations`.
- API updates that existing row to `ACCEPTED`, then background completion updates `status`, `message`, `model_version`, `accuracy`, `drift`, `processing_time_ms`, `modified_by`, `modified_on`.
- Duplicate `transactionId` returns `409 Conflict`.
- Inference drift and accuracy persist in `digital_collections.ai_configurations`.`drift` and `accuracy`.
- If `API_EXPORT_AFTER_INFERENCE=true`, inference completion creates 2 XLSX files in configured SFTP path:
  - `MD_UB_DATASET_DDMMYYYY_01.xlsx`
  - `MD_UB_SCHEDULER_DDMMYYYY_01.xlsx`
