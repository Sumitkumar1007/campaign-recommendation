from __future__ import annotations

import io
import json
import logging
import os
import socket
import sys
import threading
import urllib.request
import uuid
from pathlib import Path

import pytest
from wsgiref.simple_server import make_server


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from campaign_recommendation.api_service import (  # noqa: E402
    AIMLApiApp,
    AIMLApiService,
    ApiConfig,
    AuditLogRepository,
    AuthManager,
    BackgroundJobRunner,
    extract_metrics_snapshot,
    model_status_from_accuracy,
    read_prediction_summary,
)
import campaign_recommendation.api_service as api_service_module  # noqa: E402


class DummyAuditRepo:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.updated: list[dict] = []
        self.latest: dict[tuple[str, bool], dict] = {}

    def create_entry(self, **kwargs) -> None:
        self.created.append(kwargs)

    def update_entry(self, **kwargs) -> None:
        self.updated.append(kwargs)

    def fetch_latest(self, *, audit_type: str, completed_only: bool = False):
        return self.latest.get((audit_type, completed_only))

    def fetch_by_reference_number(self, reference_number: str):
        for payload in self.latest.values():
            if payload and payload.get("reference_number") == reference_number:
                return payload
        for payload in self.created:
            if payload.get("reference_number") == reference_number:
                return {
                    "type": payload["audit_type"],
                    "request_url": payload["request_url"],
                    "reference_number": payload["reference_number"],
                    "message": payload["message"],
                    "status": payload["status"],
                    "request_body": payload["request_body"],
                    "response_body": payload["response_body"],
                    "created_on": None,
                    "modified_on": None,
                }
        return None



class DummyJobRunner(BackgroundJobRunner):
    def __init__(self) -> None:
        super().__init__()
        self.submitted: list[str] = []

    def submit(self, job_name: str, func) -> None:  # type: ignore[override]
        self.submitted.append(job_name)


class DummyAIConfigRepo:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.updated: list[dict] = []
        self.entries: dict[str, dict] = {}

    def create_entry(self, **kwargs) -> None:
        self.created.append(kwargs)
        self.entries[kwargs["transaction_id"]] = kwargs

    def update_entry(self, **kwargs) -> None:
        self.updated.append(kwargs)
        existing = self.entries.setdefault(kwargs["transaction_id"], {"transaction_id": kwargs["transaction_id"]})
        existing.update(kwargs)

    def fetch_by_transaction_id(self, transaction_id: str):
        return self.entries.get(transaction_id)


class CapturingService(AIMLApiService):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inference_commands: list[list[str]] = []
        self.export_commands: list[list[str]] = []

    def _run_inference_job(self, *, transaction_id: str, payload: dict[str, object]) -> None:
        self.inference_commands.append(
            self._build_inference_command(
                source_month=str(payload["sourceMonth"]),
                predict_month=str(payload["predictMonth"]),
                model_name=str(payload["model"]),
            )
        )
        if self.config.export_after_inference:
            self.export_commands.append(
                self._build_export_command(
                    source_month=str(payload["sourceMonth"]),
                    predict_month=str(payload["predictMonth"]),
                    model_name=str(payload["model"]),
                )
            )


class ImmediateJobRunner(DummyJobRunner):
    def submit(self, job_name: str, func) -> None:  # type: ignore[override]
        self.submitted.append(job_name)
        func()


def build_config(*, export_after_inference: bool = False, export_write: bool = False) -> ApiConfig:
    return ApiConfig(
        host="127.0.0.1",
        port=8080,
        db_host="",
        db_port=5432,
        db_name="",
        db_user="",
        db_password="",
        target_schema="digital_collections",
        audit_table="api_audit_log",
        model_name="catboost_3m",
        auth_username="aiml",
        auth_password="secret",
        auth_secret="top-secret",
        token_ttl_seconds=60,
        feature_month_source="emi_date",
        campaign_vertical="LAP",
        campaign_vendor="prutech-cpass",
        export_after_inference=export_after_inference,
        export_trigger_state="PAUSED",
        export_write=export_write,
        log_file=REPO_ROOT / "artifacts" / "logs" / "test_aiml_api.log",
    )


def build_service(*, export_after_inference: bool = False, export_write: bool = False) -> AIMLApiService:
    config = build_config(export_after_inference=export_after_inference, export_write=export_write)
    ai_config_repo = DummyAIConfigRepo()
    ai_config_repo.entries["TRN1"] = {"transaction_id": "TRN1"}
    ai_config_repo.entries["TRNLOOKUP"] = {"transaction_id": "TRNLOOKUP"}
    ai_config_repo.entries["TRN_FIX"] = {"transaction_id": "TRN_FIX"}
    ai_config_repo.entries["TRN3"] = {"transaction_id": "TRN3"}
    ai_config_repo.entries["TRN_SKIP_AUDIT"] = {"transaction_id": "TRN_SKIP_AUDIT"}
    return AIMLApiService(
        config=config,
        auth_manager=AuthManager("aiml", "secret", "top-secret", 60),
        audit_repo=DummyAuditRepo(),
        job_runner=DummyJobRunner(),
        logger=logging.getLogger("test_aiml_api"),
        ai_config_repo=ai_config_repo,
    )


def invoke(app: AIMLApiApp, *, method: str, path: str, body: dict | None = None, token: str | None = None) -> tuple[str, dict]:
    payload = json.dumps(body or {}).encode("utf-8")
    captured: dict[str, object] = {}

    def start_response(status: str, headers) -> None:
        captured["status"] = status
        captured["headers"] = headers

    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "CONTENT_LENGTH": str(len(payload)),
        "wsgi.input": io.BytesIO(payload),
    }
    if token:
        environ["HTTP_AUTHORIZATION"] = f"Bearer {token}"
    response = b"".join(app(environ, start_response))
    return str(captured["status"]), json.loads(response.decode("utf-8"))


def invoke_http(base_url: str, *, method: str, path: str, body: dict | None = None, token: str | None = None) -> tuple[int, dict]:
    payload = json.dumps(body or {}).encode("utf-8")
    request = urllib.request.Request(
        url=f"{base_url}{path}",
        data=payload if method != "GET" or body else None,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def live_db_config() -> ApiConfig | None:
    required = [os.getenv("PGHOST"), os.getenv("PGDATABASE"), os.getenv("PGUSER"), os.getenv("PGPASSWORD")]
    if not all(required):
        return None
    return ApiConfig(
        host="127.0.0.1",
        port=8080,
        db_host=str(os.getenv("PGHOST")),
        db_port=int(os.getenv("PGPORT", "5432")),
        db_name=str(os.getenv("PGDATABASE")),
        db_user=str(os.getenv("PGUSER")),
        db_password=str(os.getenv("PGPASSWORD")),
        target_schema=os.getenv("TARGET_SCHEMA", "digital_collections"),
        audit_table=os.getenv("AUDIT_TABLE", "api_audit_log"),
        model_name="catboost_3m",
        auth_username="aiml",
        auth_password="secret",
        auth_secret="top-secret",
        token_ttl_seconds=60,
        feature_month_source="emi_date",
        campaign_vertical="LAP",
        campaign_vendor="prutech-cpass",
        export_after_inference=False,
        export_trigger_state="PAUSED",
        export_write=False,
        log_file=REPO_ROOT / "artifacts" / "logs" / "test_aiml_api_live.log",
    )


def test_model_status_thresholds() -> None:
    assert model_status_from_accuracy(70.0) == "Healthy"
    assert model_status_from_accuracy(69.99) == "Not Healthy"
    assert model_status_from_accuracy(None) == "Unknown"


def test_extract_metrics_snapshot_uses_validation_accuracy_and_gap() -> None:
    snapshot = extract_metrics_snapshot(
        {
            "train_metrics": {"average_day_accuracy": 0.81},
            "validation_metrics": {"average_day_accuracy": 0.74},
        },
        model_name="catboost_3m",
    )
    assert snapshot["currentAccuracy"] == 74.0
    assert snapshot["driftPercentage"] == 7.0


def test_auth_and_status_route() -> None:
    service = build_service()
    repo = service.audit_repo
    repo.latest[("TRAINING", True)] = {"status": "COMPLETED", "modified_on": "2026-06-09T10:00:00"}
    repo.latest[("INFERENCE", True)] = {
        "status": "COMPLETED",
        "modified_on": "2026-06-09T11:00:00",
        "driftPercentage": 4.8,
        "response_body": {"currentAccuracy": 78.5, "driftPercentage": 4.8, "modelVersion": "catboost_3m"},
    }
    app = AIMLApiApp(service)

    status, auth_payload = invoke(app, method="POST", path="/api/v1/auth", body={"username": "aiml", "password": "secret"})
    assert status.startswith("200")
    token = auth_payload["access_token"]

    status, payload = invoke(app, method="GET", path="/api/status", token=token)
    assert status.startswith("200")
    assert payload["modelStatus"] == "Healthy"
    assert payload["currentAccuracy"] == 78.5
    assert payload["currentDrift"] == 4.8


def test_training_route_creates_audit_and_submits_job() -> None:
    service = build_service()
    app = AIMLApiApp(service)
    token = service.auth_manager.issue_token("aiml")["access_token"]

    status, payload = invoke(
        app,
        method="POST",
        path="/api/v1/training",
        token=token,
        body={"transactionId": "TRN1", "months": 3},
    )

    assert status.startswith("202")
    assert payload["status"] == "ACCEPTED"
    assert payload["modelVersion"] == service.config.api_model_version
    assert service.audit_repo.created[0]["audit_type"] == "TRAINING"
    assert service.audit_repo.created[0]["request_body"]["model"] == service.config.model_name
    assert service.ai_config_repo.updated[0]["transaction_id"] == "TRN1"
    assert service.ai_config_repo.updated[0]["training_window"] == "3"
    assert service.job_runner.submitted == ["TRN1"]


def test_transaction_lookup_route() -> None:
    service = build_service()
    repo = service.audit_repo
    repo.latest[("TRAINING", True)] = {
        "type": "TRAINING",
        "request_url": "/api/v1/training",
        "reference_number": "TRNLOOKUP",
        "message": "completed",
        "status": "COMPLETED",
        "request_body": {"transactionId": "TRNLOOKUP", "months": 3},
        "response_body": {"transactionId": "TRNLOOKUP", "status": "COMPLETED"},
        "created_on": "2026-06-09T10:00:00",
        "modified_on": "2026-06-09T10:10:00",
    }
    app = AIMLApiApp(service)
    token = service.auth_manager.issue_token("aiml")["access_token"]

    status, payload = invoke(app, method="GET", path="/api/v1/transactions/TRNLOOKUP", token=token)

    assert status.startswith("200")
    assert payload["transactionId"] == "TRNLOOKUP"
    assert payload["status"] == "COMPLETED"


def test_training_requires_existing_ai_configuration_row() -> None:
    config = build_config()
    service = AIMLApiService(
        config=config,
        auth_manager=AuthManager("aiml", "secret", "top-secret", 60),
        audit_repo=DummyAuditRepo(),
        job_runner=DummyJobRunner(),
        logger=logging.getLogger("test_aiml_api"),
        ai_config_repo=DummyAIConfigRepo(),
    )
    app = AIMLApiApp(service)
    token = service.auth_manager.issue_token("aiml")["access_token"]

    status, payload = invoke(
        app,
        method="POST",
        path="/api/v1/training",
        token=token,
        body={"transactionId": "TRN_MISSING", "months": 3},
    )

    assert status.startswith("404")
    assert payload["message"] == "transactionId TRN_MISSING not found in ai_configurations."



def test_training_rejects_extra_fields() -> None:
    service = build_service()
    app = AIMLApiApp(service)
    token = service.auth_manager.issue_token("aiml")["access_token"]

    status, payload = invoke(
        app,
        method="POST",
        path="/api/v1/training",
        token=token,
        body={"transactionId": "TRN_EXTRA", "months": 3, "model": "catboost_3m"},
    )

    assert status.startswith("400")
    assert payload == {"transactionId": "TRN_EXTRA", "status": "FAILED", "message": "Unexpected fields: model"}


def test_inference_requires_existing_ai_configuration_row() -> None:
    config = build_config()
    service = AIMLApiService(
        config=config,
        auth_manager=AuthManager("aiml", "secret", "top-secret", 60),
        audit_repo=DummyAuditRepo(),
        job_runner=DummyJobRunner(),
        logger=logging.getLogger("test_aiml_api"),
        ai_config_repo=DummyAIConfigRepo(),
    )
    app = AIMLApiApp(service)
    token = service.auth_manager.issue_token("aiml")["access_token"]

    status, payload = invoke(
        app,
        method="POST",
        path="/api/v1/inference",
        token=token,
        body={"transactionId": "TRN_MISSING_INF"},
    )

    assert status.startswith("404")
    assert payload["message"] == "transactionId TRN_MISSING_INF not found in ai_configurations."



def test_inference_rejects_extra_fields() -> None:
    service = build_service()
    app = AIMLApiApp(service)
    token = service.auth_manager.issue_token("aiml")["access_token"]

    status, payload = invoke(
        app,
        method="POST",
        path="/api/v1/inference",
        token=token,
        body={"transactionId": "TRN_EXTRA", "sourceMonth": "2026-04"},
    )

    assert status.startswith("400")
    assert payload == {"transactionId": "TRN_EXTRA", "status": "FAILED", "message": "Unexpected fields: sourceMonth"}


def test_inference_rejects_non_spec_fields() -> None:
    service = build_service()
    app = AIMLApiApp(service)
    token = service.auth_manager.issue_token("aiml")["access_token"]

    status, payload = invoke(
        app,
        method="POST",
        path="/api/v1/inference",
        token=token,
        body={"transactionId": "TRN2", "sourceMonth": "2026-04", "predictMonth": "2026-06"},
    )

    assert status.startswith("400")
    assert payload == {"transactionId": "TRN2", "status": "FAILED", "message": "Unexpected fields: predictMonth, sourceMonth"}


def test_inference_command_skips_pipeline_audit_logging() -> None:
    config = build_config()
    repo = DummyAuditRepo()
    runner = ImmediateJobRunner()
    service = CapturingService(
        config=config,
        auth_manager=AuthManager("aiml", "secret", "top-secret", 60),
        audit_repo=repo,
        job_runner=runner,
        logger=logging.getLogger("test_aiml_api"),
        ai_config_repo=DummyAIConfigRepo(),
    )
    service.ai_config_repo.entries["TRN_SKIP_AUDIT"] = {"transaction_id": "TRN_SKIP_AUDIT"}
    app = AIMLApiApp(service)
    token = service.auth_manager.issue_token("aiml")["access_token"]

    status, payload = invoke(
        app,
        method="POST",
        path="/api/v1/inference",
        token=token,
        body={"transactionId": "TRN_SKIP_AUDIT"},
    )

    assert status.startswith("202")
    assert payload["status"] == "ACCEPTED"
    assert service.inference_commands
    assert "--skip-audit-log" in service.inference_commands[0]


def test_read_prediction_summary_missing_file_returns_empty_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_service_module, "LOG_DIR", tmp_path)

    assert read_prediction_summary("2026-07") == {}


def test_run_inference_job_completes_when_summary_reader_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = DummyAuditRepo()
    service = AIMLApiService(
        config=build_config(),
        auth_manager=AuthManager("aiml", "secret", "top-secret", 60),
        audit_repo=repo,
        job_runner=DummyJobRunner(),
        logger=logging.getLogger("test_aiml_api"),
        ai_config_repo=DummyAIConfigRepo(),
    )
    service.ai_config_repo.entries["TRN_FIX"] = {"transaction_id": "TRN_FIX"}

    monkeypatch.setattr(api_service_module.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(api_service_module, "read_metrics_snapshot", lambda model_name: {"modelVersion": model_name, "currentAccuracy": 78.5, "driftPercentage": 4.8})
    monkeypatch.setattr(api_service_module, "read_prediction_summary", lambda predict_month: {"drift": {"drift_percentage": 4.8, "overall_psi": 0.12, "max_feature_psi": 0.2, "status": "LOW"}})

    service._run_inference_job(
        transaction_id="TRN_FIX",
        payload={"sourceMonth": "2026-06", "predictMonth": "2026-07", "model": "catboost_3m"},
    )

    assert repo.updated
    assert repo.updated[0]["status"] == "COMPLETED"
    assert repo.updated[0]["response_body"]["driftPercentage"] == 4.8
    assert service.ai_config_repo.updated[0]["drift"] == 4.8
    assert service.ai_config_repo.updated[0]["training_window"] == "10 days"


def test_inference_export_command_enabled() -> None:
    config = build_config(export_after_inference=True, export_write=True)
    repo = DummyAuditRepo()
    runner = ImmediateJobRunner()
    service = CapturingService(
        config=config,
        auth_manager=AuthManager("aiml", "secret", "top-secret", 60),
        audit_repo=repo,
        job_runner=runner,
        logger=logging.getLogger("test_aiml_api"),
        ai_config_repo=DummyAIConfigRepo(),
    )
    service.ai_config_repo.entries["TRN3"] = {"transaction_id": "TRN3"}
    app = AIMLApiApp(service)
    token = service.auth_manager.issue_token("aiml")["access_token"]

    status, payload = invoke(
        app,
        method="POST",
        path="/api/v1/inference",
        token=token,
        body={"transactionId": "TRN3"},
    )

    assert status.startswith("202")
    assert payload["status"] == "ACCEPTED"
    assert service.export_commands
    assert "export_recommendation_workbooks.py" in service.export_commands[0][1]
    assert "--write" in service.export_commands[0]


@pytest.mark.skipif(live_db_config() is None, reason="Live Postgres env not configured")
def test_audit_log_repository_round_trip_live_db() -> None:
    config = live_db_config()
    assert config is not None
    repo = AuditLogRepository(config)
    reference_number = f"TEST-{uuid.uuid4()}"

    repo.create_entry(
        audit_type="TRAINING",
        request_url="/api/v1/training",
        reference_number=reference_number,
        request_body={"transactionId": reference_number, "months": 3},
        response_body={"transactionId": reference_number, "status": "ACCEPTED"},
        status="ACCEPTED",
        message="accepted",
    )
    repo.update_entry(
        reference_number=reference_number,
        audit_type="TRAINING",
        status="COMPLETED",
        message="completed",
        response_body={"transactionId": reference_number, "status": "COMPLETED"},
        processing_time_ms=123,
    )

    latest = repo.fetch_latest(audit_type="TRAINING", completed_only=True)
    assert latest is not None
    assert latest["status"] == "COMPLETED"
    assert latest["response_body"]["transactionId"] == reference_number


def test_http_server_end_to_end_routes() -> None:
    service = build_service()
    repo = service.audit_repo
    repo.latest[("TRAINING", True)] = {"status": "COMPLETED", "modified_on": "2026-06-09T10:00:00"}
    repo.latest[("INFERENCE", True)] = {
        "type": "INFERENCE",
        "request_url": "/api/v1/inference",
        "reference_number": "TRNHTTP",
        "message": "completed",
        "status": "COMPLETED",
        "request_body": {"transactionId": "TRNHTTP"},
        "response_body": {"transactionId": "TRNHTTP", "status": "COMPLETED", "currentAccuracy": 78.5, "driftPercentage": 4.8, "modelVersion": "catboost_3m"},
        "created_on": "2026-06-09T10:00:00",
        "modified_on": "2026-06-09T10:10:00",
    }
    app = AIMLApiApp(service)
    port = free_port()
    server = make_server("127.0.0.1", port, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    try:
        status, auth_payload = invoke_http(base_url, method="POST", path="/api/v1/auth", body={"username": "aiml", "password": "secret"})
        assert status == 200
        token = auth_payload["access_token"]

        status, health_payload = invoke_http(base_url, method="GET", path="/api/health")
        assert status == 200
        assert health_payload["service"] == "aiml-integration-api"

        status, lookup_payload = invoke_http(base_url, method="GET", path="/api/v1/transactions/TRNHTTP", token=token)
        assert status == 200
        assert lookup_payload["transactionId"] == "TRNHTTP"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_training_missing_months_returns_failed_payload() -> None:
    service = build_service()
    app = AIMLApiApp(service)
    token = service.auth_manager.issue_token("aiml")["access_token"]

    status, payload = invoke(
        app,
        method="POST",
        path="/api/v1/training",
        token=token,
        body={"transactionId": "TRN1"},
    )

    assert status.startswith("400")
    assert payload == {"transactionId": "TRN1", "status": "FAILED", "message": "months is required."}
