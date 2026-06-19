from __future__ import annotations

import io
import json
import logging
import os
import socket
import sys
import threading
import urllib.request
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
    AuthManager,
    BackgroundJobRunner,
    extract_metrics_snapshot,
    model_status_from_accuracy,
    read_prediction_summary,
)
import campaign_recommendation.api_service as api_service_module  # noqa: E402

api_service_module.AI_CONFIG_UPDATE_INITIAL_DELAY_SECONDS = 0.0
api_service_module.AI_CONFIG_UPDATE_WAIT_TIMEOUT_SECONDS = 0.0
api_service_module.AI_CONFIG_UPDATE_WAIT_INTERVAL_SECONDS = 0.0


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

    def fetch_latest(self, *, entry_type: str, status: str | None = None):
        candidates = [entry for entry in self.entries.values() if entry.get("type") == entry_type]
        if status is not None:
            candidates = [entry for entry in candidates if entry.get("status") == status]
        if not candidates:
            return None
        candidates.sort(key=lambda entry: (entry.get("modified_on") or entry.get("created_on") or "", entry.get("transaction_id") or ""))
        return candidates[-1]


class DummyApiAuditRepo:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.updated: list[dict] = []

    def create_entry(self, **kwargs) -> None:
        self.created.append(kwargs)

    def update_latest_entry(self, **kwargs) -> None:
        self.updated.append(kwargs)


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
        request_queue_size=128,
        db_host="",
        db_port=5432,
        db_name="",
        db_user="",
        db_password="",
        target_schema="digital_collections",
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
        api_model_base_version="v1.1.0",
        api_audit_table="api_audit_log",
    )


def build_service(*, export_after_inference: bool = False, export_write: bool = False) -> AIMLApiService:
    config = build_config(export_after_inference=export_after_inference, export_write=export_write)
    ai_config_repo = DummyAIConfigRepo()
    api_audit_repo = DummyApiAuditRepo()
    ai_config_repo.entries["TRN1"] = {"transaction_id": "TRN1"}
    ai_config_repo.entries["TRNLOOKUP"] = {"transaction_id": "TRNLOOKUP"}
    ai_config_repo.entries["TRN_FIX"] = {"transaction_id": "TRN_FIX"}
    ai_config_repo.entries["TRN3"] = {"transaction_id": "TRN3"}
    ai_config_repo.entries["TRN_SKIP_AUDIT"] = {"transaction_id": "TRN_SKIP_AUDIT"}
    return AIMLApiService(
        config=config,
        auth_manager=AuthManager("aiml", "secret", "top-secret", 60),
        job_runner=DummyJobRunner(),
        logger=logging.getLogger("test_aiml_api"),
        ai_config_repo=ai_config_repo,
        api_audit_repo=api_audit_repo,
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
        api_model_base_version="v1.1.0",
        api_audit_table="api_audit_log",
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


def test_summarize_subprocess_failure_adds_plain_language_reason() -> None:
    exc = api_service_module.subprocess.CalledProcessError(
        1,
        ["run_inference_pipeline"],
        output="",
        stderr="OperationalError: connection failed",
    )

    message = api_service_module.summarize_subprocess_failure(exc, step_name="run_inference_pipeline")

    assert message == (
        "The system could not connect to the required database or source system. "
        "Technical detail: run_inference_pipeline failed: OperationalError: connection failed"
    )


def test_subprocess_error_message_ignores_traceback_pointer_lines() -> None:
    exc = api_service_module.subprocess.CalledProcessError(
        1,
        ["run_inference_pipeline"],
        output="",
        stderr=(
            "Traceback (most recent call last):\n"
            "  File \"x.py\", line 1, in <module>\n"
            "    run_python_script(\n"
            "    fetch_communication_extract(args, output_file, fetch_month, logger)\n"
            "    result = subprocess.run(cmd, check=True, cwd=SCRIPTS_DIR.parent, env=env, capture_output=True, text=True)\n"
            "OperationalError: connection failed\n"
            "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
        ),
    )

    assert api_service_module.subprocess_error_message(exc) == "OperationalError: connection failed"


def test_auth_and_status_route() -> None:
    service = build_service()
    service.ai_config_repo.entries["TRN_STATUS_1"] = {
        "transaction_id": "TRN_STATUS_1",
        "type": "TRAINING",
        "status": "COMPLETED",
        "model_version": "v1.1.1",
        "modified_on": "2026-06-09T10:00:00",
    }
    service.ai_config_repo.entries["TRN_STATUS_2"] = {
        "transaction_id": "TRN_STATUS_2",
        "type": "INFERENCE",
        "status": "COMPLETED",
        "model_version": "v1.1.1",
        "accuracy": "78.5",
        "drift": "4.8",
        "modified_on": "2026-06-09T11:00:00",
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


def test_unknown_api_path_returns_not_found_before_auth() -> None:
    service = build_service()
    app = AIMLApiApp(service)

    status, payload = invoke(
        app,
        method="POST",
        path="/api/v1/authh",
        body={"username": "aiml", "password": "aiml"},
    )

    assert status.startswith("404")
    assert payload == {"message": "Not found."}


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
    assert payload["modelVersion"] == "v1.1.0"
    assert service.ai_config_repo.updated == []
    assert service.api_audit_repo.created == []
    assert service.job_runner.submitted == ["TRN1"]


def test_transaction_lookup_route() -> None:
    service = build_service()
    service.ai_config_repo.entries["TRNLOOKUP"] = {
        "transaction_id": "TRNLOOKUP",
        "type": "TRAINING",
        "model_version": "v1.1.1",
        "message": "completed",
        "status": "COMPLETED",
        "created_on": "2026-06-09T10:00:00",
        "modified_on": "2026-06-09T10:10:00",
        "training_window": "3",
    }
    app = AIMLApiApp(service)
    token = service.auth_manager.issue_token("aiml")["access_token"]

    status, payload = invoke(app, method="GET", path="/api/v1/transactions/TRNLOOKUP", token=token)

    assert status.startswith("200")
    assert payload["transactionId"] == "TRNLOOKUP"
    assert payload["status"] == "COMPLETED"
    assert payload["modelVersion"] == "v1.1.1"


def test_training_accepts_request_before_ai_configuration_row_exists() -> None:
    config = build_config()
    runner = DummyJobRunner()
    service = AIMLApiService(
        config=config,
        auth_manager=AuthManager("aiml", "secret", "top-secret", 60),
        job_runner=runner,
        logger=logging.getLogger("test_aiml_api"),
        ai_config_repo=DummyAIConfigRepo(),
        api_audit_repo=DummyApiAuditRepo(),
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

    assert status.startswith("202")
    assert payload["status"] == "ACCEPTED"
    assert runner.submitted == ["TRN_MISSING"]



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


def test_inference_accepts_request_before_ai_configuration_row_exists() -> None:
    config = build_config()
    runner = DummyJobRunner()
    service = AIMLApiService(
        config=config,
        auth_manager=AuthManager("aiml", "secret", "top-secret", 60),
        job_runner=runner,
        logger=logging.getLogger("test_aiml_api"),
        ai_config_repo=DummyAIConfigRepo(),
        api_audit_repo=DummyApiAuditRepo(),
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

    assert status.startswith("202")
    assert payload["status"] == "ACCEPTED"
    assert runner.submitted == ["TRN_MISSING_INF"]



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


def test_inference_command_uses_pipeline_without_audit_flag() -> None:
    config = build_config()
    runner = ImmediateJobRunner()
    service = CapturingService(
        config=config,
        auth_manager=AuthManager("aiml", "secret", "top-secret", 60),
        job_runner=runner,
        logger=logging.getLogger("test_aiml_api"),
        ai_config_repo=DummyAIConfigRepo(),
        api_audit_repo=DummyApiAuditRepo(),
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
    assert "--skip-audit-log" not in service.inference_commands[0]


def test_read_prediction_summary_missing_file_returns_empty_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_service_module, "LOG_DIR", tmp_path)

    assert read_prediction_summary("2026-07") == {}


def test_run_inference_job_completes_when_summary_reader_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    service = AIMLApiService(
        config=build_config(),
        auth_manager=AuthManager("aiml", "secret", "top-secret", 60),
        job_runner=DummyJobRunner(),
        logger=logging.getLogger("test_aiml_api"),
        ai_config_repo=DummyAIConfigRepo(),
        api_audit_repo=DummyApiAuditRepo(),
    )
    service.ai_config_repo.entries["TRN_FIX"] = {"transaction_id": "TRN_FIX"}
    service.ai_config_repo.entries["TRN_MODEL"] = {
        "transaction_id": "TRN_MODEL",
        "type": "TRAINING",
        "status": "COMPLETED",
        "model_version": "v1.1.3",
        "modified_on": "2026-06-09T11:00:00",
    }

    monkeypatch.setattr(
        api_service_module.subprocess,
        "run",
        lambda *args, **kwargs: api_service_module.subprocess.CompletedProcess(args[0], 0, stdout="", stderr=""),
    )
    monkeypatch.setattr(api_service_module, "read_metrics_snapshot", lambda model_name: {"modelVersion": model_name, "currentAccuracy": 78.5, "driftPercentage": 4.8})
    monkeypatch.setattr(api_service_module, "read_prediction_summary", lambda predict_month: {"drift": {"drift_percentage": 4.8, "overall_psi": 0.12, "max_feature_psi": 0.2, "status": "LOW"}})

    service._run_inference_job(
        transaction_id="TRN_FIX",
        payload={"sourceMonth": "2026-06", "predictMonth": "2026-07", "model": "catboost_3m"},
    )

    assert service.ai_config_repo.updated[-1]["drift"] == 4.8
    assert service.ai_config_repo.updated[-1]["training_window"] == "10 days"
    assert service.ai_config_repo.updated[-1]["model_version"] == "v1.1.3"
    assert json.loads(service.ai_config_repo.updated[-1]["message"])["status"] == "COMPLETED"
    assert service.api_audit_repo.updated[-1]["reference_number"] == "TRN_FIX"
    assert service.api_audit_repo.updated[-1]["request_url"] == "/api/v1/inference"
    assert service.api_audit_repo.updated[-1]["status"] == "COMPLETED"
    assert service.api_audit_repo.updated[-1]["success_count"] == 0
    assert "processing_time_ms" not in service.ai_config_repo.updated[-1]
    assert service.api_audit_repo.updated[-1]["processing_time_ms"] >= 0


def test_inference_export_command_enabled() -> None:
    config = build_config(export_after_inference=True, export_write=True)
    runner = ImmediateJobRunner()
    service = CapturingService(
        config=config,
        auth_manager=AuthManager("aiml", "secret", "top-secret", 60),
        job_runner=runner,
        logger=logging.getLogger("test_aiml_api"),
        ai_config_repo=DummyAIConfigRepo(),
        api_audit_repo=DummyApiAuditRepo(),
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


def test_http_server_end_to_end_routes() -> None:
    service = build_service()
    service.ai_config_repo.entries["TRNHTTP"] = {
        "transaction_id": "TRNHTTP",
        "type": "INFERENCE",
        "model_version": "v1.1.1",
        "message": "completed",
        "status": "COMPLETED",
        "accuracy": "78.5",
        "drift": "4.8",
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


def test_failed_training_keeps_current_model_version(monkeypatch: pytest.MonkeyPatch) -> None:
    service = AIMLApiService(
        config=build_config(),
        auth_manager=AuthManager("aiml", "secret", "top-secret", 60),
        job_runner=DummyJobRunner(),
        logger=logging.getLogger("test_aiml_api"),
        ai_config_repo=DummyAIConfigRepo(),
        api_audit_repo=DummyApiAuditRepo(),
    )
    service.ai_config_repo.entries["TRN_FAIL"] = {"transaction_id": "TRN_FAIL"}
    service.ai_config_repo.entries["TRN_MODEL"] = {
        "transaction_id": "TRN_MODEL",
        "type": "TRAINING",
        "status": "COMPLETED",
        "model_version": "v1.1.0",
        "modified_on": "2026-06-09T11:00:00",
    }

    def raise_prepare_failure(*args, **kwargs):
        raise api_service_module.subprocess.CalledProcessError(
            1,
            ["prepare_training_window"],
            stdout="",
            stderr="ValueError: No rows matched the D-5 to D+5 window.",
        )

    monkeypatch.setattr(api_service_module, "run_logged_subprocess", raise_prepare_failure)

    service._run_training_job(
        transaction_id="TRN_FAIL",
        payload={
            "transactionId": "TRN_FAIL",
            "months": 3,
            "model": "catboost_3m",
            "currentModelVersion": "v1.1.0",
            "modelVersion": "v1.1.1",
        },
    )

    assert service.ai_config_repo.updated[0]["model_version"] == "v1.1.0"
    assert service.ai_config_repo.updated[-1]["status"] == "FAILED"
    assert service.ai_config_repo.updated[-1]["model_version"] == "v1.1.0"
    assert json.loads(service.ai_config_repo.updated[-1]["message"])["status"] == "FAILED"
    assert service.api_audit_repo.updated[-1]["reference_number"] == "TRN_FAIL"
    assert service.api_audit_repo.updated[-1]["request_url"] == "/api/v1/training"
    assert service.api_audit_repo.updated[-1]["status"] == "FAILED"
    assert "processing_time_ms" not in service.ai_config_repo.updated[-1]
    assert service.api_audit_repo.updated[-1]["processing_time_ms"] >= 0


def test_training_rejects_months_when_data_config_duration_differs(monkeypatch: pytest.MonkeyPatch) -> None:
    service = build_service()
    app = AIMLApiApp(service)
    token = service.auth_manager.issue_token("aiml")["access_token"]
    monkeypatch.setattr(service, "_configured_training_month_duration", lambda: 6)

    status, payload = invoke(
        app,
        method="POST",
        path="/api/v1/training",
        token=token,
        body={"transactionId": "TRN1", "months": 3},
    )

    assert status.startswith("400")
    assert payload == {
        "transactionId": "TRN1",
        "status": "FAILED",
        "message": "months must match configured training duration 6.",
    }


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
