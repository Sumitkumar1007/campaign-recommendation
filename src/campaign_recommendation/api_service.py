from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from pathlib import Path
from typing import Any, Callable

from wsgiref.simple_server import make_server

import psycopg
from psycopg import sql


REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = REPO_ROOT / "venv" / "bin" / "python"
SCRIPTS_DIR = REPO_ROOT / "scripts"
LOG_DIR = REPO_ROOT / "artifacts" / "logs"
MODEL_DIR = REPO_ROOT / "artifacts" / "models"
METRICS_DIR = REPO_ROOT / "artifacts" / "metrics"
PREDICTIONS_DIR = REPO_ROOT / "artifacts" / "predictions"
CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "checkpoints"
DEFAULT_SFTP_EXPORT_PATH = REPO_ROOT / "artifacts" / "exports" / "sftp"


def load_dotenv(path: Path | None = None) -> None:
    env_path = path or (REPO_ROOT / ".env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class ApiConfig:
    host: str
    port: int
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    target_schema: str
    model_name: str
    auth_username: str
    auth_password: str
    auth_secret: str
    token_ttl_seconds: int
    feature_month_source: str
    campaign_vertical: str
    campaign_vendor: str
    export_after_inference: bool
    export_trigger_state: str
    export_write: bool
    log_file: Path
    ai_config_table: str = "ai_configurations"
    api_model_base_version: str = "v1.1.0"
    sftp_export_path: Path = DEFAULT_SFTP_EXPORT_PATH

    @classmethod
    def from_env(cls) -> "ApiConfig":
        load_dotenv()
        return cls(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8080")),
            db_host=os.getenv("PGHOST", ""),
            db_port=int(os.getenv("PGPORT", "5432")),
            db_name=os.getenv("PGDATABASE", ""),
            db_user=os.getenv("PGUSER", ""),
            db_password=os.getenv("PGPASSWORD", ""),
            target_schema=os.getenv("TARGET_SCHEMA", "digital_collections"),
            model_name=os.getenv("MODEL_NAME", "catboost_3m"),
            auth_username=os.getenv("API_AUTH_USERNAME", "aiml"),
            auth_password=os.getenv("API_AUTH_PASSWORD", "aiml"),
            auth_secret=os.getenv("API_AUTH_SECRET", "change-me"),
            token_ttl_seconds=int(os.getenv("API_TOKEN_TTL_SECONDS", "3600")),
            feature_month_source=os.getenv("FEATURE_MONTH_SOURCE", "emi_date"),
            campaign_vertical=os.getenv("CAMPAIGN_VERTICAL", "LAP"),
            campaign_vendor=os.getenv("CAMPAIGN_VENDOR", "prutech-cpass"),
            export_after_inference=os.getenv("API_EXPORT_AFTER_INFERENCE", "false").strip().lower() == "true",
            export_trigger_state=os.getenv("API_EXPORT_TRIGGER_STATE", "PAUSED").strip().upper(),
            export_write=os.getenv("API_EXPORT_WRITE", "false").strip().lower() == "true",
            log_file=Path(os.getenv("API_LOG_FILE", str(LOG_DIR / "aiml_api.log"))),
            ai_config_table=os.getenv("AI_CONFIG_TABLE", "ai_configurations"),
            api_model_base_version=os.getenv("API_MODEL_BASE_VERSION", "v1.1.0"),
            sftp_export_path=Path(os.getenv("API_SFTP_EXPORT_PATH", os.getenv("SFTP_EXPORT_PATH", str(DEFAULT_SFTP_EXPORT_PATH)))),
        )


def utcnow_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def qualified_identifier(schema: str, table: str) -> sql.Composed:
    return sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(table))


def parse_month(value: str) -> tuple[int, int]:
    if len(value) != 7 or value[4] != "-":
        raise ValueError(f"Invalid month {value!r}. Expected YYYY-MM.")
    year = int(value[:4])
    month = int(value[5:])
    if month < 1 or month > 12:
        raise ValueError(f"Invalid month {value!r}. Expected YYYY-MM.")
    return year, month


def current_month_yyyy_mm(now: datetime | None = None) -> str:
    current = now or datetime.now(UTC)
    return current.strftime("%Y-%m")


def add_months(yyyy_mm: str, delta: int) -> str:
    year, month = parse_month(yyyy_mm)
    total = year * 12 + (month - 1) + delta
    next_year = total // 12
    next_month = total % 12 + 1
    return f"{next_year:04d}-{next_month:02d}"


def month_label(yyyy_mm: str) -> str:
    year, month = parse_month(yyyy_mm)
    return datetime(year, month, 1).strftime("%b-%Y").upper()


def validate_next_month_pair(source_month: str, predict_month: str) -> None:
    if add_months(source_month, 1) != predict_month:
        raise ValueError(
            f"predictMonth must be exactly one month after sourceMonth. Got sourceMonth={source_month}, "
            f"predictMonth={predict_month}."
        )

def validate_allowed_fields(payload: dict[str, Any], allowed_fields: set[str]) -> None:
    extra_fields = sorted(set(payload) - allowed_fields)
    if extra_fields:
        raise ValueError(f"Unexpected fields: {', '.join(extra_fields)}")




def failure_response(transaction_id: str | None, message: str) -> dict[str, Any]:
    payload = {
        "transactionId": transaction_id or "",
        "status": "FAILED",
        "message": message,
    }
    return payload

def model_status_from_accuracy(accuracy: float | None) -> str:
    if accuracy is None:
        return "Unknown"
    return "Healthy" if accuracy >= 70.0 else "Not Healthy"


def parse_model_version(value: str | None) -> tuple[int, int, int] | None:
    if not value:
        return None
    cleaned = value.strip()
    if cleaned.startswith(("v", "V")):
        cleaned = cleaned[1:]
    parts = cleaned.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        return None
    return int(parts[0]), int(parts[1]), int(parts[2])


def format_model_version(version: tuple[int, int, int]) -> str:
    return f"v{version[0]}.{version[1]}.{version[2]}"


def bump_model_version(value: str | None, *, fallback: str) -> str:
    parsed = parse_model_version(value) or parse_model_version(fallback)
    if parsed is None:
        return fallback
    return format_model_version((parsed[0], parsed[1], parsed[2] + 1))


def extract_metrics_snapshot(metrics: dict[str, Any], *, model_name: str) -> dict[str, Any]:
    validation = metrics.get("validation_metrics") or {}
    train = metrics.get("train_metrics") or {}
    accuracy_base = validation if validation.get("average_day_accuracy") is not None else train
    accuracy_fraction = accuracy_base.get("average_day_accuracy")
    accuracy = round(float(accuracy_fraction) * 100, 2) if accuracy_fraction is not None else None

    train_fraction = train.get("average_day_accuracy")
    validation_fraction = validation.get("average_day_accuracy")
    drift_percentage = None
    if train_fraction is not None and validation_fraction is not None:
        drift_percentage = round(abs(float(train_fraction) - float(validation_fraction)) * 100, 2)

    return {
        "modelVersion": model_name,
        "currentAccuracy": accuracy,
        "driftPercentage": drift_percentage,
        "metrics": metrics,
    }


def read_metrics_snapshot(model_name: str) -> dict[str, Any]:
    candidates = [
        METRICS_DIR / f"next_month_strategy_{model_name}_metrics.json",
        METRICS_DIR / "next_month_strategy_catboost_3m_metrics.json",
        METRICS_DIR / "next_month_strategy_logistic_metrics.json",
        METRICS_DIR / "next_month_strategy_catboost_metrics.json",
    ]
    for path in candidates:
        if path.exists():
            return extract_metrics_snapshot(json.loads(path.read_text(encoding="utf-8")), model_name=model_name)
    return {
        "modelVersion": model_name,
        "currentAccuracy": None,
        "driftPercentage": None,
        "metrics": {},
    }


def prediction_summary_path(predict_month: str) -> Path:
    return LOG_DIR / f"{predict_month.replace('-', '_').lower()}_prediction_summary.json"


def subprocess_error_message(exc: subprocess.CalledProcessError) -> str:
    candidates = []
    if isinstance(exc.stderr, str) and exc.stderr.strip():
        candidates.extend(line.strip() for line in exc.stderr.splitlines() if line.strip())
    if isinstance(exc.stdout, str) and exc.stdout.strip():
        candidates.extend(line.strip() for line in exc.stdout.splitlines() if line.strip())
    for line in reversed(candidates):
        if 'Traceback' in line:
            continue
        if line.startswith('File "'):
            continue
        return line
    return str(exc)


def summarize_subprocess_failure(exc: subprocess.CalledProcessError, *, step_name: str, limit: int = 500) -> str:
    detail = subprocess_error_message(exc)
    summary = f"{step_name} failed: {detail}"
    if len(summary) <= limit:
        return summary
    return summary[: limit - 3].rstrip() + "..."


def _log_subprocess_stream(
    logger: logging.Logger,
    *,
    transaction_id: str,
    step_name: str,
    stream_name: str,
    content: str | None,
) -> None:
    if not isinstance(content, str) or not content.strip():
        logger.info("%s %s empty | transaction_id=%s", step_name, stream_name, transaction_id)
        return
    for line in content.strip().splitlines():
        logger.info("%s %s | transaction_id=%s | %s", step_name, stream_name, transaction_id, line)


def run_logged_subprocess(
    command: list[str],
    *,
    logger: logging.Logger,
    transaction_id: str,
    step_name: str,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    logger.info(
        "Starting subprocess step | transaction_id=%s step=%s cwd=%s command=%s",
        transaction_id,
        step_name,
        cwd,
        command,
    )
    started_at = time.perf_counter()
    try:
        result = subprocess.run(command, check=True, cwd=cwd, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        logger.error(
            "Failed subprocess step | transaction_id=%s step=%s returncode=%s duration_ms=%s",
            transaction_id,
            step_name,
            exc.returncode,
            duration_ms,
        )
        _log_subprocess_stream(logger, transaction_id=transaction_id, step_name=step_name, stream_name="stdout", content=exc.stdout)
        _log_subprocess_stream(logger, transaction_id=transaction_id, step_name=step_name, stream_name="stderr", content=exc.stderr)
        raise
    duration_ms = int((time.perf_counter() - started_at) * 1000)
    logger.info(
        "Completed subprocess step | transaction_id=%s step=%s returncode=%s duration_ms=%s",
        transaction_id,
        step_name,
        result.returncode,
        duration_ms,
    )
    _log_subprocess_stream(logger, transaction_id=transaction_id, step_name=step_name, stream_name="stdout", content=result.stdout)
    _log_subprocess_stream(logger, transaction_id=transaction_id, step_name=step_name, stream_name="stderr", content=result.stderr)
    return result


def read_prediction_summary(predict_month: str) -> dict[str, Any]:
    path = prediction_summary_path(predict_month)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


class AuthManager:
    def __init__(self, username: str, password: str, secret: str, ttl_seconds: int):
        self.username = username
        self.password = password
        self.secret = secret.encode("utf-8")
        self.ttl_seconds = ttl_seconds

    def authenticate(self, username: str, password: str) -> bool:
        return hmac.compare_digest(username, self.username) and hmac.compare_digest(password, self.password)

    def issue_token(self, subject: str) -> dict[str, Any]:
        expires_at = int(time.time()) + self.ttl_seconds
        payload = {"sub": subject, "exp": expires_at}
        payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        signature = hmac.new(self.secret, payload_bytes, hashlib.sha256).digest()
        token = ".".join(
            [
                base64.urlsafe_b64encode(payload_bytes).decode("utf-8").rstrip("="),
                base64.urlsafe_b64encode(signature).decode("utf-8").rstrip("="),
            ]
        )
        return {
            "access_token": token,
            "expires_in": self.ttl_seconds,
        }

    def verify_token(self, token: str) -> dict[str, Any]:
        try:
            payload_part, signature_part = token.split(".", 1)
            payload_bytes = base64.urlsafe_b64decode(payload_part + "=" * (-len(payload_part) % 4))
            actual_signature = base64.urlsafe_b64decode(signature_part + "=" * (-len(signature_part) % 4))
        except Exception as exc:  # noqa: BLE001
            raise PermissionError("Malformed token.") from exc

        expected_signature = hmac.new(self.secret, payload_bytes, hashlib.sha256).digest()
        if not hmac.compare_digest(actual_signature, expected_signature):
            raise PermissionError("Invalid token signature.")

        payload = json.loads(payload_bytes.decode("utf-8"))
        if int(payload["exp"]) < int(time.time()):
            raise PermissionError("Token expired.")
        return payload



class AIConfigurationRepository:
    def __init__(self, config: ApiConfig):
        self.config = config

    def connect(self) -> psycopg.Connection:
        return psycopg.connect(
            host=self.config.db_host,
            port=self.config.db_port,
            dbname=self.config.db_name,
            user=self.config.db_user,
            password=self.config.db_password,
            autocommit=False,
        )

    def ensure_table(self, conn: psycopg.Connection) -> None:
        conn.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {table_ref} (
                    id int8 GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    "type" varchar(50) NULL,
                    transaction_id varchar(100) NOT NULL,
                    status varchar(20) NULL,
                    message text NULL,
                    model_version varchar(30) NULL,
                    accuracy varchar(20) NULL,
                    drift varchar(20) NULL,
                    processing_time_ms varchar(255) NULL,
                    created_by varchar(20) NULL,
                    created_on timestamp NULL,
                    modified_by varchar(20) NULL,
                    modified_on timestamp NULL,
                    training_window varchar(20) NULL
                )
                """
            ).format(table_ref=qualified_identifier(self.config.target_schema, self.config.ai_config_table))
        )

    def create_entry(self, *, entry_type: str, transaction_id: str, status: str, message: str, training_window: str | None = None, model_version: str | None = None) -> None:
        with self.connect() as conn:
            self.ensure_table(conn)
            now = utcnow_naive()
            conn.execute(
                sql.SQL(
                    """
                    INSERT INTO {table_ref} (
                        "type", transaction_id, status, message, model_version,
                        created_by, created_on, modified_by, modified_on, training_window
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                ).format(table_ref=qualified_identifier(self.config.target_schema, self.config.ai_config_table)),
                (entry_type, transaction_id, status, message, model_version, "DIGITAL", now, "DIGITAL", now, training_window),
            )
            conn.commit()

    def update_entry(self, *, transaction_id: str, status: str, message: str, model_version: str | None = None, accuracy: float | None = None, drift: float | None = None, processing_time_ms: int | None = None, entry_type: str | None = None, training_window: str | None = None) -> None:
        with self.connect() as conn:
            self.ensure_table(conn)
            conn.execute(
                sql.SQL(
                    """
                    UPDATE {table_ref}
                    SET "type" = COALESCE(%s, "type"),
                        status = %s,
                        message = %s,
                        model_version = %s,
                        accuracy = %s,
                        drift = %s,
                        processing_time_ms = %s,
                        training_window = COALESCE(%s, training_window),
                        modified_by = %s,
                        modified_on = %s
                    WHERE transaction_id = %s
                    """
                ).format(table_ref=qualified_identifier(self.config.target_schema, self.config.ai_config_table)),
                (
                    entry_type,
                    status,
                    message,
                    model_version,
                    str(accuracy) if accuracy is not None else None,
                    str(drift) if drift is not None else None,
                    str(processing_time_ms) if processing_time_ms is not None else None,
                    training_window,
                    "AIML",
                    utcnow_naive(),
                    transaction_id,
                ),
            )
            conn.commit()

    def fetch_by_transaction_id(self, transaction_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            self.ensure_table(conn)
            row = conn.execute(
                sql.SQL(
                    """
                    SELECT "type", transaction_id, status, message, model_version, accuracy, drift,
                           processing_time_ms, created_on, modified_on, training_window
                    FROM {table_ref}
                    WHERE transaction_id = %s
                    ORDER BY COALESCE(modified_on, created_on) DESC
                    LIMIT 1
                    """
                ).format(table_ref=qualified_identifier(self.config.target_schema, self.config.ai_config_table)),
                (transaction_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "type": row[0],
                "transaction_id": row[1],
                "status": row[2],
                "message": row[3],
                "model_version": row[4],
                "accuracy": row[5],
                "drift": row[6],
                "processing_time_ms": row[7],
                "created_on": row[8].isoformat() if row[8] else None,
                "modified_on": row[9].isoformat() if row[9] else None,
                "training_window": row[10],
            }

    def fetch_latest(self, *, entry_type: str, status: str | None = None) -> dict[str, Any] | None:
        with self.connect() as conn:
            self.ensure_table(conn)
            conditions = ['"type" = %s']
            params: list[Any] = [entry_type]
            if status is not None:
                conditions.append("status = %s")
                params.append(status)
            row = conn.execute(
                sql.SQL(
                    """
                    SELECT "type", transaction_id, status, message, model_version, accuracy, drift,
                           processing_time_ms, created_on, modified_on, training_window
                    FROM {table_ref}
                    WHERE {conditions}
                    ORDER BY COALESCE(modified_on, created_on) DESC, id DESC
                    LIMIT 1
                    """
                ).format(
                    table_ref=qualified_identifier(self.config.target_schema, self.config.ai_config_table),
                    conditions=sql.SQL(" AND ").join(sql.SQL(part) for part in conditions),
                ),
                params,
            ).fetchone()
            if row is None:
                return None
            return {
                "type": row[0],
                "transaction_id": row[1],
                "status": row[2],
                "message": row[3],
                "model_version": row[4],
                "accuracy": row[5],
                "drift": row[6],
                "processing_time_ms": row[7],
                "created_on": row[8].isoformat() if row[8] else None,
                "modified_on": row[9].isoformat() if row[9] else None,
                "training_window": row[10],
            }


class NoopAIConfigurationRepository:
    def create_entry(self, **kwargs) -> None:
        return None

    def update_entry(self, **kwargs) -> None:
        return None

    def fetch_by_transaction_id(self, transaction_id: str) -> dict[str, Any] | None:
        return None

    def fetch_latest(self, *, entry_type: str, status: str | None = None) -> dict[str, Any] | None:
        return None


class BackgroundJobRunner:
    def __init__(self) -> None:
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def submit(self, job_name: str, func: Callable[[], None]) -> None:
        thread = threading.Thread(target=self._run_and_cleanup, args=(job_name, func), daemon=True)
        with self._lock:
            self._threads[job_name] = thread
        thread.start()

    def _run_and_cleanup(self, job_name: str, func: Callable[[], None]) -> None:
        try:
            func()
        finally:
            with self._lock:
                self._threads.pop(job_name, None)

    def is_running(self, job_name: str) -> bool:
        with self._lock:
            thread = self._threads.get(job_name)
        return bool(thread and thread.is_alive())


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("aiml_api")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class AIMLApiService:
    def __init__(
        self,
        *,
        config: ApiConfig,
        auth_manager: AuthManager,
        job_runner: BackgroundJobRunner,
        logger: logging.Logger,
        ai_config_repo: AIConfigurationRepository | NoopAIConfigurationRepository | None = None,
    ) -> None:
        self.config = config
        self.auth_manager = auth_manager
        self.job_runner = job_runner
        self.logger = logger
        self.ai_config_repo = ai_config_repo or NoopAIConfigurationRepository()

    def health(self) -> tuple[int, dict[str, Any]]:
        db_status = "not-configured"
        db_error = None
        if self.config.db_host and self.config.db_name and self.config.db_user and self.config.db_password:
            try:
                with self.ai_config_repo.connect() as conn:
                    conn.execute("SELECT 1")
                db_status = "ok"
            except Exception as exc:  # noqa: BLE001
                db_status = "error"
                db_error = str(exc)

        return HTTPStatus.OK, {
            "status": "ok" if db_status != "error" else "degraded",
            "service": "aiml-integration-api",
            "dbStatus": db_status,
            "dbError": db_error,
            "modelVersion": self.current_model_version(),
        }

    def issue_auth_token(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        username = str(payload.get("username", ""))
        password = str(payload.get("password", ""))
        if not self.auth_manager.authenticate(username, password):
            return HTTPStatus.UNAUTHORIZED, {"message": "Authentication failed."}
        return HTTPStatus.OK, self.auth_manager.issue_token(username)

    def model_status(self) -> tuple[int, dict[str, Any]]:
        training = self.ai_config_repo.fetch_latest(entry_type="TRAINING", status="COMPLETED")
        inference = self.ai_config_repo.fetch_latest(entry_type="INFERENCE", status="COMPLETED")
        metrics = read_metrics_snapshot(self.config.model_name)
        accuracy = self._coerce_float((inference or {}).get("accuracy"))
        if accuracy is None:
            accuracy = metrics.get("currentAccuracy")
        drift = self._coerce_float((inference or {}).get("drift"))
        if drift is None:
            drift = metrics.get("driftPercentage")
        payload = {
            "modelVersion": self.current_model_version(),
            "lastTrainingDate": (training or {}).get("modified_on") or (training or {}).get("created_on"),
            "lastInferenceDate": (inference or {}).get("modified_on") or (inference or {}).get("created_on"),
            "currentAccuracy": accuracy,
            "currentDrift": drift,
            "modelStatus": model_status_from_accuracy(accuracy),
            "latestTrainingStatus": (training or {}).get("status"),
            "latestInferenceStatus": (inference or {}).get("status"),
        }
        return HTTPStatus.OK, payload

    def get_transaction_status(self, reference_number: str) -> tuple[int, dict[str, Any]]:
        config_entry = self.ai_config_repo.fetch_by_transaction_id(reference_number)
        if config_entry is None:
            return HTTPStatus.NOT_FOUND, {"message": f"Transaction {reference_number} not found."}
        payload = {
            "transactionId": config_entry.get("transaction_id"),
            "type": config_entry.get("type"),
            "status": config_entry.get("status"),
            "message": config_entry.get("message"),
            "modelVersion": config_entry.get("model_version"),
            "accuracy": config_entry.get("accuracy"),
            "drift": config_entry.get("drift"),
            "createdOn": config_entry.get("created_on"),
            "modifiedOn": config_entry.get("modified_on"),
            "trainingWindow": config_entry.get("training_window"),
        }
        return HTTPStatus.OK, payload

    def current_model_version(self) -> str:
        latest_training = self.ai_config_repo.fetch_latest(entry_type="TRAINING", status="COMPLETED")
        model_version = (latest_training or {}).get("model_version")
        parsed = parse_model_version(model_version)
        if parsed is None:
            return self.config.api_model_base_version
        return format_model_version(parsed)

    def next_model_version(self) -> str:
        return bump_model_version(self.current_model_version(), fallback=self.config.api_model_base_version)

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def trigger_training(self, payload: dict[str, Any], request_url: str) -> tuple[int, dict[str, Any]]:
        transaction_id = str(payload.get("transactionId", "")).strip()
        try:
            validate_allowed_fields(payload, {"transactionId", "months"})
        except ValueError as exc:
            return HTTPStatus.BAD_REQUEST, failure_response(transaction_id, str(exc))
        if not transaction_id:
            return HTTPStatus.BAD_REQUEST, failure_response(transaction_id, "transactionId is required.")
        config_entry = self.ai_config_repo.fetch_by_transaction_id(transaction_id)
        if config_entry is None:
            return HTTPStatus.NOT_FOUND, failure_response(transaction_id, f"transactionId {transaction_id} not found in ai_configurations.")

        if self.job_runner.is_running(transaction_id):
            return HTTPStatus.CONFLICT, failure_response(transaction_id, f"Job already running for {transaction_id}.")

        if "months" not in payload:
            return HTTPStatus.BAD_REQUEST, failure_response(transaction_id, "months is required.")
        months = int(payload.get("months", 3))
        model_name = self.config.model_name
        next_model_version = self.next_model_version()
        accepted = {
            "transactionId": transaction_id,
            "status": "ACCEPTED",
            "modelVersion": next_model_version,
            "message": "Request accepted for processing.",
        }
        request_body = {
            "transactionId": transaction_id,
            "months": months,
            "model": model_name,
            "modelVersion": next_model_version,
        }
        self.ai_config_repo.update_entry(
            transaction_id=transaction_id,
            entry_type="TRAINING",
            status="ACCEPTED",
            message="Request accepted for processing.",
            training_window=str(months),
            model_version=next_model_version,
        )
        self.job_runner.submit(
            transaction_id,
            lambda: self._run_training_job(transaction_id=transaction_id, payload=request_body),
        )
        return HTTPStatus.ACCEPTED, accepted

    def trigger_inference(self, payload: dict[str, Any], request_url: str) -> tuple[int, dict[str, Any]]:
        transaction_id = str(payload.get("transactionId", "")).strip()
        try:
            validate_allowed_fields(payload, {"transactionId"})
        except ValueError as exc:
            return HTTPStatus.BAD_REQUEST, failure_response(transaction_id, str(exc))
        if not transaction_id:
            return HTTPStatus.BAD_REQUEST, failure_response(transaction_id, "transactionId is required.")
        config_entry = self.ai_config_repo.fetch_by_transaction_id(transaction_id)
        if config_entry is None:
            return HTTPStatus.NOT_FOUND, failure_response(transaction_id, f"transactionId {transaction_id} not found in ai_configurations.")

        if self.job_runner.is_running(transaction_id):
            return HTTPStatus.CONFLICT, failure_response(transaction_id, f"Job already running for {transaction_id}.")

        source_month = os.getenv("SOURCE_MONTH", current_month_yyyy_mm())
        predict_month = os.getenv("PREDICT_MONTH", add_months(source_month, 1))
        try:
            parse_month(source_month)
            parse_month(predict_month)
            validate_next_month_pair(source_month, predict_month)
        except ValueError as exc:
            return HTTPStatus.BAD_REQUEST, failure_response(transaction_id, str(exc))
        model_name = self.config.model_name
        metrics = read_metrics_snapshot(model_name)
        current_model_version = self.current_model_version()
        accepted = {
            "transactionId": transaction_id,
            "status": "ACCEPTED",
            "driftPercentage": metrics.get("driftPercentage"),
            "currentAccuracy": metrics.get("currentAccuracy"),
            "modelVersion": current_model_version,
            "message": "Request accepted for processing.",
        }
        request_body = {
            "transactionId": transaction_id,
            "sourceMonth": source_month,
            "predictMonth": predict_month,
            "model": model_name,
        }
        self.ai_config_repo.update_entry(
            transaction_id=transaction_id,
            entry_type="INFERENCE",
            status="ACCEPTED",
            message="Request accepted for processing.",
            model_version=current_model_version,
            training_window="10 days",
        )
        self.job_runner.submit(
            transaction_id,
            lambda: self._run_inference_job(transaction_id=transaction_id, payload=request_body),
        )
        return HTTPStatus.ACCEPTED, accepted

    def _run_training_job(self, *, transaction_id: str, payload: dict[str, Any]) -> None:
        started_at = time.perf_counter()
        model_name = str(payload["model"])
        months = int(payload["months"])
        target_model_version = str(payload.get("modelVersion") or self.next_model_version())
        metrics_file = METRICS_DIR / f"next_month_strategy_{model_name}_metrics.json"
        model_file = MODEL_DIR / f"next_month_strategy_{model_name}.joblib"
        prediction_file = PREDICTIONS_DIR / f"{current_month_yyyy_mm().replace('-', '_')}_strategy_predictions_{model_name}.csv"
        checkpoint_dir = CHECKPOINT_DIR / model_name

        prepare_command = self._build_prepare_training_command(months=months)
        command = self._build_training_command(
            model_name=model_name,
            months=months,
            metrics_file=metrics_file,
            model_file=model_file,
            prediction_file=prediction_file,
            checkpoint_dir=checkpoint_dir,
            train_source_months=payload.get("trainSourceMonths"),
            validation_source_months=payload.get("validationSourceMonths"),
            prediction_source_months=payload.get("predictionSourceMonths"),
        )
        self.logger.info(
            "Queued training pipeline | transaction_id=%s months=%s model=%s target_model_version=%s metrics_file=%s model_file=%s prediction_file=%s checkpoint_dir=%s",
            transaction_id,
            months,
            model_name,
            target_model_version,
            metrics_file,
            model_file,
            prediction_file,
            checkpoint_dir,
        )
        try:
            run_logged_subprocess(
                prepare_command,
                logger=self.logger,
                transaction_id=transaction_id,
                step_name="prepare_training_window",
                cwd=REPO_ROOT,
            )
            run_logged_subprocess(
                command,
                logger=self.logger,
                transaction_id=transaction_id,
                step_name="train_model",
                cwd=REPO_ROOT,
            )
            snapshot = read_metrics_snapshot(model_name)
            self.logger.info(
                "Training completed | transaction_id=%s current_accuracy=%s drift_percentage=%s metrics_file=%s model_file=%s",
                transaction_id,
                snapshot.get("currentAccuracy"),
                snapshot.get("driftPercentage"),
                metrics_file,
                model_file,
            )
            self.ai_config_repo.update_entry(
                transaction_id=transaction_id,
                status="COMPLETED",
                message="Processing completed.",
                model_version=target_model_version,
                accuracy=snapshot.get("currentAccuracy"),
                drift=snapshot.get("driftPercentage"),
                processing_time_ms=int((time.perf_counter() - started_at) * 1000),
                entry_type="TRAINING",
                training_window=str(months),
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Training job failed | transaction_id=%s", transaction_id)
            if isinstance(exc, subprocess.CalledProcessError):
                failed_step = "prepare_training_window" if exc.cmd == prepare_command else "train_model"
                error_message = summarize_subprocess_failure(exc, step_name=failed_step)
            else:
                error_message = str(exc)
            self.ai_config_repo.update_entry(
                transaction_id=transaction_id,
                status="FAILED",
                message=error_message,
                model_version=target_model_version,
                processing_time_ms=int((time.perf_counter() - started_at) * 1000),
                entry_type="TRAINING",
                training_window=str(months),
            )

    def _run_inference_job(self, *, transaction_id: str, payload: dict[str, Any]) -> None:
        started_at = time.perf_counter()
        source_month = str(payload["sourceMonth"])
        predict_month = str(payload["predictMonth"])
        model_name = str(payload["model"])
        model_version = self.current_model_version()
        command = self._build_inference_command(
            source_month=source_month,
            predict_month=predict_month,
            model_name=model_name,
        )
        self.logger.info(
            "Queued inference pipeline | transaction_id=%s source_month=%s predict_month=%s model=%s export_after_inference=%s export_write=%s",
            transaction_id,
            source_month,
            predict_month,
            model_name,
            self.config.export_after_inference,
            self.config.export_write,
        )
        try:
            run_logged_subprocess(
                command,
                logger=self.logger,
                transaction_id=transaction_id,
                step_name="run_inference_pipeline",
                cwd=REPO_ROOT,
            )
            if self.config.export_after_inference:
                export_command = self._build_export_command(
                    source_month=source_month,
                    predict_month=predict_month,
                    model_name=model_name,
                )
                run_logged_subprocess(
                    export_command,
                    logger=self.logger,
                    transaction_id=transaction_id,
                    step_name="export_recommendation_workbooks",
                    cwd=REPO_ROOT,
                )
            snapshot = read_metrics_snapshot(model_name)
            summary = read_prediction_summary(predict_month)
            drift_summary = summary.get("drift") if isinstance(summary.get("drift"), dict) else {}
            response_body = {
                "transactionId": transaction_id,
                "status": "COMPLETED",
                "message": "Metrics generated successfully",
                "modelVersion": model_version,
                "currentAccuracy": snapshot.get("currentAccuracy"),
                "driftPercentage": drift_summary.get("drift_percentage", snapshot.get("driftPercentage")),
                "overallPsi": drift_summary.get("overall_psi"),
                "maxFeaturePsi": drift_summary.get("max_feature_psi"),
                "driftStatus": drift_summary.get("status"),
                "sourceMonth": month_label(source_month),
                "predictMonth": month_label(predict_month),
                "schedulerExportTriggered": self.config.export_after_inference,
                "schedulerExportWrite": self.config.export_write if self.config.export_after_inference else None,
                "schedulerExportTriggerState": self.config.export_trigger_state if self.config.export_after_inference else None,
            }
            self.ai_config_repo.update_entry(
                transaction_id=transaction_id,
                status="COMPLETED",
                message="Metrics generated successfully",
                model_version=model_version,
                accuracy=response_body.get("currentAccuracy"),
                drift=response_body.get("driftPercentage"),
                processing_time_ms=int((time.perf_counter() - started_at) * 1000),
                entry_type="INFERENCE",
                training_window="10 days",
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Inference job failed | transaction_id=%s", transaction_id)
            if isinstance(exc, subprocess.CalledProcessError):
                failed_step = "export_recommendation_workbooks" if self.config.export_after_inference and exc.cmd != command else "run_inference_pipeline"
                error_message = summarize_subprocess_failure(exc, step_name=failed_step)
            else:
                error_message = str(exc)
            self.ai_config_repo.update_entry(
                transaction_id=transaction_id,
                status="FAILED",
                message=error_message,
                model_version=model_version,
                processing_time_ms=int((time.perf_counter() - started_at) * 1000),
                entry_type="INFERENCE",
                training_window="10 days",
            )

    def _build_prepare_training_command(self, *, months: int) -> list[str]:
        return [
            str(VENV_PYTHON),
            str(SCRIPTS_DIR / "prepare_training_window_from_postgres.py"),
            "--months",
            str(months),
            "--month-source",
            self.config.feature_month_source,
        ]

    def _build_training_command(
        self,
        *,
        model_name: str,
        months: int,
        metrics_file: Path,
        model_file: Path,
        prediction_file: Path,
        checkpoint_dir: Path,
        train_source_months: Any,
        validation_source_months: Any,
        prediction_source_months: Any,
    ) -> list[str]:
        if model_name == "logistic":
            command = [
                str(VENV_PYTHON),
                str(SCRIPTS_DIR / "train_next_month_strategy_model_logistic.py"),
                "--model-file",
                str(model_file),
                "--metrics-file",
                str(metrics_file),
                "--prediction-file",
                str(prediction_file),
            ]
        elif model_name in {"catboost", "catboost_3m"}:
            command = [
                str(VENV_PYTHON),
                str(SCRIPTS_DIR / "train_next_month_strategy_model_catboost.py"),
                "--history-window-months",
                str(months),
                "--model-file",
                str(model_file),
                "--metrics-file",
                str(metrics_file),
                "--prediction-file",
                str(prediction_file),
                "--checkpoint-dir",
                str(checkpoint_dir),
            ]
        else:
            command = [
                str(VENV_PYTHON),
                str(SCRIPTS_DIR / "train_next_month_strategy_model.py"),
                "--model-file",
                str(model_file),
                "--metrics-file",
                str(metrics_file),
                "--prediction-file",
                str(prediction_file),
            ]

        for flag, values in (
            ("--train-source-months", train_source_months),
            ("--validation-source-months", validation_source_months),
            ("--prediction-source-months", prediction_source_months),
        ):
            if isinstance(values, list) and values:
                command.append(flag)
                command.extend(str(value) for value in values)
        return command

    def _build_inference_command(self, *, source_month: str, predict_month: str, model_name: str) -> list[str]:
        return [
            str(VENV_PYTHON),
            str(SCRIPTS_DIR / "run_monthly_inference_pipeline.py"),
            "--source-month",
            source_month,
            "--predict-month",
            predict_month,
            "--model",
            model_name,
            "--feature-month-source",
            self.config.feature_month_source,
            "--campaign-vertical",
            self.config.campaign_vertical,
            "--campaign-vendor",
            self.config.campaign_vendor,
        ]

    def _build_export_command(self, *, source_month: str, predict_month: str, model_name: str) -> list[str]:
        command = [
            str(VENV_PYTHON),
            str(SCRIPTS_DIR / "export_recommendation_workbooks.py"),
            "--source-month",
            source_month,
            "--prediction-month",
            predict_month,
            "--model",
            model_name,
            "--output-dir",
            str(self.config.sftp_export_path),
        ]
        if self.config.export_write:
            command.append("--write")
        return command


def read_json_body(environ: dict[str, Any]) -> dict[str, Any]:
    content_length = int(environ.get("CONTENT_LENGTH") or 0)
    body = environ["wsgi.input"].read(content_length) if content_length else b""
    if not body:
        return {}
    return json.loads(body.decode("utf-8"))


def json_response(start_response: Callable[..., Any], status_code: int, payload: dict[str, Any]) -> list[bytes]:
    body = json.dumps(payload).encode("utf-8")
    start_response(
        f"{status_code} {HTTPStatus(status_code).phrase}",
        [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(body))),
        ],
    )
    return [body]


class AIMLApiApp:
    def __init__(self, service: AIMLApiService):
        self.service = service

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> list[bytes]:
        method = environ.get("REQUEST_METHOD", "GET").upper()
        path = environ.get("PATH_INFO", "")
        try:
            if path == "/api/health" and method == "GET":
                status_code, payload = self.service.health()
                return json_response(start_response, status_code, payload)

            if path == "/api/v1/auth" and method == "POST":
                status_code, payload = self.service.issue_auth_token(read_json_body(environ))
                return json_response(start_response, status_code, payload)

            auth_error = self._require_auth(environ)
            if auth_error is not None:
                return json_response(start_response, HTTPStatus.UNAUTHORIZED, auth_error)

            if path == "/api/status" and method == "GET":
                status_code, payload = self.service.model_status()
                return json_response(start_response, status_code, payload)

            if path.startswith("/api/v1/transactions/") and method == "GET":
                reference_number = path.removeprefix("/api/v1/transactions/").strip()
                status_code, payload = self.service.get_transaction_status(reference_number)
                return json_response(start_response, status_code, payload)

            if path == "/api/v1/training" and method == "POST":
                status_code, payload = self.service.trigger_training(read_json_body(environ), request_url=path)
                return json_response(start_response, status_code, payload)

            if path == "/api/v1/inference" and method == "POST":
                status_code, payload = self.service.trigger_inference(read_json_body(environ), request_url=path)
                return json_response(start_response, status_code, payload)

            if path in {"/api/health", "/api/status", "/api/v1/auth", "/api/v1/training", "/api/v1/inference"} or path.startswith("/api/v1/transactions/"):
                return json_response(start_response, HTTPStatus.METHOD_NOT_ALLOWED, {"message": "Method not allowed."})
            return json_response(start_response, HTTPStatus.NOT_FOUND, {"message": "Not found."})
        except PermissionError as exc:
            return json_response(start_response, HTTPStatus.UNAUTHORIZED, {"message": str(exc)})
        except ValueError as exc:
            return json_response(start_response, HTTPStatus.BAD_REQUEST, {"message": str(exc)})
        except Exception as exc:  # noqa: BLE001
            self.service.logger.exception("Unhandled API error")
            return json_response(start_response, HTTPStatus.INTERNAL_SERVER_ERROR, {"message": str(exc)})

    def _require_auth(self, environ: dict[str, Any]) -> dict[str, Any] | None:
        header = environ.get("HTTP_AUTHORIZATION", "")
        if not header.startswith("Bearer "):
            return {"message": "Missing bearer token."}
        token = header.removeprefix("Bearer ").strip()
        self.service.auth_manager.verify_token(token)
        return None


def build_app(config: ApiConfig | None = None) -> AIMLApiApp:
    resolved = config or ApiConfig.from_env()
    logger = setup_logger(resolved.log_file)
    service = AIMLApiService(
        config=resolved,
        auth_manager=AuthManager(
            username=resolved.auth_username,
            password=resolved.auth_password,
            secret=resolved.auth_secret,
            ttl_seconds=resolved.token_ttl_seconds,
        ),
        job_runner=BackgroundJobRunner(),
        logger=logger,
        ai_config_repo=AIConfigurationRepository(resolved),
    )
    return AIMLApiApp(service)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AIML integration API service.")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ApiConfig.from_env()
    if args.host:
        config = ApiConfig(**{**config.__dict__, "host": args.host})
    if args.port:
        config = ApiConfig(**{**config.__dict__, "port": args.port})
    app = build_app(config)
    with make_server(config.host, config.port, app) as server:
        app.service.logger.info("AIML API listening | host=%s port=%s", config.host, config.port)
        server.serve_forever()


if __name__ == "__main__":
    main()
