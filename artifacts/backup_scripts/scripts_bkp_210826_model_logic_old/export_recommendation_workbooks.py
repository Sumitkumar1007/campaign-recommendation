from __future__ import annotations

import argparse
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import pandas as pd
from psycopg import sql

from env_utils import load_dotenv
from export_mcollect_scheduler import dataset_query_for, fetch_campaign_rows, normalize_month_label
from postgres_utils import PostgresConfig, connect_db, qualified_identifier

try:
    import paramiko
except ImportError:  # pragma: no cover - exercised via runtime env
    paramiko = None


DEFAULT_OUTPUT_DIR = Path("artifacts/exports/sftp")


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class SFTPConfig:
    enabled: bool
    host: str
    port: int
    username: str
    password: str | None
    private_key_path: Path | None
    private_key_passphrase: str | None
    remote_path: str
    remote_dataset_path: str
    remote_scheduler_path: str
    retries: int
    retry_delay_seconds: float
    timeout_seconds: int
    fail_on_error: bool


@dataclass(frozen=True)
class ExportConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: str
    schema: str
    campaign_table: str
    mapping_table: str
    source_month: str
    prediction_month: str
    model: str
    output_dir: Path
    write: bool
    sftp: SFTPConfig


def parse_args() -> ExportConfig:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(description="Export recommendation dataset/scheduler workbooks. Dry-run by default.")
    parser.add_argument("--host", default=os.getenv("PGHOST"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5432")))
    parser.add_argument("--dbname", default=os.getenv("PGDATABASE"))
    parser.add_argument("--user", default=os.getenv("PGUSER"))
    parser.add_argument("--password", default=os.getenv("PGPASSWORD"))
    parser.add_argument("--schema", default=os.getenv("TARGET_SCHEMA", "digital_collections"))
    parser.add_argument("--campaign-table", default=os.getenv("CAMPAIGN_TABLE", "ai_ml_campaign_recommendations"))
    parser.add_argument("--mapping-table", default=os.getenv("CAMPAIGN_MAPPING_TABLE", "ai_ml_campaign_mapping"))
    parser.add_argument("--source-month", required=True)
    parser.add_argument("--prediction-month", required=True)
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "catboost_3m"))
    parser.add_argument("--output-dir", default=os.getenv("SFTP_EXPORT_PATH", str(DEFAULT_OUTPUT_DIR)))
    parser.add_argument("--write", action="store_true", help="Actually create XLSX files.")
    parser.add_argument("--sftp-upload", action="store_true", default=env_flag("SFTP_UPLOAD_ENABLED", False))
    parser.add_argument("--sftp-host", default=os.getenv("SFTP_HOST", ""))
    parser.add_argument("--sftp-port", type=int, default=int(os.getenv("SFTP_PORT", "22")))
    parser.add_argument("--sftp-username", default=os.getenv("SFTP_USERNAME", ""))
    parser.add_argument("--sftp-password", default=os.getenv("SFTP_PASSWORD"))
    parser.add_argument("--sftp-private-key-path", default=os.getenv("SFTP_PRIVATE_KEY_PATH"))
    parser.add_argument("--sftp-private-key-passphrase", default=os.getenv("SFTP_PRIVATE_KEY_PASSPHRASE"))
    parser.add_argument("--sftp-remote-path", default=os.getenv("SFTP_REMOTE_PATH", ""))
    parser.add_argument("--sftp-remote-dataset-path", default=os.getenv("SFTP_REMOTE_DATASET_PATH", os.getenv("SFTP_REMOTE_PATH", "")))
    parser.add_argument("--sftp-remote-scheduler-path", default=os.getenv("SFTP_REMOTE_SCHEDULER_PATH", os.getenv("SFTP_REMOTE_PATH", "")))
    parser.add_argument("--sftp-retries", type=int, default=int(os.getenv("SFTP_RETRIES", "3")))
    parser.add_argument("--sftp-retry-delay-seconds", type=float, default=float(os.getenv("SFTP_RETRY_DELAY_SECONDS", "5")))
    parser.add_argument("--sftp-timeout-seconds", type=int, default=int(os.getenv("SFTP_TIMEOUT_SECONDS", "30")))
    parser.add_argument("--sftp-fail-on-error", action="store_true", default=env_flag("SFTP_FAIL_ON_ERROR", False))
    args = parser.parse_args()
    missing = [
        key
        for key, value in {
            "PGHOST/--host": args.host,
            "PGDATABASE/--dbname": args.dbname,
            "PGUSER/--user": args.user,
            "PGPASSWORD/--password": args.password,
        }.items()
        if not value
    ]
    if missing:
        parser.error("Missing required environment variables or CLI args: " + ", ".join(missing))

    source_month = normalize_month_label(args.source_month)
    prediction_month = normalize_month_label(args.prediction_month)
    output_dir = Path(args.output_dir)
    sftp = SFTPConfig(
        enabled=bool(args.sftp_upload),
        host=str(args.sftp_host).strip(),
        port=int(args.sftp_port),
        username=str(args.sftp_username).strip(),
        password=args.sftp_password,
        private_key_path=Path(args.sftp_private_key_path) if args.sftp_private_key_path else None,
        private_key_passphrase=args.sftp_private_key_passphrase,
        remote_path=str(args.sftp_remote_path).strip(),
        remote_dataset_path=str(args.sftp_remote_dataset_path).strip(),
        remote_scheduler_path=str(args.sftp_remote_scheduler_path).strip(),
        retries=max(1, int(args.sftp_retries)),
        retry_delay_seconds=max(0.0, float(args.sftp_retry_delay_seconds)),
        timeout_seconds=max(1, int(args.sftp_timeout_seconds)),
        fail_on_error=bool(args.sftp_fail_on_error),
    )
    if sftp.enabled:
        validate_sftp_config(parser, sftp)

    return ExportConfig(
        host=str(args.host),
        port=int(args.port),
        dbname=str(args.dbname),
        user=str(args.user),
        password=str(args.password),
        schema=str(args.schema),
        campaign_table=str(args.campaign_table),
        mapping_table=str(args.mapping_table),
        source_month=source_month,
        prediction_month=prediction_month,
        model=str(args.model),
        output_dir=output_dir,
        write=bool(args.write),
        sftp=sftp,
    )


def validate_sftp_config(parser: argparse.ArgumentParser, config: SFTPConfig) -> None:
    missing: list[str] = []
    if not config.host:
        missing.append("SFTP_HOST/--sftp-host")
    if not config.username:
        missing.append("SFTP_USERNAME/--sftp-username")
    if not config.remote_dataset_path:
        missing.append("SFTP_REMOTE_DATASET_PATH/--sftp-remote-dataset-path")
    if not config.remote_scheduler_path:
        missing.append("SFTP_REMOTE_SCHEDULER_PATH/--sftp-remote-scheduler-path")
    if not config.password and config.private_key_path is None:
        missing.append("SFTP_PASSWORD or SFTP_PRIVATE_KEY_PATH")
    if missing:
        parser.error("Missing SFTP configuration: " + ", ".join(missing))


def fetch_campaign_reasons(conn, *, schema: str, mapping_table: str, source_month: str, prediction_month: str, model_name: str) -> dict[str, str]:
    query = sql.SQL(
        """
        SELECT campaign_name, prediction_reason
        FROM {table_ref}
        WHERE source_month = %s
          AND prediction_month = %s
          AND model_name = %s
        ORDER BY campaign_name, loan_number
        """
    ).format(table_ref=qualified_identifier(schema, mapping_table))
    reasons: dict[str, str] = {}
    with conn.cursor() as cur:
        cur.execute(query, (source_month, prediction_month, model_name))
        for campaign_name, prediction_reason in cur.fetchall():
            if campaign_name not in reasons and prediction_reason:
                reasons[str(campaign_name)] = str(prediction_reason)
    return reasons


def build_dataset_rows(campaigns: pd.DataFrame, *, schema: str, campaign_table: str, mapping_table: str) -> list[list[str]]:
    rows: list[list[str]] = []
    seen: set[str] = set()
    for _, row in campaigns.iterrows():
        dataset_name = str(row["dataset_name"])
        if dataset_name in seen:
            continue
        seen.add(dataset_name)
        rows.append([dataset_name, dataset_query_for(row, schema=schema, campaign_table=campaign_table, mapping_table=mapping_table)])
    return rows


def build_scheduler_rows(campaigns: pd.DataFrame, reasons: dict[str, str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for _, row in campaigns.iterrows():
        campaign_name = str(row["name"])
        rows.append([
            campaign_name,
            str(row["mode"]),
            str(row["date"]),
            str(row["time"]),
            str(row["template_name"]),
            str(row["dataset_name"]),
            str(row["vendor"]),
            str(row["active"]),
            reasons.get(campaign_name, ""),
        ])
    return rows


def next_output_path(output_dir: Path, prefix: str) -> Path:
    stamp = pd.Timestamp.today().strftime("%d%m%Y")
    output_dir.mkdir(parents=True, exist_ok=True)
    for seq in range(1, 100):
        path = output_dir / f"{prefix}_{stamp}_{seq:02d}.xlsx"
        if not path.exists():
            return path
    raise RuntimeError(f"No available output slot for {prefix}_{stamp}_NN.xlsx")


def _col_name(index: int) -> str:
    result = ""
    value = index
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _sheet_xml(headers: list[str], rows: list[list[str]]) -> str:
    xml_rows: list[str] = []
    for row_index, values in enumerate([headers, *rows], start=1):
        cells: list[str] = []
        for col_index, value in enumerate(values, start=1):
            ref = f"{_col_name(col_index)}{row_index}"
            cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape("" if value is None else str(value))}</t></is></c>')
        xml_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(xml_rows)}</sheetData>'
        '</worksheet>'
    )


def write_xlsx(path: Path, *, headers: list[str], rows: list[list[str]]) -> None:
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as workbook:
        workbook.writestr('[Content_Types].xml', '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/></Types>')
        workbook.writestr('_rels/.rels', '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        workbook.writestr('xl/workbook.xml', '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>')
        workbook.writestr('xl/_rels/workbook.xml.rels', '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>')
        workbook.writestr('xl/styles.xml', '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts><fills count="1"><fill><patternFill patternType="none"/></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf/></cellStyleXfs><cellXfs count="1"><xf xfId="0"/></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>')
        workbook.writestr('xl/worksheets/sheet1.xml', _sheet_xml(headers, rows))


def _load_private_key(config: SFTPConfig) -> Any:
    if config.private_key_path is None:
        return None
    if paramiko is None:
        raise RuntimeError("paramiko is required for SFTP upload but is not installed.")
    key_exceptions = tuple(
        exc for exc in [
            getattr(paramiko, 'PasswordRequiredException', None),
            getattr(paramiko, 'SSHException', None),
        ]
        if exc is not None
    )
    for key_cls in (
        getattr(paramiko, 'RSAKey', None),
        getattr(paramiko, 'Ed25519Key', None),
        getattr(paramiko, 'ECDSAKey', None),
    ):
        if key_cls is None:
            continue
        try:
            return key_cls.from_private_key_file(str(config.private_key_path), password=config.private_key_passphrase)
        except key_exceptions:
            continue
    return getattr(paramiko, 'PKey').from_private_key_file(str(config.private_key_path), password=config.private_key_passphrase)


def _connect_sftp(config: SFTPConfig):
    if paramiko is None:
        raise RuntimeError("paramiko is required for SFTP upload but is not installed.")
    transport = paramiko.Transport((config.host, config.port))
    transport.banner_timeout = config.timeout_seconds
    transport.auth_timeout = config.timeout_seconds
    transport.connect(username=config.username, password=config.password, pkey=_load_private_key(config))
    client = paramiko.SFTPClient.from_transport(transport)
    return transport, client


def ensure_remote_dir(sftp_client: Any, remote_dir: str) -> None:
    normalized = remote_dir.replace('\\', '/').strip()
    if not normalized:
        return
    current = '/' if normalized.startswith('/') else ''
    for part in [token for token in normalized.split('/') if token]:
        current = f"{current}/{part}" if current not in {'', '/'} else f"/{part}" if current == '/' else part
        try:
            sftp_client.stat(current)
        except Exception:
            sftp_client.mkdir(current)


def remote_file_path(remote_dir: str, local_path: Path) -> str:
    prefix = remote_dir.rstrip('/').rstrip('\\')
    return f"{prefix}/{local_path.name}" if prefix else local_path.name


def upload_file(local_path: Path, config: SFTPConfig, *, remote_dir: str) -> str:
    transport = None
    sftp_client = None
    remote_path = remote_file_path(remote_dir, local_path)
    try:
        transport, sftp_client = _connect_sftp(config)
        ensure_remote_dir(sftp_client, remote_dir)
        sftp_client.put(str(local_path), remote_path)
        return remote_path
    finally:
        if sftp_client is not None:
            sftp_client.close()
        if transport is not None:
            transport.close()


def upload_with_retry(local_path: Path, config: SFTPConfig, *, remote_dir: str) -> tuple[bool, str | None, str | None]:
    last_error: Exception | None = None
    for attempt in range(1, config.retries + 1):
        try:
            return True, upload_file(local_path, config, remote_dir=remote_dir), None
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < config.retries:
                time.sleep(config.retry_delay_seconds)
    message = f"SFTP upload failed for {local_path.name}: {last_error}"
    if config.fail_on_error:
        raise RuntimeError(message) from last_error
    return False, None, message


def upload_files(file_specs: list[tuple[Path, str]], config: SFTPConfig) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not config.enabled:
        return results
    for local_path, remote_dir in file_specs:
        uploaded, remote_path, error = upload_with_retry(local_path, config, remote_dir=remote_dir)
        result = {
            'localPath': str(local_path),
            'uploaded': uploaded,
            'remotePath': remote_path,
            'error': error,
        }
        results.append(result)
        if error:
            print(f"warning={error}")
        elif remote_path:
            print(f"uploaded={remote_path}")
    return results


def main() -> None:
    args = parse_args()
    config = PostgresConfig(host=args.host, port=args.port, dbname=args.dbname, user=args.user, password=args.password)
    with connect_db(config) as conn:
        campaigns = fetch_campaign_rows(conn, schema=args.schema, campaign_table=args.campaign_table, source_month=args.source_month, prediction_month=args.prediction_month, model_name=args.model)
        reasons = fetch_campaign_reasons(conn, schema=args.schema, mapping_table=args.mapping_table, source_month=args.source_month, prediction_month=args.prediction_month, model_name=args.model)

    dataset_rows = build_dataset_rows(campaigns, schema=args.schema, campaign_table=args.campaign_table, mapping_table=args.mapping_table)
    scheduler_rows = build_scheduler_rows(campaigns, reasons)
    dataset_path = next_output_path(args.output_dir, 'MD_UB_DATASET')
    scheduler_path = next_output_path(args.output_dir, 'MD_UB_SCHEDULER')

    print(f'dataset_rows={len(dataset_rows)} path={dataset_path}')
    print(f'scheduler_rows={len(scheduler_rows)} path={scheduler_path}')
    if not args.write:
        print('dry_run=true')
        return

    write_xlsx(dataset_path, headers=['Name', 'Query'], rows=dataset_rows)
    write_xlsx(scheduler_path, headers=['Name', 'Mode', 'Date', 'Time', 'Template Name', 'Dataset Name', 'Vendor', 'Active', 'Reason'], rows=scheduler_rows)
    results = upload_files(
        [
            (dataset_path, args.sftp.remote_dataset_path),
            (scheduler_path, args.sftp.remote_scheduler_path),
        ],
        args.sftp,
    )
    print('dry_run=false')
    if results:
        print(f'sftp_uploads={results}')


if __name__ == '__main__':
    main()
