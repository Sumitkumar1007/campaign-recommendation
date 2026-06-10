from __future__ import annotations

import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import export_recommendation_workbooks as erw


class FakeTransport:
    def __init__(self, address):
        self.address = address
        self.banner_timeout = None
        self.auth_timeout = None
        self.connect_calls = []
        self.closed = False

    def connect(self, **kwargs):
        self.connect_calls.append(kwargs)

    def close(self):
        self.closed = True


class FakeSFTPClient:
    put_calls = 0
    mkdir_calls: list[str] = []
    stat_calls: list[str] = []
    closed = False

    @classmethod
    def from_transport(cls, transport):
        return cls()

    def stat(self, path):
        self.stat_calls.append(path)
        if path in {'/remote', '/remote/drop'}:
            return object()
        raise FileNotFoundError(path)

    def mkdir(self, path):
        self.mkdir_calls.append(path)

    def put(self, local_path, remote_path):
        type(self).put_calls += 1
        if type(self).put_calls == 1:
            raise OSError('temporary failure')
        self.last_upload = (local_path, remote_path)

    def close(self):
        self.closed = True


class FakeParamiko(types.SimpleNamespace):
    pass


def test_upload_with_retry_succeeds_after_transient_failure(tmp_path, monkeypatch):
    local_file = tmp_path / 'sample.xlsx'
    local_file.write_text('x', encoding='utf-8')

    fake_paramiko = FakeParamiko(
        Transport=FakeTransport,
        SFTPClient=FakeSFTPClient,
        PKey=types.SimpleNamespace(from_private_key_file=lambda *args, **kwargs: None),
        RSAKey=None,
        Ed25519Key=None,
        ECDSAKey=None,
    )
    monkeypatch.setattr(erw, 'paramiko', fake_paramiko)
    monkeypatch.setattr(erw.time, 'sleep', lambda seconds: None)

    config = erw.SFTPConfig(
        enabled=True,
        host='example.com',
        port=22,
        username='user',
        password='secret',
        private_key_path=None,
        private_key_passphrase=None,
        remote_path='/remote/drop',
        retries=3,
        retry_delay_seconds=0.0,
        timeout_seconds=10,
        fail_on_error=True,
    )

    uploaded, remote_path, error = erw.upload_with_retry(local_file, config)

    assert uploaded is True
    assert remote_path == '/remote/drop/sample.xlsx'
    assert error is None
    assert FakeSFTPClient.put_calls == 2
