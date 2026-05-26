from __future__ import annotations

from collections.abc import Iterable, Mapping


_QUARTZ_JOB_DATA_MAP_PREFIX = bytes.fromhex(
    "aced0005737200156f72672e71756172747a2e4a6f62446174614d6170"
    "9fb083e8bfa9b0cb020000787200266f72672e71756172747a2e7574696c"
    "732e537472696e674b65794469727479466c61674d61708208e8c3fbc5"
    "5d280200015a0013616c6c6f77735472616e7369656e7444617461787200"
    "1d6f72672e71756172747a2e7574696c732e4469727479466c61674d6170"
    "13e62ead28760ace0200025a000564697274794c00036d617074000f4c6a"
    "6176612f7574696c2f4d61703b787001737200116a6176612e7574696c"
    "2e486173684d61700507dac1c31660d103000246000a6c6f616446616374"
    "6f724900097468726573686f6c6478703f4000000000000c770800000010"
)


def _java_modified_utf8(value: str) -> bytes:
    """Encode a Python string like Java ObjectOutputStream.writeUTF.

    The MCollect values are currently plain ASCII, but this supports the
    Java modified UTF-8 form for BMP characters so regional names can still
    be serialized if needed.
    """
    output = bytearray()
    for char in value:
        code = ord(char)
        if code == 0:
            output.extend((0xC0, 0x80))
        elif code <= 0x7F:
            output.append(code)
        elif code <= 0x7FF:
            output.extend((0xC0 | (code >> 6), 0x80 | (code & 0x3F)))
        elif code <= 0xFFFF:
            output.extend((0xE0 | (code >> 12), 0x80 | ((code >> 6) & 0x3F), 0x80 | (code & 0x3F)))
        else:
            raise ValueError(
                f"Cannot Java-serialize non-BMP character {char!r}; use a Java helper for this value."
            )
    if len(output) > 65535:
        raise ValueError("Java TC_STRING supports at most 65,535 encoded bytes.")
    return bytes(output)


def _tc_string(value: object) -> bytes:
    encoded = _java_modified_utf8(str(value))
    return b"\x74" + len(encoded).to_bytes(2, "big") + encoded


def serialize_quartz_job_data_map(entries: Mapping[str, object] | Iterable[tuple[str, object]]) -> bytes:
    """Return Java-serialized Quartz JobDataMap bytes for string keys/values.

    Existing MCollect Quartz rows store job_data as Java native serialization
    of org.quartz.JobDataMap. This intentionally mirrors the observed wire
    format for a String -> String JobDataMap so Python can create compatible
    bytea values until MCollect exposes a scheduler API.
    """
    items = list(entries.items() if isinstance(entries, Mapping) else entries)
    if not items:
        raise ValueError("Quartz JobDataMap must contain at least one entry.")
    if len(items) > 2**31 - 1:
        raise ValueError("Too many entries for Java HashMap serialization.")

    payload = bytearray(_QUARTZ_JOB_DATA_MAP_PREFIX)
    payload.extend(len(items).to_bytes(4, "big", signed=True))
    for key, value in items:
        payload.extend(_tc_string(key))
        payload.extend(_tc_string(value))
    payload.extend(b"\x78\x00")
    return bytes(payload)


def build_mcollect_job_data(dataset_name: str, template_name: str, vendor: str | None = None) -> bytes:
    entries: list[tuple[str, str]] = [("Datasets", dataset_name)]
    if vendor:
        entries.append(("Vendor", vendor))
    entries.append(("Template", template_name))
    return serialize_quartz_job_data_map(entries)
