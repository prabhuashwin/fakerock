"""Encoder for the AWS binary event-stream format (application/vnd.amazon.eventstream).

Message = total_length(u32) | headers_length(u32) | prelude CRC32 | headers | payload | message CRC32.
Header  = name_len(u8) | name | type 7 (string) | value_len(u16) | value.
"""
import json
import struct
import zlib


def _header(name: str, value: str) -> bytes:
    n, v = name.encode("utf-8"), value.encode("utf-8")
    return struct.pack(">B", len(n)) + n + b"\x07" + struct.pack(">H", len(v)) + v


def encode_message(headers: dict, payload: bytes) -> bytes:
    hb = b"".join(_header(k, v) for k, v in headers.items())
    total = 12 + len(hb) + len(payload) + 4
    prelude = struct.pack(">II", total, len(hb))
    prelude += struct.pack(">I", zlib.crc32(prelude) & 0xFFFFFFFF)
    body = prelude + hb + payload
    return body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def event(event_type: str, payload_dict: dict) -> bytes:
    return encode_message({":event-type": event_type, ":content-type": "application/json",
                           ":message-type": "event"}, json.dumps(payload_dict).encode("utf-8"))


def exception(exception_type: str, message: str) -> bytes:
    return encode_message({":exception-type": exception_type, ":content-type": "application/json",
                           ":message-type": "exception"}, json.dumps({"message": message}).encode("utf-8"))
