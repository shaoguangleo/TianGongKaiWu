"""Strict, bounded decoding of display PNGs; no scientific flux reconstruction."""

import hashlib
import io
from pathlib import Path
import re
import stat
import struct
import zlib

import numpy as np
from PIL import Image

MAX_PNG_BYTES = 64 * 1024 * 1024
MAX_PNG_PIXELS = 16_000_000
_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _inspect_png(content: bytes) -> tuple[int, int, str]:
    """Check dimensions before allocation, and all CRCs including the IEND CRC."""
    if not content.startswith(_SIGNATURE):
        raise ValueError("需要真实 PNG 文件，不能仅修改扩展名")
    offset, width, height, mode = 8, 0, 0, ""
    seen_data = False
    data_ended = False
    decoder = zlib.decompressobj()
    expected_bytes = 0
    decoded_bytes = 0
    while offset < len(content):
        if offset + 12 > len(content):
            raise ValueError("PNG chunk 截断")
        length = struct.unpack_from(">I", content, offset)[0]
        kind = content[offset + 4:offset + 8]
        end = offset + 12 + length
        if end > len(content):
            raise ValueError("PNG chunk 内容截断")
        data = memoryview(content)[offset + 8:end - 4]
        crc = zlib.crc32(data, zlib.crc32(kind))
        if crc != struct.unpack_from(">I", content, end - 4)[0]:
            raise ValueError("PNG chunk CRC 校验失败")
        if not re.fullmatch(rb"[A-Za-z]{4}", kind):
            raise ValueError("PNG chunk 类型无效")
        if offset == 8 and kind != b"IHDR":
            raise ValueError("PNG 首个 chunk 必须是 IHDR")
        if kind == b"IHDR":
            if offset != 8 or length != 13:
                raise ValueError("PNG IHDR 重复或长度无效")
            width, height, depth, color, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", data,
            )
            if not width or not height or width * height > MAX_PNG_PIXELS:
                raise ValueError("PNG 尺寸无效或超过 16000000 pixel 上限")
            if depth != 8 or color not in (2, 6):
                raise ValueError("PNG 只支持 8-bit RGB/RGBA 模式")
            if compression != 0 or filtering != 0 or interlace not in (0, 1):
                raise ValueError("PNG 编码参数不受支持")
            mode = "RGB" if color == 2 else "RGBA"
            channels = 3 if color == 2 else 4
            # Each nonempty scanline has one filter byte. Adam7 has seven passes.
            passes = [(0, 0, 1, 1)] if interlace == 0 else [
                (0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4),
                (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2),
            ]
            for x, y, dx, dy in passes:
                pass_width = max(0, (width - x + dx - 1) // dx)
                pass_height = max(0, (height - y + dy - 1) // dy)
                if pass_width and pass_height:
                    expected_bytes += (pass_width * channels + 1) * pass_height
        elif kind in (b"acTL", b"fcTL", b"fdAT"):
            raise ValueError("不支持 APNG 动画")
        elif kind == b"tRNS":
            raise ValueError("PNG tRNS transparency 不满足全不透明 alpha 策略")
        elif kind == b"IDAT":
            if data_ended:
                raise ValueError("PNG IDAT chunks 必须连续")
            seen_data = True
            pending = data
            try:
                while pending:
                    # Pillow can accept a complete pixel buffer with a missing
                    # zlib checksum. Check stream EOF independently, discarding
                    # bounded output blocks instead of allocating another image.
                    block = decoder.decompress(pending, min(65536, expected_bytes - decoded_bytes + 1))
                    decoded_bytes += len(block)
                    if decoded_bytes > expected_bytes or decoder.unused_data:
                        raise ValueError("PNG zlib 流包含多余像素或尾随数据")
                    pending = decoder.unconsumed_tail
            except zlib.error as exc:
                raise ValueError(f"PNG zlib 流损坏: {exc}") from exc
        elif kind == b"IEND":
            if length != 0 or not seen_data or end != len(content):
                raise ValueError("PNG IEND 无效、缺少像素数据或含尾随字节")
            if not decoder.eof or decoded_bytes != expected_bytes:
                raise ValueError("PNG zlib 流或像素数据截断")
            return width, height, mode
        elif kind != b"PLTE" and kind[0] < ord("a"):
            raise ValueError("PNG 含不支持的 critical chunk")
        if seen_data and kind != b"IDAT":
            data_ended = True
        offset = end
    raise ValueError("PNG 缺少 IEND，文件可能截断")


def read_png(
    path: str | Path, *, expected_sha256: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Return contiguous RGB CHW float32 in [0, 1], plus frozen-byte metadata.

    The byte limit is 64 MiB and the decoded image limit is 16 million pixels.
    Hashing, verification and full decoding use one bounded byte snapshot; the
    original RGB/RGBA mode is retained in metadata. No resizing is performed.
    """
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ValueError("expected_sha256 必须是 64 位小写十六进制 SHA256")
    path = Path(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("PNG 路径必须指向普通文件")
    if info.st_size > MAX_PNG_BYTES:
        raise ValueError("PNG 文件大小超过 64 MiB 上限")
    with path.open("rb") as stream:
        content = stream.read(MAX_PNG_BYTES + 1)
    if len(content) > MAX_PNG_BYTES:
        raise ValueError("PNG 文件大小超过 64 MiB 上限")
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("PNG SHA256 与清单不一致，拒绝读取变更后的图片")
    width, height, mode = _inspect_png(content)
    try:
        # verify does not decode pixels. Reopen the same snapshot before load.
        with Image.open(io.BytesIO(content), formats=["PNG"]) as verified:
            verified.verify()
        with Image.open(io.BytesIO(content), formats=["PNG"]) as decoded:
            decoded.load()
            if decoded.size != (width, height) or decoded.mode != mode:
                raise ValueError("PNG 解码尺寸或模式与 IHDR 不一致")
            pixels = np.asarray(decoded)
            if mode == "RGBA" and np.any(pixels[..., 3] != 255):
                raise ValueError("PNG alpha 必须全不透明 (opaque)")
            image = np.ascontiguousarray(pixels[..., :3].transpose(2, 0, 1), dtype=np.float32)
            image /= np.float32(255)
    except (OSError, SyntaxError, EOFError, ValueError, struct.error) as exc:
        raise ValueError(f"PNG 验证或完整解码失败: {exc}") from exc
    return image, {"width": width, "height": height, "mode": mode, "sha256": digest}
