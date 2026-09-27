"""Pinned ICU 77.1 C API binding for Any-Latin; Latin-ASCII transliteration."""
import ctypes
import os
import platform
from pathlib import Path

from .normalization import normalize_business_name


class ICU:
    def __init__(self, native_dir=None):
        native = Path(native_dir or os.environ.get("V6_ICU_NATIVE_DIR") or
                      (Path(__file__).resolve().parents[1] / "native" / "icu"))
        if platform.system() == "Windows":
            names = ("icudt77.dll", "icuuc77.dll", "icuin77.dll")
            missing = [name for name in names if not (native / name).is_file()]
            if missing:
                raise FileNotFoundError(
                    f"ICU 77.1 Windows DLLs missing under {native}: {', '.join(missing)}. "
                    "Copy the official x64 ICU4C 77.1 DLLs there; no transliteration fallback is allowed."
                )
            self._dll_dir = os.add_dll_directory(str(native))
            suffix = "77"
            load = ctypes.CDLL
            data_name, uc_name, i18n_name = names
        elif platform.system() == "Linux":
            suffix = "77"
            load = ctypes.CDLL
            data_name, uc_name, i18n_name = ("libicudata.so.77", "libicuuc.so.77", "libicui18n.so.77")
            # Linux parity-fixture generation can opt into the same pinned library set.
            if not all((native / name).is_file() for name in (data_name, uc_name, i18n_name)):
                native = Path("/usr/lib64")
        else:
            raise RuntimeError(f"Unsupported ICU deployment platform: {platform.system()}")

        try:
            self.data = load(str(native / data_name))
            self.uc = load(str(native / uc_name))
            lib = load(str(native / i18n_name))
        except OSError as exc:
            raise RuntimeError(f"Could not load the pinned ICU 77.1 libraries from {native}: {exc}") from exc

        version = (ctypes.c_uint8 * 4)()
        version_fn = getattr(self.uc, f"u_getVersion_{suffix}")
        version_fn.argtypes = [ctypes.POINTER(ctypes.c_uint8)]
        version_fn.restype = None
        version_fn(version)
        if tuple(version) != (77, 1, 0, 0):
            raise RuntimeError(f"ICU version mismatch: expected 77.1.0.0, got {tuple(version)}")

        U16, I32 = ctypes.c_uint16, ctypes.c_int32
        self.U16, self.I32 = U16, I32
        op = getattr(lib, f"utrans_openU_{suffix}")
        op.restype = ctypes.c_void_p
        op.argtypes = [ctypes.POINTER(U16), I32, I32, ctypes.POINTER(U16), I32, ctypes.c_void_p, ctypes.POINTER(I32)]
        self.trans = getattr(lib, f"utrans_transUChars_{suffix}")
        self.trans.restype = None
        self.trans.argtypes = [ctypes.c_void_p, ctypes.POINTER(U16), ctypes.POINTER(I32), I32, I32,
                               ctypes.POINTER(I32), ctypes.POINTER(I32)]
        raw = "Any-Latin; Latin-ASCII".encode("utf-16-le")
        ident = (U16 * (len(raw) // 2)).from_buffer_copy(raw)
        status = I32(0)
        self.handle = op(ident, len(raw) // 2, 0, None, 0, None, ctypes.byref(status))
        if not self.handle or status.value > 0:
            raise RuntimeError(f"ICU transliterator init failed with status {status.value}")
        self.close_fn = getattr(lib, f"utrans_close_{suffix}")
        self.close_fn.argtypes = [ctypes.c_void_p]
        self.lib = lib

    def __call__(self, value):
        raw = value.encode("utf-16-le")
        length = len(raw) // 2
        capacity = max(256, length * 12 + 32)
        buffer = (self.U16 * capacity)()
        if raw:
            ctypes.memmove(buffer, raw, len(raw))
        current, limit, error = self.I32(length), self.I32(length), self.I32(0)
        self.trans(self.handle, buffer, ctypes.byref(current), capacity, 0,
                   ctypes.byref(limit), ctypes.byref(error))
        if error.value > 0:
            raise RuntimeError(f"ICU transliteration failed with status {error.value}")
        decoded = ctypes.string_at(ctypes.addressof(buffer), current.value * 2).decode("utf-16-le")
        return normalize_business_name(decoded)

    def close(self):
        self.close_fn(self.handle)
        self.handle = None
        if getattr(self, "_dll_dir", None):
            self._dll_dir.close()
