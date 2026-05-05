"""
Rust-accelerated regex engine for boilerplate cleaning.

This module is a thin Python shim around the ``boilerplate_cleaner_rs``
PyO3 extension.  It exposes the same surface API that the rest of the
package uses so callers never need to know whether the Rust binary is
available or not — the Python fallback is used transparently.

Public API
----------
clean_text(text, *, use_legal_patterns, use_common_patterns, custom_patterns) -> str
clean_texts(texts, *, use_legal_patterns, use_common_patterns, custom_patterns) -> list[str]
list_turkish_patterns() -> list[str]
list_common_patterns() -> list[str]
RUST_AVAILABLE : bool
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
#  Try to load the compiled Rust extension
# ──────────────────────────────────────────────────────────────

try:
    import boilerplate_cleaner_rs as _rs  # type: ignore[import]

    RUST_AVAILABLE = True
    logger.debug("boilerplate_cleaner_rs: Rust extension loaded — fast path active.")
except ImportError:
    _rs = None  # type: ignore[assignment]
    RUST_AVAILABLE = False
    logger.info(
        "boilerplate_cleaner_rs: Rust extension not found — falling back to pure Python. "
        "Run `maturin develop --release` inside "
        "rust_extensions/boilerplate_cleaner_rs/ to enable the fast path."
    )

# ──────────────────────────────────────────────────────────────
#  Python fallback pattern lists (mirrors lib.rs exactly)
# ──────────────────────────────────────────────────────────────

_TURKISH_LEGAL_RAW: List[str] = [
    r"TÜRKİYE\s+CUMHURİYETİ[^\n]*",
    r"T\.C\.\s*[^\n]*",
    r"Sayfa\s+\d+\s*/\s*\d+[^\n]*",
    r"Sayfa\s+\d+[^\n]*",
    r"Belge\s+No:\s*[^\n]*",
    r"Referans\s+No:\s*[^\n]*",
    r"Dosya\s+No:\s*[^\n]*",
    r"Kayıt\s+No:\s*[^\n]*",
    r"Bu\s+sözleşme\s+taraflar\s+arasında\s+imzalanmıştır[^\n]*",
    r"Taraflar\s+arasında\s+imzalanan\s+sözleşme[^\n]*",
    r"Mahkeme\s+kararı\s+kesinleşmiştir[^\n]*",
    r"Yargıtay\s+kararı\s+kesinleşmiştir[^\n]*",
    r"Danıştay\s+kararı\s+kesinleşmiştir[^\n]*",
    r"Anayasa\s+Mahkemesi\s+kararı[^\n]*",
    r"Bu\s+karar\s+kesinleşmiştir[^\n]*",
    r"Karar\s+kesinleşmiştir[^\n]*",
    r"Tarih:\s*\d{1,2}\.\d{1,2}\.\d{4}[^\n]*",
    r"Tarih\s+ve\s+saat:\s*[^\n]*",
    r"İmza\s+tarihi:\s*[^\n]*",
    r"Yürürlük\s+tarihi:\s*[^\n]*",
    r"Yayın\s+tarihi:\s*[^\n]*",
    r"Madde\s+\d+[^\n]*",
    r"Bölüm\s+\d+[^\n]*",
    r"Kısım\s+\d+[^\n]*",
    r"Fıkra\s+\d+[^\n]*",
    r"Bent\s+\d+[^\n]*",
    r"Yukarıda\s+adı\s+ve\s+soyadı\s+yazılı[^\n]*",
    r"Taraflar\s+arasında\s+anlaşma\s+sağlanmıştır[^\n]*",
    r"Bu\s+belge\s+ile\s+ilgili\s+[^\n]*",
    r"Belgenin\s+devamı\s+[^\n]*",
    r"Ek\s+\d+[^\n]*",
    r"Ekler\s+[^\n]*",
    r"Mahkeme\s+adı:\s*[^\n]*",
    r"Mahkeme\s+merkezi:\s*[^\n]*",
    r"Duruşma\s+tarihi:\s*[^\n]*",
    r"Duruşma\s+saati:\s*[^\n]*",
    r"Duruşma\s+yeri:\s*[^\n]*",
    r"Dava\s+türü:\s*[^\n]*",
    r"Dava\s+konusu:\s*[^\n]*",
    r"Talep\s+edilen:\s*[^\n]*",
    r"Talep\s+eden:\s*[^\n]*",
    r"Davalı:\s*[^\n]*",
    r"Durum:\s*[^\n]*",
    r"Sonuç:\s*[^\n]*",
    r"Karar:\s*[^\n]*",
    r"Hüküm:\s*[^\n]*",
    r"GİRİŞ[^\n]*",
    r"SONUÇ[^\n]*",
    r"KARAR[^\n]*",
    r"HÜKÜM[^\n]*",
    r"GEREKÇE[^\n]*",
    r"MADDELER[^\n]*",
    r"EK[^\n]*",
    r"Bu\s+karar\s+ile\s+ilgili\s+[^\n]*",
    r"Karar\s+bu\s+şekilde\s+verilmiştir[^\n]*",
    r"Hüküm\s+bu\s+şekilde\s+verilmiştir[^\n]*",
    r"Taraflar\s+bilgilendirilmiştir[^\n]*",
    r"İtiraz\s+hakkı\s+saklıdır[^\n]*",
    r"Yasal\s+süre\s+içinde\s+[^\n]*",
    r"15/\d{2}/\d{4}\s+tarihinde\s+kesin\s+olarak,\s+oyçokluğuyla\s+karar\s+verildi\.",
    r"\(X\)\s+KARŞI\s+OY\s*:\s*[^\n]*",
    r"KARAR\s+SONUCU\s*:\s*[^\n]*",
    r"kesin\s+olarak,\s+oyçokluğuyla\s+karar\s+verildi\.",
    r"oyçokluğuyla\s+karar\s+verildi\.",
    r"karar\s+verildi\.",
]

_COMMON_RAW: List[str] = [
    r"Copyright[^\n]*",
    r"All\s+rights\s+reserved[^\n]*",
    r"Confidential[^\n]*",
    r"Proprietary[^\n]*",
    r"Page\s+\d+\s+of\s+\d+[^\n]*",
    r"Generated\s+on[^\n]*",
    r"Last\s+updated[^\n]*",
    r"Version\s+\d+\.\d+[^\n]*",
    r"Document\s+ID:[^\n]*",
    r"Reference:[^\n]*",
    r"Telif\s+hakkı[^\n]*",
    r"Tüm\s+haklar\s+saklıdır[^\n]*",
    r"Gizli[^\n]*",
    r"Özel[^\n]*",
    r"Oluşturulma\s+tarihi[^\n]*",
    r"Son\s+güncelleme[^\n]*",
    r"Sürüm\s+\d+\.\d+[^\n]*",
    r"Belge\s+kimliği[^\n]*",
    r"Referans[^\n]*",
]

# Compile once at import time (Python fallback only).
_COMPILED_TURKISH = [re.compile(p, re.IGNORECASE) for p in _TURKISH_LEGAL_RAW]
_COMPILED_COMMON = [re.compile(p, re.IGNORECASE) for p in _COMMON_RAW]
_WS_RE = re.compile(r"\s{2,}")


def _py_clean_one(
    text: str,
    use_legal: bool,
    use_common: bool,
    custom_compiled: List[re.Pattern],  # type: ignore[type-arg]
) -> str:
    """Pure-Python fallback — same semantics as the Rust implementation."""
    result = text
    if use_legal:
        for pat in _COMPILED_TURKISH:
            result = pat.sub("", result)
    if use_common:
        for pat in _COMPILED_COMMON:
            result = pat.sub("", result)
    for pat in custom_compiled:
        result = pat.sub("", result)
    result = _WS_RE.sub(" ", result).strip()
    return result


# ──────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────


def clean_text(
    text: str,
    *,
    use_legal_patterns: bool = True,
    use_common_patterns: bool = True,
    custom_patterns: Optional[List[str]] = None,
) -> str:
    """Clean boilerplate from a single string.

    Uses the Rust extension when available; falls back to Python otherwise.

    Parameters
    ----------
    text:
        Input text to clean.
    use_legal_patterns:
        Apply Turkish legal boilerplate patterns (default ``True``).
    use_common_patterns:
        Apply common English+Turkish boilerplate patterns (default ``True``).
    custom_patterns:
        Optional list of extra regex strings (compiled with IGNORECASE).

    Returns
    -------
    str
        The cleaned text.
    """
    if RUST_AVAILABLE:
        return _rs.clean_text(  # type: ignore[union-attr]
            text,
            use_legal_patterns,
            use_common_patterns,
            custom_patterns,
        )
    custom_compiled = (
        [re.compile(p, re.IGNORECASE) for p in custom_patterns]
        if custom_patterns
        else []
    )
    return _py_clean_one(text, use_legal_patterns, use_common_patterns, custom_compiled)


def clean_texts(
    texts: List[str],
    *,
    use_legal_patterns: bool = True,
    use_common_patterns: bool = True,
    custom_patterns: Optional[List[str]] = None,
) -> List[str]:
    """Clean boilerplate from a list of strings (parallel when Rust is available).

    Parameters
    ----------
    texts:
        Input strings to clean.
    use_legal_patterns:
        Apply Turkish legal patterns (default ``True``).
    use_common_patterns:
        Apply common boilerplate patterns (default ``True``).
    custom_patterns:
        Optional extra regex patterns.

    Returns
    -------
    list[str]
        Cleaned strings in the same order as the input.
    """
    if RUST_AVAILABLE:
        return _rs.clean_texts(  # type: ignore[union-attr]
            texts,
            use_legal_patterns,
            use_common_patterns,
            custom_patterns,
        )
    custom_compiled = (
        [re.compile(p, re.IGNORECASE) for p in custom_patterns]
        if custom_patterns
        else []
    )
    return [_py_clean_one(t, use_legal_patterns, use_common_patterns, custom_compiled) for t in texts]


def list_turkish_patterns() -> List[str]:
    """Return the list of built-in Turkish legal boilerplate patterns."""
    if RUST_AVAILABLE:
        return _rs.list_turkish_patterns()  # type: ignore[union-attr]
    return list(_TURKISH_LEGAL_RAW)


def list_common_patterns() -> List[str]:
    """Return the list of built-in common boilerplate patterns."""
    if RUST_AVAILABLE:
        return _rs.list_common_patterns()  # type: ignore[union-attr]
    return list(_COMMON_RAW)


__all__ = [
    "RUST_AVAILABLE",
    "clean_text",
    "clean_texts",
    "list_turkish_patterns",
    "list_common_patterns",
]
