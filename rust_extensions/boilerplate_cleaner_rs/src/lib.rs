use pyo3::prelude::*;
use regex::{Regex, RegexSet, RegexSetBuilder};
use rayon::prelude::*;

// ──────────────────────────────────────────────────────────────
//  Pattern lists (mirrors boilerplate_cleaner.py patterns)
// ──────────────────────────────────────────────────────────────

/// All Turkish-legal boilerplate patterns, compiled into one RegexSet.
static TURKISH_LEGAL_RAW: &[&str] = &[
    // Document headers / footers
    r"(?i)TÜRKİYE\s+CUMHURİYETİ[^\n]*",
    r"(?i)T\.C\.\s*[^\n]*",
    r"(?i)Sayfa\s+\d+\s*/\s*\d+[^\n]*",
    r"(?i)Sayfa\s+\d+[^\n]*",
    r"(?i)Belge\s+No:\s*[^\n]*",
    r"(?i)Referans\s+No:\s*[^\n]*",
    r"(?i)Dosya\s+No:\s*[^\n]*",
    r"(?i)Kayıt\s+No:\s*[^\n]*",
    // Contract boilerplate
    r"(?i)Bu\s+sözleşme\s+taraflar\s+arasında\s+imzalanmıştır[^\n]*",
    r"(?i)Taraflar\s+arasında\s+imzalanan\s+sözleşme[^\n]*",
    r"(?i)Mahkeme\s+kararı\s+kesinleşmiştir[^\n]*",
    r"(?i)Yargıtay\s+kararı\s+kesinleşmiştir[^\n]*",
    r"(?i)Danıştay\s+kararı\s+kesinleşmiştir[^\n]*",
    r"(?i)Anayasa\s+Mahkemesi\s+kararı[^\n]*",
    r"(?i)Bu\s+karar\s+kesinleşmiştir[^\n]*",
    r"(?i)Karar\s+kesinleşmiştir[^\n]*",
    // Legal metadata
    r"(?i)Tarih:\s*\d{1,2}\.\d{1,2}\.\d{4}[^\n]*",
    r"(?i)Tarih\s+ve\s+saat:\s*[^\n]*",
    r"(?i)İmza\s+tarihi:\s*[^\n]*",
    r"(?i)Yürürlük\s+tarihi:\s*[^\n]*",
    r"(?i)Yayın\s+tarihi:\s*[^\n]*",
    // Document structure
    r"(?i)Madde\s+\d+[^\n]*",
    r"(?i)Bölüm\s+\d+[^\n]*",
    r"(?i)Kısım\s+\d+[^\n]*",
    r"(?i)Fıkra\s+\d+[^\n]*",
    r"(?i)Bent\s+\d+[^\n]*",
    // Boilerplate phrases
    r"(?i)Yukarıda\s+adı\s+ve\s+soyadı\s+yazılı[^\n]*",
    r"(?i)Taraflar\s+arasında\s+anlaşma\s+sağlanmıştır[^\n]*",
    r"(?i)Bu\s+belge\s+ile\s+ilgili\s+[^\n]*",
    r"(?i)Belgenin\s+devamı\s+[^\n]*",
    r"(?i)Ek\s+\d+[^\n]*",
    r"(?i)Ekler\s+[^\n]*",
    // Court/institution
    r"(?i)Mahkeme\s+adı:\s*[^\n]*",
    r"(?i)Mahkeme\s+merkezi:\s*[^\n]*",
    r"(?i)Duruşma\s+tarihi:\s*[^\n]*",
    r"(?i)Duruşma\s+saati:\s*[^\n]*",
    r"(?i)Duruşma\s+yeri:\s*[^\n]*",
    // Classification
    r"(?i)Dava\s+türü:\s*[^\n]*",
    r"(?i)Dava\s+konusu:\s*[^\n]*",
    r"(?i)Talep\s+edilen:\s*[^\n]*",
    r"(?i)Talep\s+eden:\s*[^\n]*",
    r"(?i)Davalı:\s*[^\n]*",
    // Status
    r"(?i)Durum:\s*[^\n]*",
    r"(?i)Sonuç:\s*[^\n]*",
    r"(?i)Karar:\s*[^\n]*",
    r"(?i)Hüküm:\s*[^\n]*",
    // Section headings
    r"(?i)GİRİŞ[^\n]*",
    r"(?i)SONUÇ[^\n]*",
    r"(?i)KARAR[^\n]*",
    r"(?i)HÜKÜM[^\n]*",
    r"(?i)GEREKÇE[^\n]*",
    r"(?i)MADDELER[^\n]*",
    r"(?i)EK[^\n]*",
    // Boilerplate endings
    r"(?i)Bu\s+karar\s+ile\s+ilgili\s+[^\n]*",
    r"(?i)Karar\s+bu\s+şekilde\s+verilmiştir[^\n]*",
    r"(?i)Hüküm\s+bu\s+şekilde\s+verilmiştir[^\n]*",
    r"(?i)Taraflar\s+bilgilendirilmiştir[^\n]*",
    r"(?i)İtiraz\s+hakkı\s+saklıdır[^\n]*",
    r"(?i)Yasal\s+süre\s+içinde\s+[^\n]*",
    // Additional patterns
    r"(?i)15/\d{2}/\d{4}\s+tarihinde\s+kesin\s+olarak,\s+oyçokluğuyla\s+karar\s+verildi\.",
    r"(?i)\(X\)\s+KARŞI\s+OY\s*:\s*[^\n]*",
    r"(?i)KARAR\s+SONUCU\s*:\s*[^\n]*",
    r"(?i)kesin\s+olarak,\s+oyçokluğuyla\s+karar\s+verildi\.",
    r"(?i)oyçokluğuyla\s+karar\s+verildi\.",
    r"(?i)karar\s+verildi\.",
];

static COMMON_RAW: &[&str] = &[
    // English
    r"(?i)Copyright[^\n]*",
    r"(?i)All\s+rights\s+reserved[^\n]*",
    r"(?i)Confidential[^\n]*",
    r"(?i)Proprietary[^\n]*",
    r"(?i)Page\s+\d+\s+of\s+\d+[^\n]*",
    r"(?i)Generated\s+on[^\n]*",
    r"(?i)Last\s+updated[^\n]*",
    r"(?i)Version\s+\d+\.\d+[^\n]*",
    r"(?i)Document\s+ID:[^\n]*",
    r"(?i)Reference:[^\n]*",
    // Turkish equivalents
    r"(?i)Telif\s+hakkı[^\n]*",
    r"(?i)Tüm\s+haklar\s+saklıdır[^\n]*",
    r"(?i)Gizli[^\n]*",
    r"(?i)Özel[^\n]*",
    r"(?i)Oluşturulma\s+tarihi[^\n]*",
    r"(?i)Son\s+güncelleme[^\n]*",
    r"(?i)Sürüm\s+\d+\.\d+[^\n]*",
    r"(?i)Belge\s+kimliği[^\n]*",
    r"(?i)Referans[^\n]*",
];

// ──────────────────────────────────────────────────────────────
//  Compiled pattern sets (built once, reused across calls)
// ──────────────────────────────────────────────────────────────

struct PatternSets {
    turkish_set: RegexSet,
    turkish_regexes: Vec<Regex>,
    common_set: RegexSet,
    common_regexes: Vec<Regex>,
    whitespace_re: Regex,
}

impl PatternSets {
    fn build() -> Self {
        let turkish_set = RegexSetBuilder::new(TURKISH_LEGAL_RAW)
            .build()
            .expect("Turkish legal RegexSet should compile");
        let turkish_regexes: Vec<Regex> = TURKISH_LEGAL_RAW
            .iter()
            .map(|p| Regex::new(p).expect("pattern compiles"))
            .collect();

        let common_set = RegexSetBuilder::new(COMMON_RAW)
            .build()
            .expect("Common RegexSet should compile");
        let common_regexes: Vec<Regex> = COMMON_RAW
            .iter()
            .map(|p| Regex::new(p).expect("pattern compiles"))
            .collect();

        let whitespace_re = Regex::new(r"\s{2,}").expect("whitespace pattern compiles");

        PatternSets {
            turkish_set,
            turkish_regexes,
            common_set,
            common_regexes,
            whitespace_re,
        }
    }

    /// Remove all matches for every pattern in a set from `text`.
    fn apply_set<'a>(&self, text: &'a str, set: &RegexSet, regexes: &[Regex]) -> String {
        // Fast path: if no patterns match at all, skip the heavy work.
        if set.matches(text).matched_any() {
            let mut result = text.to_owned();
            for re in regexes {
                if re.is_match(&result) {
                    result = re.replace_all(&result, "").into_owned();
                }
            }
            result
        } else {
            text.to_owned()
        }
    }
}

// Thread-local so we pay the build cost only once per thread.
thread_local! {
    static PATTERNS: PatternSets = PatternSets::build();
}

// ──────────────────────────────────────────────────────────────
//  Core cleaning logic
// ──────────────────────────────────────────────────────────────

/// Clean a single string.
fn clean_one(
    text: &str,
    use_legal: bool,
    use_common: bool,
    custom_regexes: &[Regex],
) -> String {
    PATTERNS.with(|p| {
        let mut result = text.to_owned();

        if use_legal {
            result = p.apply_set(&result, &p.turkish_set, &p.turkish_regexes);
        }
        if use_common {
            result = p.apply_set(&result, &p.common_set, &p.common_regexes);
        }

        // Apply caller-supplied custom patterns.
        for re in custom_regexes {
            if re.is_match(&result) {
                result = re.replace_all(&result, "").into_owned();
            }
        }

        // Collapse multiple whitespace runs to a single space and trim.
        result = p
            .whitespace_re
            .replace_all(&result, " ")
            .trim()
            .to_owned();

        result
    })
}

// ──────────────────────────────────────────────────────────────
//  PyO3 bindings
// ──────────────────────────────────────────────────────────────

/// Clean a single text string.
///
/// Parameters
/// ----------
/// text : str
///     The input text to clean.
/// use_legal_patterns : bool, optional
///     Whether to apply Turkish legal boilerplate patterns (default True).
/// use_common_patterns : bool, optional
///     Whether to apply common (English + Turkish) boilerplate patterns (default True).
/// custom_patterns : list[str], optional
///     Additional regex patterns to strip (compiled with IGNORECASE).
///
/// Returns
/// -------
/// str
///     The cleaned text.
#[pyfunction]
#[pyo3(signature = (text, use_legal_patterns=true, use_common_patterns=true, custom_patterns=None))]
fn clean_text(
    text: &str,
    use_legal_patterns: bool,
    use_common_patterns: bool,
    custom_patterns: Option<Vec<String>>,
) -> PyResult<String> {
    let custom_regexes = compile_custom(custom_patterns)?;
    Ok(clean_one(text, use_legal_patterns, use_common_patterns, &custom_regexes))
}

/// Clean a list of text strings in parallel (Rayon).
///
/// Parameters
/// ----------
/// texts : list[str]
///     The input strings to clean.
/// use_legal_patterns : bool, optional
///     Apply Turkish legal patterns (default True).
/// use_common_patterns : bool, optional
///     Apply common boilerplate patterns (default True).
/// custom_patterns : list[str], optional
///     Extra regex patterns.
///
/// Returns
/// -------
/// list[str]
///     Cleaned strings in the same order as the input.
#[pyfunction]
#[pyo3(signature = (texts, use_legal_patterns=true, use_common_patterns=true, custom_patterns=None))]
fn clean_texts(
    texts: Vec<String>,
    use_legal_patterns: bool,
    use_common_patterns: bool,
    custom_patterns: Option<Vec<String>>,
) -> PyResult<Vec<String>> {
    let custom_regexes = compile_custom(custom_patterns)?;
    let results: Vec<String> = texts
        .par_iter()
        .map(|t| clean_one(t, use_legal_patterns, use_common_patterns, &custom_regexes))
        .collect();
    Ok(results)
}

/// Return the list of all built-in Turkish legal patterns.
#[pyfunction]
fn list_turkish_patterns() -> Vec<String> {
    TURKISH_LEGAL_RAW.iter().map(|s| s.to_string()).collect()
}

/// Return the list of all built-in common boilerplate patterns.
#[pyfunction]
fn list_common_patterns() -> Vec<String> {
    COMMON_RAW.iter().map(|s| s.to_string()).collect()
}

// ──────────────────────────────────────────────────────────────
//  Helpers
// ──────────────────────────────────────────────────────────────

fn compile_custom(patterns: Option<Vec<String>>) -> PyResult<Vec<Regex>> {
    match patterns {
        None => Ok(vec![]),
        Some(pats) => pats
            .iter()
            .map(|p| {
                Regex::new(p).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Invalid regex pattern '{}': {}",
                        p, e
                    ))
                })
            })
            .collect(),
    }
}

// ──────────────────────────────────────────────────────────────
//  Module definition
// ──────────────────────────────────────────────────────────────

#[pymodule]
fn boilerplate_cleaner_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(clean_text, m)?)?;
    m.add_function(wrap_pyfunction!(clean_texts, m)?)?;
    m.add_function(wrap_pyfunction!(list_turkish_patterns, m)?)?;
    m.add_function(wrap_pyfunction!(list_common_patterns, m)?)?;
    Ok(())
}
