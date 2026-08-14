# Excel faylını oxuyur, agent üçün kontekst hazırlayır.

import os
from typing import Dict

import pandas as pd

UPLOAD_DIR = "data/uploads"


def save_uploaded_file(uploaded_file) -> str:
    """Streamlit UploadedFile-ı diskə yazır, yolunu qaytarır."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def load_excel(path: str) -> Dict[str, pd.DataFrame]:
    """Bütün sheet-ləri oxuyur. Sheet adları açar, DataFrame-lər dəyərlər kimi."""
    try:
        return pd.read_excel(path, sheet_name=None, engine="openpyxl")
    except Exception as e:
        raise ValueError(f"Excel oxunarkən xəta: {e}")


def _cardinality_notes(df: pd.DataFrame) -> str:
    """
    Hər sütun üçün unikal dəyər sayını sətir sayı ilə müqayisə edir.
    nunique == n_rows olan sütunlar çox güman ki SƏTİR/ƏMƏLİYYAT identifikatorudur
    (məs. hər sifarişə/sətrə fərqli bir kod), REAL təkrarlanan entity (müştəri,
    məhsul və s.) DEYİL — modelin "unikal müştəri sayı" kimi sualları səhv sütunla
    (məs. sətir ID-si ilə) cavablandırmasının qarşısını almaq üçün bu fərq
    açıq şəkildə göstərilir.

    Aşağı-kardinallıqlı (kateqoriya kimi görünən) mətn sütunları üçün isə real
    unikal dəyərlər siyahı ilə göstərilir — modelin string filter yazanda
    kateqoriya adının dəqiq yazılışını (böyük/kiçik hərf, diakritiklər: ə/ö/ü/ş/
    ç/ğ/ı və s.) TƏXMİN etməsinin qarşısını almaq üçün. Səhv yazılmış filter
    (məs. "Gözlemədə" əvəzinə "Gözləmədə") heç bir sətirlə uyğun gəlməyəcək və
    səssizcə 0/boş nəticə verəcək — bu, modelin ən çox buraxdığı görünməz xətadır.
    """
    n_rows = len(df)
    if n_rows == 0:
        return ""

    CATEGORY_MAX_UNIQUE = 20

    lines = ["\n[Sütun kardinallığı — unikal dəyər sayı / sətir sayı]"]
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        ratio_line = f"  - {col}: {nunique} unikal dəyər ({n_rows} sətirdən)"
        # Continuous numeric columns (price, amount, measurements...) naturally
        # have near-unique values per row - that's not an identity signal, so
        # only flag non-numeric (text/categorical/ID-like) columns to avoid
        # noisy false positives that would dilute the real warnings.
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        if nunique == n_rows and n_rows > 1 and not is_numeric:
            ratio_line += (
                "  ⚠️ HƏR SƏTİRDƏ FƏRQLİDİR → bu, sətir/əməliyyat "
                "identifikatoru ola bilər, təkrarlanan bir müştəri/məhsul/mağaza "
                "kimi ƏŞYANI (entity) təmsil etməyə bilər. 'Neçə unikal X var' "
                "tipli suallarda bunun əvəzinə həmin X-i təbii şəkildə "
                "təkrarlayan (aşağıda daha az unikal dəyəri olan) sütunu yoxla."
            )
        elif not is_numeric and 1 < nunique <= CATEGORY_MAX_UNIQUE:
            values = sorted(str(v) for v in df[col].dropna().unique())
            ratio_line += f"  → REAL dəyərlər (dəqiq yazılışı ilə, filter yazanda copy et): {values}"
        lines.append(ratio_line)
    return "\n".join(lines)


def build_context(sheets: Dict[str, pd.DataFrame]) -> str:
    """
    Agent üçün Excel haqqında tam kontekst mətn hazırlayır.
    Hər sheet üçün: ölçü, sütunlar, nümunə sətirlər, statistika, kardinallıq.
    """
    parts = []
    for name, df in sheets.items():
        lines = [f"=== SHEET: '{name}' ==="]
        lines.append(f"Ölçü: {df.shape[0]} sətir × {df.shape[1]} sütun")
        lines.append(f"Sütunlar: {list(df.columns)}")
        lines.append(f"Sütun tipləri:\n{df.dtypes.to_string()}")
        lines.append(_cardinality_notes(df))
        lines.append(f"\n[İlk 5 sətir]\n{df.head(5).to_markdown(index=False)}")

        num_df = df.select_dtypes(include="number")
        if not num_df.empty:
            lines.append(f"\n[Ədədi statistika]\n{num_df.describe().round(2).to_markdown()}")

        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def get_dataframe_summary(sheets: Dict[str, pd.DataFrame]) -> str:
    """Sidebar üçün qısa xülasə."""
    return "\n".join(
        f"**{name}**: {df.shape[0]} sətir, {df.shape[1]} sütun"
        for name, df in sheets.items()
    )