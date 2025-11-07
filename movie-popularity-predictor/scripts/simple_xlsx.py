"""Minimal XLSX reader/writer without external dependencies."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence
from zipfile import ZipFile, ZIP_DEFLATED
import math
import pandas as pd
from pandas import DataFrame
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape


_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _excel_column_letter(index: int) -> str:
    """Convert a 1-based column index to an Excel-style column label."""

    if index <= 0:
        raise ValueError("Column indexes must be 1-based and positive")
    letters: List[str] = []
    while index:
        index, remainder = divmod(index - 1, 26)
        letters.append(chr(65 + remainder))
    return "".join(reversed(letters))


def _format_cell(row_idx: int, col_idx: int, value) -> str:
    ref = f"{_excel_column_letter(col_idx)}{row_idx}"
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return f'<c r="{ref}"/>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}"><v>{value}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def _sheet_xml(df: DataFrame) -> str:
    columns = [str(col) for col in df.columns]
    data_rows = list(df.itertuples(index=False, name=None))
    total_rows = len(data_rows) + 1 if columns else len(data_rows)
    if total_rows == 0:
        # Excel requires the dimension to be valid even for empty sheets.
        dimension = "A1:A1"
    else:
        max_col_letter = _excel_column_letter(max(len(columns), 1))
        dimension = f"A1:{max_col_letter}{max(total_rows, 1)}"

    rows_xml: List[str] = []
    if columns:
        header_cells = [
            _format_cell(1, col_idx + 1, header) for col_idx, header in enumerate(columns)
        ]
        rows_xml.append(f'<row r="1">{"".join(header_cells)}</row>')
        start_row = 2
    else:
        start_row = 1

    for offset, values in enumerate(data_rows):
        row_index = start_row + offset
        cells = [
            _format_cell(row_index, col_idx + 1, value)
            for col_idx, value in enumerate(values)
        ]
        rows_xml.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    sheet_body = "".join(rows_xml)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<worksheet xmlns="{_MAIN_NS}">'
        f'<dimension ref="{dimension}"/>'
        f'<sheetData>{sheet_body}</sheetData>'
        '</worksheet>'
    )


def _write_zip_entries(zf: ZipFile, entries: Sequence[tuple[str, str]]) -> None:
    for path, content in entries:
        zf.writestr(path, content)


def dataframe_to_xlsx(df: DataFrame, destination: Path) -> None:
    """Serialise ``df`` into a single-sheet XLSX file at ``destination``."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()

    sheet_xml = _sheet_xml(df)
    entries = [
        (
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
</Types>""",
        ),
        (
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>""",
        ),
        (
            "docProps/app.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Python</Application>
</Properties>""",
        ),
        (
            "docProps/core.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:creator>movie-popularity-predictor</dc:creator>
  <cp:lastModifiedBy>movie-popularity-predictor</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>""",
        ),
        (
            "xl/workbook.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Sheet1" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>""",
        ),
        (
            "xl/_rels/workbook.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>""",
        ),
        (
            "xl/styles.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>
  <fills count="1"><fill><patternFill patternType="none"/></fill></fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>""",
        ),
        ("xl/worksheets/sheet1.xml", sheet_xml),
    ]

    with ZipFile(destination, "w", ZIP_DEFLATED) as zf:
        _write_zip_entries(zf, entries)


def read_xlsx(path: Path) -> DataFrame:
    """Load the first worksheet from ``path`` into a DataFrame."""

    with ZipFile(path, "r") as zf:
        with zf.open("xl/worksheets/sheet1.xml") as sheet_file:
            tree = ET.parse(sheet_file)

    root = tree.getroot()
    ns = {"main": _MAIN_NS}
    sheet_data = root.find("main:sheetData", ns)
    if sheet_data is None:
        return pd.DataFrame()

    rows: List[List[str]] = []
    for row in sheet_data.findall("main:row", ns):
        cells: List[str] = []
        for cell in row.findall("main:c", ns):
            cell_type = cell.get("t")
            if cell_type == "inlineStr":
                text_elem = cell.find("main:is/main:t", ns)
                cells.append(text_elem.text if text_elem is not None else "")
            else:
                value_elem = cell.find("main:v", ns)
                cells.append(value_elem.text if value_elem is not None else "")
        rows.append(cells)

    if not rows:
        return pd.DataFrame()

    header = rows[0]
    data_rows = [
        row + [""] * (len(header) - len(row))
        for row in rows[1:]
    ]
    df = pd.DataFrame(data_rows, columns=header)
    return df.apply(pd.to_numeric, errors="ignore")


__all__ = ["dataframe_to_xlsx", "read_xlsx"]
