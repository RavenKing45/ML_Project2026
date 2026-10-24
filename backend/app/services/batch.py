"""Score an uploaded CSV of payments (``POST /api/v1/predict/batch``; master guide Parts 9 and 11).

Upload (:func:`read_upload`): one ``multipart/form-data`` field named ``file`` whose filename ends
in ``.csv``. The body is streamed through ``python-multipart`` and kept in memory; it is never
written to disk (Starlette's ``UploadFile`` would spool a file over 1 MB to a temporary file),
and reading stops as soon as the file exceeds ``MAX_UPLOAD_MB`` (413). A ``Content-Length`` that
is already too large is refused before anything is read.

CSV (:func:`parse_batch_csv`): UTF-8 text, comma-separated, with a header row. Required columns
are the model's manifest ``features`` (``V1..V28, Amount``); ``Time`` (unless the model uses it)
and unknown columns are ignored; an optional ``Class`` column (0/1) adds ``metrics``. More than
``MAX_BATCH_ROWS`` data rows is a 413. Every cell is checked, and the file is rejected with a 400
listing up to ``MAX_LISTED_ROWS`` row numbers when a cell is not a finite number (the text
``NaN`` included), an ``Amount`` lies outside [0, ``AMOUNT_MAX``] or a ``Class`` is not 0 or 1. An
EMPTY cell is a missing value: the pipeline's median imputer fills it, and ``warnings`` counts
them. The exception is an empty ``Amount``: that row is listed in ``errors`` and not scored,
because its amount could not be shown or added to ``flagged_amount_total``.

Scoring (:func:`score_batch`) is vectorised: one ``predict_proba`` call for every row. With
``Class``, ``metrics`` are :func:`fraudstudy.evaluation.compute_metrics` at the threshold used.
Row numbers are 0-based data rows (the header is not counted), and results keep input order.
File contents and filenames are never logged.
"""

import io
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from fastapi import Request
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import MultipartParser, parse_options_header

from backend.app.errors import STATUS_CODES, ApiError
from backend.app.services.artifacts import LoadedModel
from backend.app.services.inference import (
    DISCLAIMER,
    elapsed_ms,
    label_for,
    resolve_threshold,
    score_rows,
)
from fraudstudy.contracts import (
    AMOUNT_MAX,
    BatchResponse,
    BatchRowError,
    BatchRowResult,
    Metrics,
    ModelRef,
)
from fraudstudy.evaluation import compute_metrics

FILE_FIELD = "file"
CSV_SUFFIX = ".csv"
CSV_ENCODING = "utf-8-sig"  # UTF-8, with or without the byte-order mark Excel writes
MULTIPART_TYPE = b"multipart/form-data"
BYTES_PER_MB = 1024 * 1024
MULTIPART_OVERHEAD_BYTES = 64 * 1024  # boundaries and part headers allowed on top of the file
MAX_LISTED_ROWS = 5
MAX_LISTED_COLUMNS = 10
# CSV column names: the contract's Transaction fields plus the dataset's label.
AMOUNT_COLUMN = "Amount"
TIME_COLUMN = "Time"
CLASS_COLUMN = "Class"
CLASS_VALUES = (0.0, 1.0)
EMPTY_AMOUNT_MESSAGE = "Amount is empty, so this row was not scored; fill in its Amount."
BAD_REQUEST = STATUS_CODES[400]
TOO_LARGE = STATUS_CODES[413]


@dataclass(frozen=True)
class UploadedFile:
    """The uploaded CSV, held in memory.

    Attributes:
        filename: The client's filename (ends in ``.csv``); never logged.
        data: The file's bytes.
    """

    filename: str
    data: bytes


@dataclass(frozen=True)
class ParsedBatch:
    """A validated upload.

    Attributes:
        X: The model's input columns in manifest order, as floats; NaN marks an empty cell.
        y: ``Class`` as 0/1 integers, or ``None`` when the CSV has no ``Class`` column.
        empty_cells: Input column -> number of empty cells, for columns that have any.
        ignored_columns: Columns that are neither model inputs nor ``Time`` / ``Class``.
    """

    X: pd.DataFrame
    y: np.ndarray | None
    empty_cells: dict[str, int]
    ignored_columns: list[str]


def bad_request(message: str, details: object = None) -> ApiError:
    """A 400 ``bad_request`` :class:`ApiError`."""
    return ApiError(400, BAD_REQUEST, message, details)


def file_too_large(max_bytes: int) -> ApiError:
    """The 413 for an upload over ``max_bytes`` (``MAX_UPLOAD_MB``)."""
    return ApiError(
        413,
        TOO_LARGE,
        f"The file is larger than {max_bytes / BYTES_PER_MB:g} MB (MAX_UPLOAD_MB); split it into "
        "smaller files.",
        {"max_bytes": max_bytes},
    )


class UploadCollector:
    """``python-multipart`` callbacks that keep the ``file`` part in memory.

    Other parts are skipped. The filename is checked as soon as the part's headers end, and
    the size after every chunk.

    Args:
        max_bytes: Largest allowed file, in bytes.
    """

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.data = bytearray()
        self.filename: str | None = None
        self._in_file = False
        self._field = b""
        self._value = b""
        self._disposition = b""

    def callbacks(self) -> dict[str, Callable[..., None]]:
        """The callback mapping ``MultipartParser`` takes."""
        return {
            "on_part_begin": self.on_part_begin,
            "on_header_field": self.on_header_field,
            "on_header_value": self.on_header_value,
            "on_header_end": self.on_header_end,
            "on_headers_finished": self.on_headers_finished,
            "on_part_data": self.on_part_data,
            "on_part_end": self.on_part_end,
        }

    def on_part_begin(self) -> None:
        """Start a part: forget the previous part's headers."""
        self._in_file = False
        self._disposition = b""

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        """Collect (part of) a header name."""
        self._field += data[start:end]

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        """Collect (part of) a header value."""
        self._value += data[start:end]

    def on_header_end(self) -> None:
        """Keep ``Content-Disposition``; other part headers are not needed."""
        if self._field.strip().lower() == b"content-disposition":
            self._disposition = self._value
        self._field = self._value = b""

    def on_headers_finished(self) -> None:
        """Decide whether this part is the file and check its filename.

        Raises:
            ApiError: 400 for a second ``file`` part or a filename not ending in ``.csv``.
        """
        _, options = parse_options_header(self._disposition)
        if options.get(b"name") != FILE_FIELD.encode():
            return
        if self.filename is not None:
            raise bad_request(f"Send exactly one '{FILE_FIELD}' field.")
        filename = options.get(b"filename", b"").decode("utf-8", errors="replace").strip()
        if not filename.lower().endswith(CSV_SUFFIX):
            raise bad_request(
                f"Upload a CSV file: the '{FILE_FIELD}' field needs a filename ending in .csv."
            )
        self.filename = filename
        self._in_file = True

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        """Append file bytes (other parts are skipped).

        Raises:
            ApiError: 413 once the file exceeds ``max_bytes``.
        """
        if not self._in_file:
            return
        self.data += data[start:end]
        if len(self.data) > self.max_bytes:
            raise file_too_large(self.max_bytes)

    def on_part_end(self) -> None:
        """End a part."""
        self._in_file = False


async def read_upload(request: Request, max_bytes: int) -> UploadedFile:
    """Stream the multipart body and return the ``file`` field, in memory only.

    Reading stops at the first chunk that takes the file past ``max_bytes`` or the whole body
    past ``max_bytes + MULTIPART_OVERHEAD_BYTES``, so an oversized upload is never read in full.

    Args:
        request: The request (its body must not have been read).
        max_bytes: Largest allowed file (``MAX_UPLOAD_MB`` in bytes).

    Returns:
        The :class:`UploadedFile`.

    Raises:
        ApiError: 400 if the body is not ``multipart/form-data``, is malformed, or has no ``file``
            field / a non-``.csv`` filename; 413 if it is too large.
    """
    content_type, options = parse_options_header(request.headers.get("content-type"))
    boundary = options.get(b"boundary")
    if content_type.strip().lower() != MULTIPART_TYPE or not boundary:  # media types ignore case
        raise bad_request(
            f"Send the CSV as multipart/form-data in a field named '{FILE_FIELD}'."
        )
    max_body = max_bytes + MULTIPART_OVERHEAD_BYTES
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > max_body:
        raise file_too_large(max_bytes)
    collector = UploadCollector(max_bytes)
    parser = MultipartParser(boundary, collector.callbacks())
    received = 0
    try:
        async for chunk in request.stream():
            received += len(chunk)
            if received > max_body:
                raise file_too_large(max_bytes)
            parser.write(chunk)
        parser.finalize()
    except MultipartParseError as exc:
        raise bad_request(
            f"The upload is not valid multipart/form-data ({type(exc).__name__})."
        ) from exc
    if collector.filename is None:
        raise bad_request(f"No '{FILE_FIELD}' field: send the CSV in a field named '{FILE_FIELD}'.")
    return UploadedFile(collector.filename, bytes(collector.data))


def read_csv_text(data: bytes, max_rows: int) -> pd.DataFrame:
    """Read the CSV with every cell as text (no type guessing, no NA strings).

    The header row is read as data (``header=None``), so the header fixes the number of fields:
    a longer row is an error instead of pandas silently turning the extra leading fields into
    an index (which would shift every value one column, e.g. with a trailing comma per row).
    Header names are stripped of surrounding spaces. At most ``max_rows + 1`` data rows are
    parsed, enough to tell that a file has too many.

    Args:
        data: The file's bytes.
        max_rows: ``MAX_BATCH_ROWS``.

    Returns:
        The text table; cells missing from a short row are ``""``.

    Raises:
        ApiError: 400 if the bytes are not UTF-8 or not a readable CSV (including a row with more
            fields than the header), there are no data rows or two columns share a name; 413 if
            there are more than ``max_rows`` data rows.
    """
    try:
        table = pd.read_csv(
            io.BytesIO(data),
            header=None,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            nrows=max_rows + 2,  # the header row plus one data row over the limit
            encoding=CSV_ENCODING,
        ).fillna("")
    except UnicodeDecodeError as exc:
        raise bad_request("The file is not UTF-8 text; save it as 'CSV UTF-8'.") from exc
    except pd.errors.EmptyDataError as exc:
        raise bad_request("The file is empty; it needs a header row and data rows.") from exc
    except (pd.errors.ParserError, ValueError) as exc:
        raise bad_request(
            f"The file could not be read as CSV ({type(exc).__name__}); check that no row has "
            "more comma-separated fields than the header."
        ) from exc
    if len(table) - 1 > max_rows:
        raise ApiError(
            413,
            TOO_LARGE,
            f"The file has more than {max_rows:,} data rows (MAX_BATCH_ROWS); split it into "
            "smaller files.",
            {"max_rows": max_rows},
        )
    if len(table) < 2:
        raise bad_request("The file has a header but no data rows.")
    names = [str(name).strip() for name in table.iloc[0]]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise bad_request(
            f"Duplicate column name(s): {', '.join(duplicates)}; keep one of each.",
            {"duplicate_columns": duplicates},
        )
    frame = table.iloc[1:].reset_index(drop=True)
    frame.columns = names
    return frame


def to_numbers(text: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert one text column to floats.

    Args:
        text: Cells as strings (``""`` = empty).

    Returns:
        ``(values, empty, invalid)``: floats with NaN for empty and invalid cells; ``empty``
        marks blank cells; ``invalid`` marks non-blank cells that are not finite numbers.
    """
    stripped = text.str.strip()
    empty = (stripped == "").to_numpy()
    values = np.array(pd.to_numeric(stripped.mask(empty), errors="coerce"), dtype=float)  # a copy
    invalid = ~empty & ~np.isfinite(values)
    values[invalid] = np.nan
    return values, empty, invalid


def rows_error(problem: str, mask: np.ndarray, columns: list[str], fix: str) -> ApiError:
    """A 400 naming up to ``MAX_LISTED_ROWS`` offending rows (never their values).

    Args:
        problem: What is wrong, e.g. ``"Non-numeric values"``.
        mask: Offending rows (boolean, one per data row).
        columns: Columns with offending cells.
        fix: What to do about it.

    Returns:
        The :class:`ApiError`, with ``details = {"rows", "n_rows", "columns"}``.
    """
    rows = np.flatnonzero(mask)
    shown = [int(row) for row in rows[:MAX_LISTED_ROWS]]
    more = len(rows) - len(shown)
    listed = ", ".join(map(str, shown)) + (f" and {more} more" if more else "")
    return bad_request(
        f"{problem} in column(s) {', '.join(columns)} at rows {listed} (0-based data rows, header "
        f"not counted). {fix}",
        {"rows": shown, "n_rows": int(len(rows)), "columns": columns},
    )


def parse_batch_csv(data: bytes, features: list[str], max_rows: int) -> ParsedBatch:
    """Read and check an uploaded CSV (rules in the module docstring).

    Args:
        data: The file's bytes.
        features: The model's input columns (manifest order).
        max_rows: ``MAX_BATCH_ROWS``.

    Returns:
        The :class:`ParsedBatch`.

    Raises:
        ApiError: 400 or 413, see :func:`read_csv_text`, plus 400 for missing columns, invalid
            numbers, an out-of-range ``Amount`` or an invalid ``Class``.
    """
    frame = read_csv_text(data, max_rows)
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise bad_request(
            f"Missing column(s): {', '.join(missing)}. The CSV needs {', '.join(features)} "
            f"({TIME_COLUMN} and {CLASS_COLUMN} are optional).",
            {"missing_columns": missing},
        )
    values: dict[str, np.ndarray] = {}
    empty_cells: dict[str, int] = {}
    invalid = np.zeros(len(frame), dtype=bool)
    bad_columns: list[str] = []
    for column in features:
        values[column], empty, bad = to_numbers(frame[column])
        if empty.any():
            empty_cells[column] = int(empty.sum())
        if bad.any():
            invalid |= bad
            bad_columns.append(column)
    if bad_columns:
        raise rows_error(
            "Non-numeric or infinite values",
            invalid,
            bad_columns,
            "Use plain numbers, and leave a cell empty if its value is missing.",
        )
    amount = values.get(AMOUNT_COLUMN)
    if amount is not None:
        out_of_range = (amount < 0.0) | (amount > AMOUNT_MAX)  # NaN (empty) compares False
        if out_of_range.any():
            raise rows_error(
                f"Amount outside [0, {AMOUNT_MAX:,.0f}]",
                out_of_range,
                [AMOUNT_COLUMN],
                "Fix the amounts.",
            )
    y = None
    if CLASS_COLUMN in frame.columns:
        labels, _, _ = to_numbers(frame[CLASS_COLUMN])
        not_label = ~np.isin(labels, CLASS_VALUES)
        if not_label.any():
            raise rows_error(
                "Class values other than 0 or 1",
                not_label,
                [CLASS_COLUMN],
                "Use 0 (legitimate) or 1 (fraud) in every row, or remove the Class column.",
            )
        y = labels.astype(int)
    used = {*features, TIME_COLUMN, CLASS_COLUMN}
    ignored = [column for column in frame.columns if column not in used]
    return ParsedBatch(pd.DataFrame(values, columns=features), y, empty_cells, ignored)


def batch_warnings(
    parsed: ParsedBatch,
    n_unscored: int,
    y_scored: np.ndarray | None,
    dataset_fraud_rate: float | None,
) -> list[str]:
    """Notes for the response: imputed cells, unscored rows, ignored columns, metrics caveats.

    Args:
        parsed: The parsed upload.
        n_unscored: Rows not scored (empty ``Amount``).
        y_scored: ``Class`` of the scored rows, or ``None``.
        dataset_fraud_rate: ``fraud_rate`` from ``eda.json``, or ``None``.

    Returns:
        The warnings, in that order.
    """
    warnings: list[str] = []
    imputed = {c: n for c, n in parsed.empty_cells.items() if c != AMOUNT_COLUMN}
    if imputed:
        n_cells = sum(imputed.values())
        cells = "empty cell" if n_cells == 1 else "empty cells"
        warnings.append(
            f"{n_cells} {cells} (in {', '.join(imputed)}) filled with training medians by the "
            "model's preprocessing."
        )
    if n_unscored:
        rows = "row" if n_unscored == 1 else "rows"
        warnings.append(f"Not scored: {n_unscored} {rows} with an empty Amount (see errors).")
    if parsed.ignored_columns:
        shown = parsed.ignored_columns[:MAX_LISTED_COLUMNS]
        more = len(parsed.ignored_columns) - len(shown)
        warnings.append(
            f"Ignored columns the model does not use: {', '.join(shown)}"
            + (f" and {more} more." if more else ".")
        )
    if y_scored is not None:
        if len(np.unique(y_scored)) < len(CLASS_VALUES):
            warnings.append(
                "No metrics: the Class column needs both 0 and 1 among the scored rows."
            )
        else:
            share = f"{y_scored.mean():.2%} of its scored rows are fraud"
            if dataset_fraud_rate is not None:
                share += f", {dataset_fraud_rate:.2%} in the full dataset"
            warnings.append(
                f"Metrics describe this file only ({share}); precision depends on the fraud share."
            )
    return warnings


def score_batch(
    model: LoadedModel,
    upload: UploadedFile,
    requested_threshold: float | None,
    max_rows: int,
    started: float,
    dataset_fraud_rate: float | None = None,
) -> BatchResponse:
    """Parse, check and score an uploaded CSV with one model.

    Args:
        model: The model to score with.
        upload: The uploaded file.
        requested_threshold: The request's threshold, or ``None`` for the model's tuned one.
        max_rows: ``MAX_BATCH_ROWS``.
        started: ``time.perf_counter()`` when the request reached the endpoint.
        dataset_fraud_rate: ``eda.json``'s fraud rate (named in the metrics warning), or ``None``.

    Returns:
        The ``BatchResponse``.

    Raises:
        ApiError: From :func:`parse_batch_csv` (400 / 413).
    """
    entry = model.entry
    parsed = parse_batch_csv(upload.data, list(entry.features), max_rows)
    threshold, source = resolve_threshold(entry, requested_threshold)
    amount = parsed.X[AMOUNT_COLUMN].to_numpy(dtype=float)
    scored = ~np.isnan(amount)
    rows = np.flatnonzero(scored)
    if rows.size:
        scores, calibrated = score_rows(model, parsed.X.iloc[rows])
    else:
        scores, calibrated = np.zeros(0), None
    flagged = scores >= threshold
    y_scored = None if parsed.y is None else parsed.y[rows]
    results = [
        BatchRowResult(
            row=int(row),
            amount=float(amount[row]),
            risk_score=float(scores[i]),
            calibrated_probability=None if calibrated is None else float(calibrated[i]),
            label=label_for(float(scores[i]), threshold),
            true_class=None if parsed.y is None else int(parsed.y[row]),
        )
        for i, row in enumerate(rows)
    ]
    metrics: Metrics | None = None
    if y_scored is not None and len(np.unique(y_scored)) == len(CLASS_VALUES):
        metrics = compute_metrics(y_scored, scores, threshold)
    n_flagged = int(flagged.sum())
    return BatchResponse(
        model=ModelRef(id=entry.id, model=entry.model, technique=entry.technique),
        threshold=threshold,
        threshold_source=source,
        n_rows=len(parsed.X),
        n_scored=int(rows.size),
        n_flagged=n_flagged,
        flagged_fraction=n_flagged / rows.size if rows.size else 0.0,
        flagged_amount_total=float(amount[rows][flagged].sum()),
        metrics=metrics,
        results=results,
        errors=[
            BatchRowError(row=int(row), message=EMPTY_AMOUNT_MESSAGE)
            for row in np.flatnonzero(~scored)
        ],
        warnings=batch_warnings(parsed, int((~scored).sum()), y_scored, dataset_fraud_rate),
        disclaimer=DISCLAIMER,
        latency_ms=elapsed_ms(started),
    )
