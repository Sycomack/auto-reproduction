from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..llm import VisionClient
from ..task import TaskSpec, VisualInput


def _fitz():
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError(
            "Visual PDF preparation requires PyMuPDF. Install the project with: "
            "pip install -e ."
        ) from exc
    return pymupdf


def _slug(value: str) -> str:
    rendered = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return rendered or "paper-visual"


def _normalized_rect(rect: Any, page_rect: Any) -> list[float]:
    return [
        round(float(rect.x0 / page_rect.width), 6),
        round(float(rect.y0 / page_rect.height), 6),
        round(float(rect.x1 / page_rect.width), 6),
        round(float(rect.y1 / page_rect.height), 6),
    ]


def _valid_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(1.0, x1), min(1.0, y1)
    if x1 - x0 < 0.02 or y1 - y0 < 0.02:
        return None
    return [round(x0, 6), round(y0, 6), round(x1, 6), round(y1, 6)]


def _locator_bbox(data: dict[str, Any] | None) -> list[float] | None:
    if not data or data.get("found") is False:
        return None
    return _valid_bbox(data.get("bbox"))


def _compose_bbox(parent: list[float], child: list[float]) -> list[float]:
    """Convert crop-relative child coordinates back to normalized page space."""
    px0, py0, px1, py1 = parent
    cx0, cy0, cx1, cy1 = child
    width = px1 - px0
    height = py1 - py0
    return [
        round(px0 + cx0 * width, 6),
        round(py0 + cy0 * height, 6),
        round(px0 + cx1 * width, 6),
        round(py0 + cy1 * height, 6),
    ]


def _response_text(response: dict[str, Any]) -> str:
    content = response.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return ""


def _parse_json_response(response: dict[str, Any]) -> dict[str, Any] | None:
    text = _response_text(response)
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    if not fenced:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _find_caption(
    pdf_path: Path, figure_label: str, requested_page: int | None
) -> tuple[int, list[float] | None]:
    fitz = _fitz()
    with fitz.open(pdf_path) as document:
        if requested_page is not None:
            if requested_page > document.page_count:
                raise ValueError(
                    f"Requested page {requested_page} exceeds PDF page count "
                    f"{document.page_count}"
                )
            page_indexes = [requested_page - 1]
        else:
            page_indexes = range(document.page_count)
        for page_index in page_indexes:
            page = document[page_index]
            matches = page.search_for(figure_label)
            if matches:
                return page_index + 1, _normalized_rect(matches[0], page.rect)
        if requested_page is not None:
            return requested_page, None
    raise ValueError(f"Could not locate caption '{figure_label}' in the paper")


def render_pdf_region(
    pdf_path: Path,
    page_number: int,
    output_path: Path,
    bbox: list[float] | None = None,
    dpi: int = 220,
) -> dict[str, Any]:
    fitz = _fitz()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(pdf_path) as document:
        if page_number < 1 or page_number > document.page_count:
            raise ValueError(f"PDF page is out of range: {page_number}")
        page = document[page_number - 1]
        clip = None
        if bbox is not None:
            valid_bbox = _valid_bbox(bbox)
            if valid_bbox is None:
                raise ValueError(f"Invalid normalized crop bbox: {bbox}")
            x0, y0, x1, y1 = valid_bbox
            clip = fitz.Rect(
                x0 * page.rect.width,
                y0 * page.rect.height,
                x1 * page.rect.width,
                y1 * page.rect.height,
            )
        scale = dpi / 72.0
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False
        )
        pixmap.save(output_path)
    return {
        "width": pixmap.width,
        "height": pixmap.height,
        "dpi": dpi,
    }


def _caption_fallback_bbox(caption_bbox: list[float] | None) -> list[float]:
    if caption_bbox is None:
        return [0.0, 0.0, 1.0, 1.0]
    _, caption_y0, _, caption_y1 = caption_bbox
    return [
        0.02,
        round(max(0.0, caption_y0 - 0.58), 6),
        0.98,
        round(min(1.0, caption_y1 + 0.04), 6),
    ]


def _figure_localization_prompt(visual: VisualInput) -> str:
    return (
        f"Locate the complete figure {visual.figure_label!r} on this rendered "
        "paper page. Return only "
        "a JSON object with found, bbox, caption_bbox, and confidence. bbox must "
        "cover all panels but exclude surrounding body text and, when possible, the "
        "caption. Coordinates must be normalized [x0, y0, x1, y1] values between "
        "0 and 1 relative to the full page. Do not transcribe or analyze the figure "
        "in this localization step."
    )


def _focus_localization_prompt(visual: VisualInput) -> str:
    return (
        f"Within this crop of {visual.figure_label!r}, locate only the target "
        f"subfigure {visual.focus!r}. Return only a JSON object with found, bbox, "
        "and confidence. bbox must tightly include the target panel title, axes, "
        "tick labels, plotted data, and any legend needed to interpret that panel, "
        "while excluding neighboring panels and the overall figure caption. "
        "Coordinates must be normalized [x0, y0, x1, y1] values between 0 and 1 "
        "relative to this supplied figure crop. Do not transcribe values in this "
        "localization step."
    )


def _analysis_prompt(
    visual: VisualInput,
    extra_prompt: str = "",
    declared_reference: dict[str, list[dict[str, float]]] | None = None,
) -> str:
    purpose = visual.purpose or "evidence for the supplied reproduction claim"
    focus = f" Focus only on {visual.focus}." if visual.focus else ""
    suffix = f" Additional request: {extra_prompt}" if extra_prompt else ""
    coordinate_constraint = ""
    if declared_reference:
        coordinates = {
            name: [point["x"] for point in points]
            for name, points in declared_reference.items()
        }
        coordinate_constraint = (
            " The task declares these experimental series and x coordinates: "
            f"{json.dumps(coordinates, ensure_ascii=True)}. These coordinates are "
            "experimental conditions, not measured y values. Use the exact declared "
            "series names and return only visible data markers at these coordinates. "
            "Do not report axis ticks such as 80 or 40 unless they are explicitly "
            "listed as experimental coordinates. A declared one-point series may "
            "represent a visible horizontal baseline; report that baseline once at "
            "its declared coordinate even when the line has no marker. Never infer "
            "a y value from the declared coordinates."
        )
    return (
        f"Analyze {visual.figure_label!r} as {purpose}.{focus} Return only one JSON "
        "object with figure_label, focus, caption, panel_count, panels, "
        "qualitative_findings, and uncertainties. Each panels item must contain "
        "panel_title, dataset, model, metric, x_axis, y_axis, and series. Each series "
        "item must contain name and points; every point must be an object with x, y, "
        "and uncertainty. Transcribe concrete numeric points whenever they are "
        "legible, while clearly marking visually estimated values. Preserve axis "
        "units and legend names. Use null and explain the uncertainty when a value "
        "cannot be read. Never infer missing values from prior knowledge."
        f"{coordinate_constraint}{suffix}"
    )


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _declared_chart_reference(
    task: TaskSpec,
) -> tuple[dict[str, list[dict[str, float]]], float | None]:
    protocol = task.raw.get("reproduction_protocol")
    if not isinstance(protocol, dict):
        return {}, None
    reported = protocol.get("reported_results")
    if not isinstance(reported, dict):
        return {}, None
    raw_series = reported.get("series")
    if not isinstance(raw_series, dict):
        return {}, None

    reference: dict[str, list[dict[str, float]]] = {}
    for raw_name, raw_points in raw_series.items():
        if not isinstance(raw_points, list):
            continue
        points: list[dict[str, float]] = []
        for raw_point in raw_points:
            if not isinstance(raw_point, dict):
                continue
            x_key = next(
                (
                    key
                    for key in ("x", "budget_percent")
                    if key in raw_point and _numeric(raw_point[key]) is not None
                ),
                None,
            )
            if x_key is None:
                x_key = next(
                    (
                        key
                        for key, value in raw_point.items()
                        if key.endswith("_percent") and _numeric(value) is not None
                    ),
                    None,
                )
            if x_key is None:
                continue
            y_candidates = [
                _numeric(value)
                for key, value in raw_point.items()
                if key != x_key and key not in {"id", "uncertainty"}
            ]
            y_values = [value for value in y_candidates if value is not None]
            point = {"x": float(raw_point[x_key])}
            if len(y_values) == 1:
                point["y"] = y_values[0]
            points.append(point)
        if points:
            reference[str(raw_name)] = points

    tolerance = _numeric(reported.get("absolute_digitization_tolerance"))
    return reference, tolerance


def _series_key(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _validate_analysis(
    analysis: dict[str, Any],
    visual: VisualInput,
    declared_reference: dict[str, list[dict[str, float]]] | None,
    absolute_tolerance: float | None,
) -> dict[str, Any]:
    errors: list[str] = []
    panels = analysis.get("panels")
    if not isinstance(panels, list):
        panels = []
        errors.append("analysis.panels must be a list")
    if visual.focus:
        panel_count = _numeric(analysis.get("panel_count"))
        if panel_count != 1 or len(panels) != 1:
            errors.append(
                "Focused visual analysis must contain exactly one target panel"
            )

    validation: dict[str, Any] = {
        "status": "passed",
        "errors": errors,
        "series": [],
    }
    if not declared_reference:
        validation["status"] = "failed" if errors else "passed"
        return validation

    observed_series: dict[str, dict[str, Any]] = {}
    for panel in panels:
        if not isinstance(panel, dict) or not isinstance(panel.get("series"), list):
            continue
        for series in panel["series"]:
            if isinstance(series, dict) and series.get("name"):
                observed_series.setdefault(_series_key(series["name"]), series)

    for expected_name, expected_points in declared_reference.items():
        expected_by_x = {float(point["x"]): point for point in expected_points}
        series = observed_series.get(_series_key(expected_name))
        result: dict[str, Any] = {
            "name": expected_name,
            "expected_x": sorted(expected_by_x),
            "observed_x": [],
            "missing_x": [],
            "unexpected_x": [],
            "point_comparisons": [],
        }
        if series is None:
            result["missing_x"] = sorted(expected_by_x)
            errors.append(f"Missing declared series: {expected_name}")
            validation["series"].append(result)
            continue

        raw_points = series.get("points")
        if not isinstance(raw_points, list):
            raw_points = []
        observed_by_x: dict[float, dict[str, Any]] = {}
        for raw_point in raw_points:
            if not isinstance(raw_point, dict):
                continue
            x_value = _numeric(raw_point.get("x"))
            if x_value is None:
                continue
            if x_value in observed_by_x:
                errors.append(
                    f"Duplicate x coordinate {x_value:g} in series {expected_name}"
                )
            observed_by_x[x_value] = raw_point

        result["observed_x"] = sorted(observed_by_x)
        result["missing_x"] = sorted(set(expected_by_x) - set(observed_by_x))
        result["unexpected_x"] = sorted(set(observed_by_x) - set(expected_by_x))
        if result["missing_x"]:
            errors.append(
                f"Series {expected_name} is missing x coordinates: "
                f"{result['missing_x']}"
            )
        if result["unexpected_x"]:
            errors.append(
                f"Series {expected_name} contains undeclared x coordinates: "
                f"{result['unexpected_x']}"
            )

        for x_value, expected_point in sorted(expected_by_x.items()):
            observed_point = observed_by_x.get(x_value)
            if observed_point is None or "y" not in expected_point:
                continue
            observed_y = _numeric(observed_point.get("y"))
            expected_y = expected_point["y"]
            absolute_error = (
                None if observed_y is None else abs(observed_y - expected_y)
            )
            within_tolerance = (
                None
                if absolute_error is None or absolute_tolerance is None
                else absolute_error <= absolute_tolerance
            )
            result["point_comparisons"].append(
                {
                    "x": x_value,
                    "expected_y": expected_y,
                    "observed_y": observed_y,
                    "absolute_error": absolute_error,
                    "within_tolerance": within_tolerance,
                }
            )
            if observed_y is None:
                errors.append(
                    f"Series {expected_name} has no numeric y value at x={x_value:g}"
                )
            elif within_tolerance is False:
                errors.append(
                    f"Series {expected_name} at x={x_value:g} exceeds the visual "
                    f"reference tolerance"
                )
        validation["series"].append(result)

    validation["status"] = "failed" if errors else "passed"
    validation["absolute_tolerance"] = absolute_tolerance
    return validation


def analyze_visual_reference(
    pdf_path: Path,
    visual: VisualInput,
    assets_dir: Path,
    relative_prefix: str,
    vision_client: VisionClient | None,
    extra_prompt: str = "",
    declared_reference: dict[str, list[dict[str, float]]] | None = None,
    absolute_tolerance: float | None = None,
) -> dict[str, Any]:
    page_number, caption_bbox = _find_caption(
        pdf_path, visual.figure_label, visual.page
    )
    stem = _slug(visual.visual_id)
    page_path = assets_dir / f"{stem}-page.png"
    figure_path = assets_dir / f"{stem}-figure.png"
    focus_path = assets_dir / f"{stem}.png"
    page_render = render_pdf_region(pdf_path, page_number, page_path, dpi=140)

    figure_bbox = None
    localization_method = "caption_heuristic"
    locator_response: dict[str, Any] | None = None
    locator_data: dict[str, Any] | None = None
    locator_error = ""
    if vision_client is not None:
        try:
            locator_response = vision_client.analyze(
                page_path, _figure_localization_prompt(visual), detail="original"
            )
            locator_data = _parse_json_response(locator_response)
            figure_bbox = _locator_bbox(locator_data)
            if figure_bbox is not None:
                localization_method = "vision_bbox"
        except Exception as exc:
            locator_error = f"{type(exc).__name__}: {exc}"
    if figure_bbox is None:
        figure_bbox = _caption_fallback_bbox(caption_bbox)

    figure_render = render_pdf_region(
        pdf_path, page_number, figure_path, bbox=figure_bbox, dpi=260
    )
    analysis_path = figure_path
    focus_bbox_in_figure: list[float] | None = None
    focus_bbox: list[float] | None = None
    focus_render: dict[str, Any] | None = None
    focus_locator_response: dict[str, Any] | None = None
    focus_locator_data: dict[str, Any] | None = None
    focus_locator_error = ""
    focus_localization_method = "not_requested"
    if visual.focus:
        focus_localization_method = "not_available"
        if vision_client is not None:
            try:
                focus_locator_response = vision_client.analyze(
                    figure_path,
                    _focus_localization_prompt(visual),
                    detail="original",
                )
                focus_locator_data = _parse_json_response(focus_locator_response)
                focus_bbox_in_figure = _locator_bbox(focus_locator_data)
                if focus_bbox_in_figure is None:
                    focus_locator_error = (
                        "Vision model did not return a valid target-panel bbox"
                    )
                else:
                    focus_bbox = _compose_bbox(figure_bbox, focus_bbox_in_figure)
                    focus_render = render_pdf_region(
                        pdf_path,
                        page_number,
                        focus_path,
                        bbox=focus_bbox,
                        dpi=420,
                    )
                    analysis_path = focus_path
                    focus_localization_method = "vision_bbox"
            except Exception as exc:
                focus_locator_error = f"{type(exc).__name__}: {exc}"

    analysis_response: dict[str, Any] | None = None
    analysis: dict[str, Any] | None = None
    analysis_validation: dict[str, Any] | None = None
    analysis_error = ""
    focus_ready = not visual.focus or focus_bbox is not None
    if vision_client is not None and focus_ready:
        try:
            analysis_response = vision_client.analyze(
                analysis_path,
                _analysis_prompt(
                    visual,
                    extra_prompt,
                    declared_reference=declared_reference,
                ),
                detail="original",
            )
            analysis = _parse_json_response(analysis_response)
            if not _response_text(analysis_response):
                analysis_error = "Vision model returned empty content"
            if analysis is not None:
                analysis_validation = _validate_analysis(
                    analysis,
                    visual,
                    declared_reference,
                    absolute_tolerance,
                )
        except Exception as exc:
            analysis_error = f"{type(exc).__name__}: {exc}"

    if visual.focus and not focus_ready:
        status = "focus_localization_failed"
    elif analysis is not None and analysis_validation is not None:
        status = (
            "analyzed"
            if analysis_validation["status"] == "passed"
            else "analysis_invalid"
        )
    elif analysis is not None:
        status = "analyzed"
    elif analysis_response is not None and _response_text(analysis_response):
        status = "analyzed_unstructured"
    elif analysis_error:
        status = "analysis_failed"
    else:
        status = "rendered_only"

    record: dict[str, Any] = {
        "id": visual.visual_id,
        "figure_label": visual.figure_label,
        "purpose": visual.purpose,
        "focus": visual.focus,
        "status": status,
        "page": page_number,
        "bbox": figure_bbox,
        "focus_bbox": focus_bbox,
        "focus_bbox_within_figure": focus_bbox_in_figure,
        "caption_bbox": caption_bbox,
        "localization_method": localization_method,
        "focus_localization_method": focus_localization_method,
        "assets": {
            "page_preview": f"{relative_prefix}/{page_path.name}",
            "figure_crop": f"{relative_prefix}/{figure_path.name}",
        },
        "render": {"page_preview": page_render, "figure_crop": figure_render},
    }
    if focus_render is not None:
        record["assets"]["focus_crop"] = f"{relative_prefix}/{focus_path.name}"
        record["render"]["focus_crop"] = focus_render
    if locator_response is not None:
        record["locator"] = {
            "model": locator_response.get("model", ""),
            "parsed": locator_data,
            "raw": _response_text(locator_response),
            "usage": locator_response.get("usage", {}),
        }
    if locator_error:
        record["locator_error"] = locator_error
    if focus_locator_response is not None:
        record["focus_locator"] = {
            "model": focus_locator_response.get("model", ""),
            "parsed": focus_locator_data,
            "raw": _response_text(focus_locator_response),
            "usage": focus_locator_response.get("usage", {}),
        }
    if focus_locator_error:
        record["focus_locator_error"] = focus_locator_error
    if analysis_response is not None:
        record["analysis"] = analysis or {"raw": _response_text(analysis_response)}
        record["analysis_model"] = analysis_response.get("model", "")
        record["analysis_usage"] = analysis_response.get("usage", {})
    if analysis_validation is not None:
        record["analysis_validation"] = analysis_validation
    if analysis_error:
        record["analysis_error"] = analysis_error
    return record


def prepare_paper_evidence(
    task: TaskSpec,
    pdf_path: Path,
    workspace_dir: Path,
    vision_client: VisionClient | None = None,
) -> Path:
    evidence_path = workspace_dir / "paper_evidence.json"
    assets_dir = workspace_dir / "paper_assets"
    records = []
    declared_reference, absolute_tolerance = (
        _declared_chart_reference(task)
        if len(task.visual_inputs) == 1
        else ({}, None)
    )
    if task.visual_inputs:
        assets_dir.mkdir(parents=True, exist_ok=True)
    for visual in task.visual_inputs:
        try:
            records.append(
                analyze_visual_reference(
                    pdf_path=pdf_path,
                    visual=visual,
                    assets_dir=assets_dir,
                    relative_prefix="paper_assets",
                    vision_client=vision_client,
                    declared_reference=declared_reference,
                    absolute_tolerance=absolute_tolerance,
                )
            )
        except Exception as exc:
            records.append(
                {
                    "id": visual.visual_id,
                    "figure_label": visual.figure_label,
                    "purpose": visual.purpose,
                    "focus": visual.focus,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    payload = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "vision_analysis_enabled": vision_client is not None,
        "visual_inputs": records,
    }
    evidence_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return evidence_path
