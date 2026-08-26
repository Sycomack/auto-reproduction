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
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "Visual PDF preparation requires PyMuPDF. Install the project with: "
            "pip install -e ."
        ) from exc
    return fitz


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


def _localization_prompt(visual: VisualInput) -> str:
    target = (
        f"the region {visual.focus!r} within {visual.figure_label!r}"
        if visual.focus
        else repr(visual.figure_label)
    )
    return (
        f"Locate {target} on this rendered paper page. Return only "
        "a JSON object with found, bbox, caption_bbox, and confidence. bbox must "
        "cover the complete requested region, including its axes, legend, and panel "
        "title. When no subregion is requested, include all figure panels and the "
        "caption. Coordinates must be normalized [x0, y0, x1, y1] values between 0 "
        "and 1 relative to the full page. Do not transcribe or analyze the figure in "
        "this localization step."
    )


def _analysis_prompt(visual: VisualInput, extra_prompt: str = "") -> str:
    purpose = visual.purpose or "evidence for the supplied reproduction claim"
    focus = f" Focus only on {visual.focus}." if visual.focus else ""
    suffix = f" Additional request: {extra_prompt}" if extra_prompt else ""
    return (
        f"Analyze {visual.figure_label!r} as {purpose}.{focus} Return only one JSON "
        "object with figure_label, focus, caption, panel_count, panels, "
        "qualitative_findings, and uncertainties. Each panels item must contain "
        "panel_title, dataset, model, metric, x_axis, y_axis, and series. Each series "
        "item must contain name and points; every point must be an object with x, y, "
        "and uncertainty. Transcribe concrete numeric points whenever they are "
        "legible, while clearly marking visually estimated values. Preserve axis "
        "units and legend names. Use null and explain the uncertainty when a value "
        f"cannot be read. Never infer missing values from prior knowledge.{suffix}"
    )


def analyze_visual_reference(
    pdf_path: Path,
    visual: VisualInput,
    assets_dir: Path,
    relative_prefix: str,
    vision_client: VisionClient | None,
    extra_prompt: str = "",
) -> dict[str, Any]:
    page_number, caption_bbox = _find_caption(
        pdf_path, visual.figure_label, visual.page
    )
    stem = _slug(visual.visual_id)
    page_path = assets_dir / f"{stem}-page.png"
    crop_path = assets_dir / f"{stem}.png"
    page_render = render_pdf_region(pdf_path, page_number, page_path, dpi=140)

    bbox = None
    localization_method = "caption_heuristic"
    locator_response: dict[str, Any] | None = None
    locator_data: dict[str, Any] | None = None
    locator_error = ""
    if vision_client is not None:
        try:
            locator_response = vision_client.analyze(
                page_path, _localization_prompt(visual), detail="original"
            )
            locator_data = _parse_json_response(locator_response)
            bbox = _valid_bbox(locator_data.get("bbox")) if locator_data else None
            if bbox is not None:
                localization_method = "vision_bbox"
        except Exception as exc:
            locator_error = f"{type(exc).__name__}: {exc}"
    if bbox is None:
        bbox = _caption_fallback_bbox(caption_bbox)

    crop_render = render_pdf_region(
        pdf_path, page_number, crop_path, bbox=bbox, dpi=260
    )
    analysis_response: dict[str, Any] | None = None
    analysis: dict[str, Any] | None = None
    analysis_error = ""
    if vision_client is not None:
        try:
            analysis_response = vision_client.analyze(
                crop_path, _analysis_prompt(visual, extra_prompt), detail="original"
            )
            analysis = _parse_json_response(analysis_response)
            if not _response_text(analysis_response):
                analysis_error = "Vision model returned empty content"
        except Exception as exc:
            analysis_error = f"{type(exc).__name__}: {exc}"

    if analysis is not None:
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
        "bbox": bbox,
        "caption_bbox": caption_bbox,
        "localization_method": localization_method,
        "assets": {
            "page_preview": f"{relative_prefix}/{page_path.name}",
            "figure_crop": f"{relative_prefix}/{crop_path.name}",
        },
        "render": {"page_preview": page_render, "figure_crop": crop_render},
    }
    if locator_response is not None:
        record["locator"] = {
            "model": locator_response.get("model", ""),
            "parsed": locator_data,
            "raw": _response_text(locator_response),
            "usage": locator_response.get("usage", {}),
        }
    if locator_error:
        record["locator_error"] = locator_error
    if analysis_response is not None:
        record["analysis"] = analysis or {"raw": _response_text(analysis_response)}
        record["analysis_model"] = analysis_response.get("model", "")
        record["analysis_usage"] = analysis_response.get("usage", {})
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
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "vision_analysis_enabled": vision_client is not None,
        "visual_inputs": records,
    }
    evidence_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return evidence_path
