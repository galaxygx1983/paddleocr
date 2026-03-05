#!/usr/bin/env python3
"""
PaddleOCR Wrapper - Main OCR processing module

Provides high-level OCR functionality with:
- Multiple recognition modes (general, table, handwriting, multilingual)
- JSON output with text, position, and confidence scores
- Batch processing support
- Performance modes (accuracy vs speed)
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Optional imports with availability flags
PADDLEOCR_AVAILABLE = False
OPENCV_AVAILABLE = False

try:
    from paddleocr import PaddleOCR, PPStructureV3

    PADDLEOCR_AVAILABLE = True
except ImportError:
    print(
        "Warning: PaddleOCR not available. Install with: pip install paddleocr paddlepaddle"
    )

try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    pass  # OpenCV optional for image reading workarounds


@dataclass
class OCRResult:
    """Single OCR result item"""

    text: str
    confidence: float
    position: List[List[int]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    type: str = "text"  # text, table_cell, handwritten

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "position": self.position,
            "type": self.type,
        }


@dataclass
class OCRPageResult:
    """OCR result for a single page/image"""

    input_path: str
    page_index: Optional[int]
    results: List[OCRResult]
    processing_time: float
    model_settings: Dict[str, Any] = field(default_factory=dict)
    raw_result: Optional[Dict] = None  # Store raw PaddleOCR result for advanced use

    def to_dict(self) -> Dict:
        avg_confidence = (
            sum(r.confidence for r in self.results) / len(self.results)
            if self.results
            else 0.0
        )
        return {
            "input_path": self.input_path,
            "page_index": self.page_index,
            "processing_time_seconds": round(self.processing_time, 3),
            "model_settings": self.model_settings,
            "results": [r.to_dict() for r in self.results],
            "summary": {
                "total_items": len(self.results),
                "avg_confidence": round(avg_confidence, 4),
                "text_items": len([r for r in self.results if r.type == "text"]),
                "table_items": len([r for r in self.results if r.type == "table_cell"]),
                "handwritten_items": len(
                    [r for r in self.results if r.type == "handwritten"]
                ),
            },
        }


@dataclass
class BatchOCRResult:
    """Batch OCR processing result"""

    batch_id: str
    timestamp: str
    total_files: int
    successful: int
    failed: int
    results: List[OCRPageResult]
    errors: List[Dict]
    total_processing_time: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "batch_id": self.batch_id,
            "timestamp": self.timestamp,
            "total_files": self.total_files,
            "successful": self.successful,
            "failed": self.failed,
            "total_processing_time_seconds": round(self.total_processing_time, 3),
            "results": [r.to_dict() for r in self.results],
            "errors": self.errors,
        }


class PaddleOCRWrapper:
    """
    Main wrapper for PaddleOCR functionality

    Usage:
        ocr = PaddleOCRWrapper(language='ch', performance='balanced')
        result = ocr.recognize('image.png')
    """

    def __init__(self, language: str = "ch", performance: str = "balanced", **kwargs):
        """
        Initialize PaddleOCR wrapper

        Args:
            language: Language code(s) - 'ch' (Chinese), 'en' (English), 'ml' (multilingual)
            performance: 'accuracy', 'balanced', or 'speed'
            **kwargs: Additional PaddleOCR parameters
        """
        if not PADDLEOCR_AVAILABLE:
            raise ImportError(
                "PaddleOCR not available. Install: pip install paddleocr paddlepaddle"
            )

        self.language = language
        self.performance = performance
        self.kwargs = kwargs

        # Initialize PaddleOCR (basic OCR)
        # Note: use_angle_cls is deprecated, use use_textline_orientation instead
        self.ocr = PaddleOCR(
            use_textline_orientation=True,  # Enable text line orientation detection
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            **kwargs,
        )

        # Lazy initialization for PPStructureV3 (requires extra dependencies)
        self._table_ocr = None

        self.batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    @property
    def table_ocr(self):
        """Lazy load PPStructureV3 only when needed"""
        if self._table_ocr is None:
            try:
                self._table_ocr = PPStructureV3(
                    use_doc_orientation_classify=False, use_doc_unwarping=False
                )
            except Exception as e:
                print(
                    f"Warning: PPStructureV3 not available ({e}). Layout analysis disabled."
                )
                self._table_ocr = False  # Mark as unavailable
        return self._table_ocr

    def recognize(self, image_path: str) -> OCRPageResult:
        """
        Perform OCR on a single image

        Args:
            image_path: Path to image file

        Returns:
            OCRPageResult with all detected text
        """
        import time
        import tempfile
        import shutil

        start_time = time.time()

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Handle Windows path encoding issues for Chinese/non-ASCII paths
        temp_path = None
        try:
            # Get original extension
            original_ext = os.path.splitext(image_path)[1].lower() or ".jpg"

            # Copy file to temp ASCII path using shutil (preserves binary data)
            temp_dir = tempfile.gettempdir()
            temp_filename = f"paddleocr_{int(time.time())}{original_ext}"
            temp_path = os.path.join(temp_dir, temp_filename)

            # Copy using binary mode to preserve file content
            with open(image_path, "rb") as src:
                with open(temp_path, "wb") as dst:
                    dst.write(src.read())

            image_path = temp_path

            # Perform OCR
            raw_results = self.ocr.predict(image_path)

            # Parse results - PaddleOCR 3.x returns a list of dicts
            results = []

            for res in raw_results:
                # Handle new PaddleOCR 3.x result format
                # Results is a dict-like object with keys: rec_texts, rec_scores, dt_polys, etc.
                if hasattr(res, "rec_texts") and res.rec_texts:
                    rec_texts = res.rec_texts
                    rec_scores = (
                        res.rec_scores
                        if hasattr(res, "rec_scores")
                        else [0.0] * len(rec_texts)
                    )
                    dt_polys = (
                        res.dt_polys
                        if hasattr(res, "dt_polys")
                        else [None] * len(rec_texts)
                    )

                    for text, score, poly in zip(rec_texts, rec_scores, dt_polys):
                        # Skip empty or very low confidence results
                        if not text or len(text.strip()) == 0:
                            continue

                        # Convert poly to list if needed
                        if poly is not None:
                            if hasattr(poly, "tolist"):
                                poly_list = poly.tolist()
                            elif isinstance(poly, list):
                                poly_list = poly
                            else:
                                poly_list = list(poly)
                        else:
                            poly_list = [[0, 0], [0, 0], [0, 0], [0, 0]]

                        results.append(
                            OCRResult(
                                text=text,
                                confidence=float(score),
                                position=poly_list,
                                type="text",
                            )
                        )

            processing_time = time.time() - start_time

            # Store raw result info for advanced use
            raw_result = None
            if raw_results:
                first_res = raw_results[0]
                raw_result = {
                    "rec_texts": list(first_res.rec_texts)
                    if hasattr(first_res, "rec_texts")
                    else [],
                    "rec_scores": list(first_res.rec_scores)
                    if hasattr(first_res, "rec_scores")
                    else [],
                    "angle": getattr(first_res, "angle", None),
                }

            return OCRPageResult(
                input_path=image_path,
                page_index=None,
                results=results,
                processing_time=processing_time,
                model_settings={
                    "language": self.language,
                    "performance_mode": self.performance,
                },
                raw_result=raw_result,
            )

        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    def recognize_with_layout(self, image_path: str) -> Dict:
        """
        Perform OCR with layout analysis (tables, images, etc.)

        Returns:
            Dictionary with layout detection and OCR results
        """
        import time

        start_time = time.time()

        # Check if PPStructureV3 is available
        if not self.table_ocr or self.table_ocr is False:
            return {
                "error": "PPStructureV3 not available. Install extra dependencies: pip install paddleocr[all]",
                "layout": [],
                "text_regions": [],
                "tables": [],
                "processing_time": time.time() - start_time,
            }

        # Get layout analysis
        layout_results = self.table_ocr.predict(image_path)

        # Parse structure results
        parsed = {
            "layout": [],
            "text_regions": [],
            "tables": [],
            "processing_time": time.time() - start_time,
        }

        for res in layout_results:
            if hasattr(res, "layout_det_res") and res.layout_det_res:
                boxes = (
                    res.layout_det_res.get("boxes", [])
                    if isinstance(res.layout_det_res, dict)
                    else []
                )
                for box in boxes:
                    parsed["layout"].append(
                        {
                            "type": box.get("label", "unknown"),
                            "confidence": box.get("score", 0),
                            "position": box.get("coordinate", []),
                        }
                    )

            if hasattr(res, "overall_ocr_res") and res.overall_ocr_res:
                ocr_data = res.overall_ocr_res
                if "rec_texts" in ocr_data:
                    rec_scores = ocr_data.get("rec_scores", [])
                    for idx, text in enumerate(ocr_data["rec_texts"]):
                        score = rec_scores[idx] if idx < len(rec_scores) else 0.0
                        parsed["text_regions"].append(
                            {"text": text, "confidence": float(score)}
                        )

        return parsed

    def batch_recognize(self, image_paths: List[str]) -> BatchOCRResult:
        """
        Perform OCR on multiple images

        Args:
            image_paths: List of image file paths

        Returns:
            BatchOCRResult with all results
        """
        import time

        batch_start = time.time()

        results = []
        errors = []
        successful = 0
        failed = 0

        for i, image_path in enumerate(image_paths):
            try:
                result = self.recognize(image_path)
                results.append(result)
                successful += 1

            except Exception as e:
                errors.append({"file": image_path, "error": str(e), "index": i})
                failed += 1

        batch_result = BatchOCRResult(
            batch_id=self.batch_id,
            timestamp=datetime.now().isoformat(),
            total_files=len(image_paths),
            successful=successful,
            failed=failed,
            results=results,
            errors=errors,
            total_processing_time=time.time() - batch_start,
        )

        return batch_result

    def batch_from_directory(
        self,
        input_dir: str,
        extensions: tuple = (".png", ".jpg", ".jpeg", ".bmp", ".tiff"),
        recursive: bool = False,
    ) -> BatchOCRResult:
        """
        Process all images in a directory

        Args:
            input_dir: Input directory path
            extensions: File extensions to process
            recursive: Search recursively in subdirectories

        Returns:
            BatchOCRResult
        """
        image_paths = []

        if recursive:
            for ext in extensions:
                image_paths.extend(Path(input_dir).rglob(f"*{ext}"))
                image_paths.extend(Path(input_dir).rglob(f"*{ext.upper()}"))
        else:
            for ext in extensions:
                image_paths.extend(Path(input_dir).glob(f"*{ext}"))
                image_paths.extend(Path(input_dir).glob(f"*{ext.upper()}"))

        image_paths = [str(p) for p in image_paths]
        image_paths.sort()

        if not image_paths:
            raise ValueError(
                f"No images found in {input_dir} with extensions {extensions}"
            )

        print(f"Found {len(image_paths)} images to process")

        return self.batch_recognize(image_paths)

    def save_results(
        self,
        result: Union[OCRPageResult, BatchOCRResult],
        output_path: str,
        output_format: str = "json",
    ) -> str:
        """
        Save OCR results to file

        Args:
            result: OCR result object
            output_path: Output file path
            output_format: 'json' or 'txt'

        Returns:
            Path to saved file
        """
        if output_format == "json":
            output_path = str(output_path)
            if not output_path.endswith(".json"):
                output_path += ".json"

            with open(output_path, "w", encoding="utf-8") as f:
                if hasattr(result, "to_dict"):
                    json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
                else:
                    json.dump(result, f, ensure_ascii=False, indent=2)

        elif output_format == "txt":
            output_path = str(output_path)
            if not output_path.endswith(".txt"):
                output_path += ".txt"

            with open(output_path, "w", encoding="utf-8") as f:
                if hasattr(result, "results"):
                    for item in result.results:
                        f.write(f"{item.text}\t{item.confidence:.4f}\n")
                else:
                    for res in result.get("results", []):
                        for item in res.get("results", []):
                            f.write(f"{item['text']}\t{item['confidence']:.4f}\n")

        return output_path

    def get_text_only(self, result: OCRPageResult) -> str:
        """
        Extract plain text from OCR result

        Args:
            result: OCR result

        Returns:
            Plain text string
        """
        return "\n".join(item.text for item in result.results)

    def get_text_lines(
        self, result: OCRPageResult, confidence_threshold: float = 0.5
    ) -> List[str]:
        """
        Extract text organized by lines (sorted by vertical position)

        Args:
            result: OCR result
            confidence_threshold: Minimum confidence for text inclusion (default 0.5)

        Returns:
            List of text lines in reading order
        """
        # Filter by confidence and sort by vertical position
        filtered = [
            item for item in result.results if item.confidence >= confidence_threshold
        ]
        filtered.sort(key=lambda x: x.position[0][1])  # Sort by top Y coordinate

        # Group by lines (same Y position)
        text_lines = []
        current_line = []
        current_y = None
        line_threshold = 30  # pixels threshold for same line

        for item in filtered:
            top_y = item.position[0][1]
            if current_y is None:
                current_y = top_y
                current_line.append(item.text)
            elif abs(top_y - current_y) <= line_threshold:
                current_line.append(item.text)
            else:
                text_lines.append(" ".join(current_line))
                current_line = [item.text]
                current_y = top_y

        if current_line:
            text_lines.append(" ".join(current_line))

        return text_lines

    def get_high_confidence_results(
        self, result: OCRPageResult, threshold: float = 0.85
    ) -> List[OCRResult]:
        """
        Get only high-confidence OCR results

        Args:
            result: OCR result
            threshold: Minimum confidence (default 0.85)

        Returns:
            List of high-confidence OCRResult items
        """
        return [item for item in result.results if item.confidence >= threshold]


def main():
    """CLI entry point for PaddleOCR wrapper"""
    import argparse

    parser = argparse.ArgumentParser(
        description="PaddleOCR Wrapper - Multi-purpose OCR Tool"
    )
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument(
        "-o", "--output", default="ocr_results.json", help="Output file path"
    )
    parser.add_argument("-l", "--language", default="ch", help="Language: ch, en, ml")
    parser.add_argument(
        "-p",
        "--performance",
        default="balanced",
        choices=["accuracy", "balanced", "speed"],
        help="Performance mode",
    )
    parser.add_argument(
        "--batch", action="store_true", help="Enable batch processing for directory"
    )
    parser.add_argument(
        "--format", default="json", choices=["json", "txt"], help="Output format"
    )
    parser.add_argument("--layout", action="store_true", help="Enable layout analysis")
    parser.add_argument(
        "--recursive", action="store_true", help="Search recursively in directories"
    )

    args = parser.parse_args()

    try:
        ocr = PaddleOCRWrapper(language=args.language, performance=args.performance)

        if os.path.isdir(args.input) or args.batch:
            result = ocr.batch_from_directory(args.input, recursive=args.recursive)
        else:
            if args.layout:
                result = ocr.recognize_with_layout(args.input)
            else:
                result = ocr.recognize(args.input)

        output_path = args.output
        ocr.save_results(result, output_path, output_format=args.format)
        print(f"Results saved to: {output_path}")

        # Print summary
        if hasattr(result, "results"):
            total = len(result.results) if hasattr(result, "results") else 0
            if (
                hasattr(result, "results")
                and isinstance(result.results, list)
                and len(result.results) > 0
            ):
                if hasattr(result.results[0], "results"):
                    total = sum(len(r.results) for r in result.results)
            print(f"Total text items detected: {total}")

        elif isinstance(result, dict) and "text_regions" in result:
            print(f"Total text regions detected: {len(result.get('text_regions', []))}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
