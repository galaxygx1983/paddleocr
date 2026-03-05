#!/usr/bin/env python3
"""
Batch OCR Processor - Process multiple images efficiently

Features:
- Multi-threaded processing
- Progress tracking
- Error handling and recovery
- Result aggregation
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from paddleocr_wrapper import PaddleOCRWrapper, OCRPageResult, BatchOCRResult

    OCR_WRAPPER_AVAILABLE = True
except ImportError:
    OCR_WRAPPER_AVAILABLE = False
    print("Warning: paddleocr_wrapper not available")


@dataclass
class BatchProgress:
    """Track batch processing progress"""

    total: int = 0
    completed: int = 0
    failed: int = 0
    start_time: float = 0.0
    results: List[OCRPageResult] = field(default_factory=list)
    errors: List[Dict] = field(default_factory=list)

    @property
    def progress(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.completed + self.failed) / self.total * 100

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    @property
    def eta(self) -> float:
        if self.completed == 0:
            return 0.0
        avg_time = self.elapsed_time / self.completed
        remaining = self.total - self.completed - self.failed
        return avg_time * remaining


class BatchOCRProcessor:
    """
    Batch OCR processing with multi-threading and progress tracking
    """

    def __init__(
        self,
        language: str = "ch",
        performance: str = "balanced",
        max_workers: int = 4,
        **kwargs,
    ):
        """
        Initialize batch processor

        Args:
            language: OCR language
            performance: Performance mode
            max_workers: Maximum parallel workers
            **kwargs: Additional PaddleOCR arguments
        """
        if not OCR_WRAPPER_AVAILABLE:
            raise ImportError("paddleocr_wrapper required")

        self.language = language
        self.performance = performance
        self.max_workers = max_workers
        self.kwargs = kwargs

        # Create OCR instance for each worker
        self.ocr_kwargs = kwargs

    def _create_ocr(self):
        """Create a new OCR instance"""
        return PaddleOCRWrapper(
            language=self.language, performance=self.performance, **self.kwargs
        )

    def process_single(self, image_path: str) -> Dict:
        """Process a single image

        Args:
            image_path: Path to image

        Returns:
            Dictionary with result or error
        """
        try:
            ocr = self._create_ocr()
            result = ocr.recognize(image_path)

            return {"success": True, "file": image_path, "result": result}
        except Exception as e:
            return {"success": False, "file": image_path, "error": str(e)}

    def process_batch(
        self, image_paths: List[str], show_progress: bool = True
    ) -> BatchOCRResult:
        """
        Process multiple images with multi-threading

        Args:
            image_paths: List of image paths
            show_progress: Show progress bar

        Returns:
            BatchOCRResult with all results
        """
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_time = time.time()

        results = []
        errors = []
        completed = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.process_single, path): path for path in image_paths
            }

            # Process with progress bar
            if show_progress:
                pbar = tqdm(total=len(image_paths), desc="Processing images")

            for future in as_completed(future_to_path):
                path = future_to_path[future]

                try:
                    output = future.result()
                    if output["success"]:
                        results.append(output["result"])
                        completed += 1
                    else:
                        errors.append({"file": path, "error": output["error"]})
                        failed += 1
                except Exception as e:
                    errors.append({"file": path, "error": str(e)})
                    failed += 1

                if show_progress:
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "completed": completed,
                            "failed": failed,
                            "progress": f"{completed + failed}/{len(image_paths)}",
                        }
                    )

            if show_progress:
                pbar.close()

        total_time = time.time() - start_time

        return BatchOCRResult(
            batch_id=batch_id,
            timestamp=datetime.now().isoformat(),
            total_files=len(image_paths),
            successful=completed,
            failed=failed,
            results=results,
            errors=errors,
            total_processing_time=total_time,
        )

    def process_directory(
        self,
        input_dir: str,
        extensions: tuple = (".png", ".jpg", ".jpeg", ".bmp", ".tiff"),
        recursive: bool = False,
        show_progress: bool = True,
    ) -> BatchOCRResult:
        """
        Process all images in a directory

        Args:
            input_dir: Input directory
            extensions: File extensions to process
            recursive: Search recursively
            show_progress: Show progress bar

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
            raise ValueError(f"No images found in {input_dir}")

        print(f"Found {len(image_paths)} images to process")

        return self.process_batch(image_paths, show_progress)

    def save_results(
        self,
        result: BatchOCRResult,
        output_path: str,
        output_format: str = "json",
        include_details: bool = True,
    ):
        """
        Save batch results to file

        Args:
            result: BatchOCRResult
            output_path: Output file path
            output_format: 'json' or 'txt'
            include_details: Include full result details
        """
        if output_format == "json":
            output_path = str(output_path)
            if not output_path.endswith(".json"):
                output_path += ".json"

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        elif output_format == "txt":
            output_path = str(output_path)
            if not output_path.endswith(".txt"):
                output_path += ".txt"

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"Batch ID: {result.batch_id}\n")
                f.write(f"Timestamp: {result.timestamp}\n")
                f.write(f"Total Files: {result.total_files}\n")
                f.write(f"Successful: {result.successful}\n")
                f.write(f"Failed: {result.failed}\n")
                f.write(f"Processing Time: {result.total_processing_time:.2f}s\n")
                f.write("\n" + "=" * 50 + "\n\n")

                for i, page_result in enumerate(result.results):
                    f.write(
                        f"\n--- Image {i + 1}: {os.path.basename(page_result.input_path)} ---\n"
                    )
                    for item in page_result.results:
                        f.write(f"[{item.confidence:.4f}] {item.text}\n")

        print(f"Results saved to: {output_path}")

    def get_summary(self, result: BatchOCRResult) -> str:
        """Generate summary report"""
        summary_lines = [
            "=" * 50,
            "BATCH OCR PROCESSING SUMMARY",
            "=" * 50,
            f"Batch ID: {result.batch_id}",
            f"Timestamp: {result.timestamp}",
            f"Total Files: {result.total_files}",
            f"Successful: {result.successful}",
            f"Failed: {result.failed}",
            f"Processing Time: {result.total_processing_time:.2f}s",
            f"Throughput: {result.total_files / result.total_processing_time:.2f} images/s",
            "",
            "SUCCESS RATE: {:.1f}%".format(
                result.successful / result.total_files * 100
                if result.total_files > 0
                else 0
            ),
            "",
        ]

        # Count result types
        total_texts = sum(len(r.results) for r in result.results)

        if total_texts > 0:
            avg_confidence = (
                sum(sum(item.confidence for item in r.results) for r in result.results)
                / total_texts
            )

            summary_lines.extend(
                [
                    f"Total Text Items: {total_texts}",
                    f"Average Confidence: {avg_confidence:.4f}",
                ]
            )

        if result.errors:
            summary_lines.extend(
                [
                    "",
                    "ERRORS:",
                    "-" * 20,
                ]
            )
            for error in result.errors[:10]:  # Show first 10 errors
                summary_lines.append(
                    f"  - {os.path.basename(error['file'])}: {error['error']}"
                )

            if len(result.errors) > 10:
                summary_lines.append(f"  ... and {len(result.errors) - 10} more errors")

        return "\n".join(summary_lines)


def main():
    """CLI entry point for batch processing"""
    import argparse

    parser = argparse.ArgumentParser(description="Batch OCR Processor")
    parser.add_argument("input", help="Input directory or image file")
    parser.add_argument(
        "-o", "--output", default="batch_results", help="Output file/directory"
    )
    parser.add_argument("-l", "--language", default="ch", help="Language: ch, en, ml")
    parser.add_argument(
        "-p",
        "--performance",
        default="balanced",
        choices=["accuracy", "balanced", "speed"],
        help="Performance mode",
    )
    parser.add_argument("-w", "--workers", type=int, default=4, help="Max workers")
    parser.add_argument("--recursive", action="store_true", help="Recursive search")
    parser.add_argument(
        "--format", default="json", choices=["json", "txt"], help="Output format"
    )
    parser.add_argument("--summary", action="store_true", help="Print summary")
    parser.add_argument("--no-progress", action="store_true", help="Hide progress bar")

    args = parser.parse_args()

    if not OCR_WRAPPER_AVAILABLE:
        print("Error: paddleocr_wrapper not available")
        return

    try:
        processor = BatchOCRProcessor(
            language=args.language,
            performance=args.performance,
            max_workers=args.workers,
        )

        if os.path.isdir(args.input):
            result = processor.process_directory(
                args.input, recursive=args.recursive, show_progress=not args.no_progress
            )
        else:
            single_result = processor.process_batch(
                [args.input], show_progress=not args.no_progress
            )
            result = single_result

        # Save results
        processor.save_results(result, args.output, output_format=args.format)

        # Print summary
        if args.summary:
            print("\n" + processor.get_summary(result))

        # Final stats
        print(f"\nCompleted: {result.successful}/{result.total_files} files")
        print(f"Total time: {result.total_processing_time:.2f}s")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
