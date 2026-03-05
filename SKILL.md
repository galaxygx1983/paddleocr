---
name: paddleocr
description: Comprehensive OCR wrapper for PaddleOCR 3.x with support for Chinese, English, and multilingual documents. Includes Windows path encoding handling, batch processing, JSON output with text positions and confidence scores. Use when Claude needs to extract text from images, scanned documents, screenshots, certificates, or any visual content.
---

# PaddleOCR Skill

## Overview

This skill provides a comprehensive wrapper around PaddleOCR 3.x, enabling high-quality OCR for various use cases. Supports Chinese, English, and multilingual documents with configurable accuracy/speed trade-offs.

## Requirements

```bash
# Required: NumPy < 2.x (PaddleOCR compatibility)
pip install "numpy<2"

# Install PaddleOCR and dependencies
pip install paddlepaddle paddleocr opencv-python

# For GPU support
pip install paddlepaddle-gpu
```

## Quick Start

### Basic OCR on a Single Image

```python
from paddleocr_wrapper import PaddleOCRWrapper

# Initialize OCR with Chinese language (default)
ocr = PaddleOCRWrapper(language='ch')

# Recognize text from image
result = ocr.recognize('document.png')

# Print extracted text with confidence
for item in result.results:
    print(f"[{item.confidence:.2f}] {item.text}")

# Save results to JSON
ocr.save_results(result, 'output.json')
```

### Extract Text Organized by Lines

```python
# Get text organized by reading order (top to bottom)
text_lines = ocr.get_text_lines(result)

for line in text_lines:
    print(line)
```

### Filter High-Confidence Results

```python
# Get only high-confidence results (> 0.85)
high_quality = ocr.get_high_confidence_results(result, threshold=0.85)
for item in high_quality:
    print(f"[{item.confidence:.2f}] {item.text}")
```

## Features

### 1. Windows Path Support

Handles Chinese and special character paths automatically by copying to temporary ASCII paths.

```python
# Works with Chinese paths
result = ocr.recognize('E:\\临时文件\\图片.jpg')
```

### 2. JSON Output Format

```json
{
  "input_path": "document.png",
  "page_index": null,
  "processing_time_seconds": 1.234,
  "model_settings": {
    "language": "ch",
    "performance_mode": "balanced"
  },
  "results": [
    {
      "text": "识别的文字",
      "confidence": 0.9979,
      "position": [[248, 368], [2806, 394], [2803, 675], [245, 649]],
      "type": "text"
    }
  ],
  "summary": {
    "total_items": 18,
    "avg_confidence": 0.8496,
    "text_items": 18,
    "table_items": 0,
    "handwritten_items": 0
  }
}
```

### 3. Batch Processing

```python
from batch_processor import BatchOCRProcessor

processor = BatchOCRProcessor(language='ch', workers=4)
result = processor.process_directory('images/', recursive=True)
processor.save_results(result, 'batch_output.json')
```

### 4. Image Preprocessing

```python
from image_preprocess import ImagePreprocessor

# Full preprocessing pipeline
outputs = ImagePreprocessor.preprocess(
    'input.jpg',
    output_dir='processed/',
    deskew=True,
    denoise=True,
    enhance_contrast=True
)
```

## Helper Methods

### get_text_lines(result, confidence_threshold=0.5)

Extract text organized by lines in reading order.

```python
# Get text lines sorted top to bottom
lines = ocr.get_text_lines(result)

# Filter by minimum confidence
lines = ocr.get_text_lines(result, confidence_threshold=0.7)
```

### get_high_confidence_results(result, threshold=0.85)

Get only high-confidence OCR results.

```python
# Get results with confidence > 0.85
high_conf = ocr.get_high_confidence_results(result, threshold=0.85)

# Get results with confidence > 0.9
very_high = ocr.get_high_confidence_results(result, threshold=0.9)
```

### save_results(result, output_path, output_format='json')

Save OCR results to JSON or TXT.

```python
# Save as JSON (default)
ocr.save_results(result, 'output.json')

# Save as plain text
ocr.save_results(result, 'output.txt', output_format='txt')
```

## Common Use Cases

### 1. Extract Text from Documents

```python
ocr = PaddleOCRWrapper(language='ch')
result = ocr.recognize('document.png')

for item in result.results:
    print(item.text)
```

### 2. Extract Text from Certificates/Awards

```python
ocr = PaddleOCRWrapper(language='ch')

result = ocr.recognize('certificate.jpg')

# Get organized text lines
lines = ocr.get_text_lines(result)
for line in lines:
    print(line)

# Save results
ocr.save_results(result, 'certificate_ocr.json')
```

### 3. Batch Process Document Scans

```python
from batch_processor import BatchOCRProcessor

processor = BatchOCRProcessor(
    language='ch',
    workers=4
)

result = processor.process_directory(
    'scans/',
    recursive=True,
    show_progress=True
)

print(f"Processed {result.successful}/{result.total_files} files")
```

### 4. Process Screenshots

```python
ocr = PaddleOCRWrapper(language='ch')

result = ocr.recognize('screenshot.png')

# Get high-confidence text only
high_quality = ocr.get_high_confidence_results(result, threshold=0.9)
for item in high_quality:
    print(item.text)
```

### 5. Multilingual Document OCR

```python
# Mix of Chinese and English
ocr = PaddleOCRWrapper(language='ml')
result = ocr.recognize('bilingual_document.png')

# Separate by confidence
high_confidence = ocr.get_high_confidence_results(result, threshold=0.9)
low_confidence = [r for r in result.results if r.confidence <= 0.9]
```

## Command Line Interface

### Single Image
```bash
python scripts/paddleocr_wrapper.py document.png -o results.json -l ch
```

### Batch Processing
```bash
python scripts/batch_processor.py ./images -o batch_results -l ch -w 4 --summary
```

### Image Preprocessing
```bash
python scripts/image_preprocess.py scan.png --all -o processed/
```

## API Reference

See [references/api_reference.md](references/api_reference.md) for complete API documentation.

## Examples

See [references/examples.md](references/examples.md) for common usage patterns.

## Troubleshooting

See [references/troubleshooting.md](references/troubleshooting.md) for common issues and solutions.

## Installation

```bash
# Install PaddleOCR and dependencies
pip install paddleocr paddlepaddle opencv-python

# For GPU support
pip install paddlepaddle-gpu
```

## Notes

- **PaddleOCR 3.x API**: Uses `predict()` method instead of deprecated `ocr()` method
- **Text Orientation**: Automatically detects text rotation with `use_textline_orientation=True`
- **Windows Path Handling**: Automatically copies files with Chinese/non-ASCII paths to temp directory
- **Confidence Filtering**: Results with very low confidence (< 0.5) are automatically filtered out
- **Layout Analysis**: PPStructureV3 (table/layout analysis) is lazy-loaded and optional

## Tips

1. **Performance**: Default settings work well for most use cases. Adjust confidence threshold for quality control.

2. **Certificate/Document OCR**: Use default settings, extract text with `get_text_lines()` for best reading order.

3. **Batch Processing**: Use 2-4 workers for optimal performance on most systems.

4. **Confidence Thresholding**:
   ```python
   # High confidence only (> 0.85)
   high_quality = ocr.get_high_confidence_results(result, threshold=0.85)

   # All results above 0.5
   filtered = [r for r in result.results if r.confidence > 0.5]
   ```

5. **Language Selection**: Use `ch` for Chinese, `en` for English, `ml` for multilingual documents.
