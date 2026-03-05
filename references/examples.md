# Common Usage Examples

## Basic Examples

### 1. Quick Start - Single Image OCR

```python
#!/usr/bin/env python3
"""Quick start example for PaddleOCR"""

from paddleocr_wrapper import PaddleOCRWrapper

def main():
    # Initialize OCR with accuracy mode
    ocr = PaddleOCRWrapper(language='ch', performance='accuracy')
    
    # Recognize text from image (supports Chinese paths)
    result = ocr.recognize('sample_document.png')
    
    # Print results
    print(f"Found {len(result.results)} text items:")
    for i, item in enumerate(result.results):
        print(f"  {i+1}. [{item.confidence:.2f}] {item.text}")
    
    # Sort by vertical position
    sorted_results = sorted(result.results, key=lambda x: x.position[0][1])
    
    # Save to JSON
    ocr.save_results(result, 'output.json')
    print(f"\nResults saved to output.json")

if __name__ == "__main__":
    main()
```

### 2. Windows Chinese Path Support

```python
# PaddleOCRWrapper automatically handles Chinese paths
from paddleocr_wrapper import PaddleOCRWrapper

ocr = PaddleOCRWrapper(language='ch', performance='accuracy')

# Works with Chinese paths on Windows
result = ocr.recognize(r'E:\临时文件\图片.jpg')

# Or with Chinese filename
result = ocr.recognize(r'E:\documents\聘书.jpg')
```

### 3. Batch Processing Multiple Images

```python
#!/usr/bin/env python3
"""Batch OCR processing example"""

from batch_processor import BatchOCRProcessor

def main():
    # Create processor with 4 workers
    processor = BatchOCRProcessor(
        language='ch',
        performance='balanced',
        max_workers=4
    )
    
    # Process directory
    print("Processing images in 'documents/'...")
    result = processor.process_directory(
        'documents/',
        recursive=True,
        show_progress=True
    )
    
    # Save results
    processor.save_results(result, 'batch_output.json', output_format='json')
    
    # Print summary
    print(f"\n{processor.get_summary(result)}")
    
    # Print total processing time
    print(f"\nTotal time: {result.total_processing_time:.2f}s")

if __name__ == "__main__":
    main()
```

### 3. Image Preprocessing + OCR

```python
#!/usr/bin/env python3
"""Preprocessing pipeline example"""

from paddleocr_wrapper import PaddleOCRWrapper
from image_preprocess import ImagePreprocessor

def main():
    # Preprocess image
    print("Preprocessing image...")
    outputs = ImagePreprocessor.preprocess(
        'scan_low_quality.jpg',
        output_dir='preprocessed/',
        deskew=True,
        denoise=True,
        enhance_contrast=True
    )
    
    print(f"Preprocessed image: {outputs['final']}")
    
    # OCR on preprocessed image
    ocr = PaddleOCRWrapper(language='ch', performance='accuracy')
    result = ocr.recognize(outputs['final'])
    
    print(f"\nExtracted {len(result.results)} text items:")
    for item in result.results:
        print(f"  [{item.confidence:.2f}] {item.text}")

if __name__ == "__main__":
    main()
```

### 4. Multilingual Document OCR

```python
#!/usr/bin/env python3
"""Multilingual OCR example"""

from paddleocr_wrapper import PaddleOCRWrapper

def main():
    # Use multilingual mode
    ocr = PaddleOCRWrapper(language='ml', performance='accuracy')
    
    # Process bilingual document
    result = ocr.recognize('bilingual_document.png')
    
    # Separate by confidence
    high_conf = [r for r in result.results if r.confidence > 0.9]
    medium_conf = [r for r in result.results if 0.7 < r.confidence <= 0.9]
    low_conf = [r for r in result.results if r.confidence <= 0.7]
    
    print("High confidence (>0.9):")
    for item in high_conf:
        print(f"  {item.text}")
    
    print(f"\nMedium confidence (0.7-0.9): {len(medium_conf)} items")
    print(f"Low confidence (<0.7): {len(low_conf)} items")
    
    # Save all results
    ocr.save_results(result, 'multilingual_output.json')

if __name__ == "__main__":
    main()
```

### 5. Layout Analysis for Table Detection

```python
#!/usr/bin/env python3
"""Layout analysis example"""

from paddleocr_wrapper import PaddleOCRWrapper

def main():
    ocr = PaddleOCRWrapper(language='ch', performance='accuracy')
    
    # Get layout analysis
    result = ocr.recognize_with_layout('document_with_tables.png')
    
    # Print detected layout regions
    print("Detected layout regions:")
    for region in result['layout']:
        print(f"  Type: {region['type']:15s} Confidence: {region['confidence']:.3f}")
        print(f"    Position: {region['position']}")
    
    # Extract text from text regions
    print(f"\nExtracted text ({len(result['text_regions'])} items):")
    for item in result['text_regions']:
        print(f"  [{item['confidence']:.2f}] {item['text']}")
    
    # Save results
    import json
    with open('layout_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
```

### 6. Confidence Filtering and Text Extraction (New Helper Methods)

```python
#!/usr/bin/env python3
"""Advanced result filtering example with new helper methods"""

from paddleocr_wrapper import PaddleOCRWrapper

def extract_important_text(image_path: str, min_confidence: float = 0.85):
    """Extract high-confidence text from image using helper methods"""
    ocr = PaddleOCRWrapper(language='ch')
    result = ocr.recognize(image_path)

    # Method 1: Use get_high_confidence_results helper
    reliable_text = ocr.get_high_confidence_results(result, threshold=min_confidence)

    print(f"High confidence text (>{min_confidence}):")
    for item in reliable_text:
        print(f"  [{item.confidence:.2f}] {item.text}")

    # Method 2: Get text organized by lines (reading order)
    print("\nText by reading order:")
    lines = ocr.get_text_lines(result)
    for i, line in enumerate(lines, 1):
        print(f"  Line {i}: {line}")

    # Method 3: Filter by confidence manually
    uncertain_text = [r for r in result.results if r.confidence < min_confidence]

    if uncertain_text:
        print(f"\nUncertain text ({len(uncertain_text)} items, confidence < {min_confidence}):")
        for item in uncertain_text:
            print(f"  [{item.confidence:.2f}] {item.text}")

    # Save plain text organized by lines
    with open('extracted_text.txt', 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')

    # Save detailed JSON
    ocr.save_results(result, 'detailed_results.json')

    return result

if __name__ == "__main__":
    result = extract_important_text('document.png', min_confidence=0.8)
```

### 7. Extract Text from Certificate/Award (Real-world Example)

```python
#!/usr/bin/env python3
"""Extract text from certificates or awards with organized output"""

from paddleocr_wrapper import PaddleOCRWrapper

def extract_certificate_info(image_path: str):
    """Extract and organize certificate information"""
    ocr = PaddleOCRWrapper(language='ch')

    # Perform OCR
    result = ocr.recognize(image_path)

    print("=" * 50)
    print("CERTIFICATE OCR RESULTS")
    print("=" * 50)

    # Get all results sorted by confidence
    print(f"\nAll detected text ({len(result.results)} items):")
    for i, item in enumerate(sorted(result.results, key=lambda x: -x.confidence), 1):
        print(f"  [{i}] Confidence: {item.confidence:.4f}")
        print(f"      Text: {item.text}")
        print(f"      Position: {item.position[0]}")  # Top-left coordinate

    # Get organized text by reading order
    print("\n" + "=" * 50)
    print("TEXT BY READING ORDER (Top to Bottom)")
    print("=" * 50)

    lines = ocr.get_text_lines(result)
    for i, line in enumerate(lines, 1):
        print(f"  Line {i}: {line}")

    # Get high-confidence results only
    print("\n" + "=" * 50)
    print("HIGH CONFIDENCE RESULTS (>0.9)")
    print("=" * 50)

    high_conf = ocr.get_high_confidence_results(result, threshold=0.9)
    for item in high_conf:
        print(f"  {item.text}")

    # Save results
    ocr.save_results(result, 'certificate_ocr.json')
    print(f"\nResults saved to certificate_ocr.json")

    # Print processing info
    print(f"\nProcessing time: {result.processing_time:.3f}s")
    print(f"Average confidence: {sum(r.confidence for r in result.results)/len(result.results):.4f}")

    return result

if __name__ == "__main__":
    result = extract_certificate_info('certificate.jpg')
```

### 7. Command Line Usage

```bash
# Single image OCR
python scripts/paddleocr_wrapper.py document.png -o results.json -l ch -p balanced

# Batch processing
python scripts/batch_processor.py ./images -o batch_results -l ch -w 4 --summary

# Image preprocessing
python scripts/image_preprocess.py scan.png --all -o processed/

# Process with accuracy mode
python scripts/paddleocr_wrapper.py document.png -p accuracy -o accurate_results.json
```

### 8. Performance Comparison

```python
#!/usr/bin/env python3
"""Compare different performance modes"""

from paddleocr_wrapper import PaddleOCRWrapper
import time

def benchmark_mode(performance: str, image_path: str) -> dict:
    """Benchmark a performance mode"""
    ocr = PaddleOCRWrapper(performance=performance)
    
    start = time.time()
    result = ocr.recognize(image_path)
    elapsed = time.time() - start
    
    return {
        'mode': performance,
        'time': elapsed,
        'items': len(result.results),
        'avg_confidence': sum(r.confidence for r in result.results) / len(result.results) if result.results else 0
    }

def main():
    image = 'test_document.png'
    
    modes = ['speed', 'balanced', 'accuracy']
    results = []
    
    for mode in modes:
        print(f"Testing {mode} mode...")
        result = benchmark_mode(mode, image)
        results.append(result)
        print(f"  Time: {result['time']:.2f}s, Items: {result['items']}, Avg Conf: {result['avg_confidence']:.3f}")
    
    print("\n" + "="*50)
    print("Benchmark Summary:")
    for r in results:
        print(f"  {r['mode']:10s}: {r['time']:.2f}s, {r['items']} items, {r['avg_confidence']:.3f} confidence")

if __name__ == "__main__":
    main()
```

## Tips

1. **Choose the right performance mode**:
   - `speed`: Real-time applications, previews
   - `balanced`: General use (default)
   - `accuracy`: Critical documents, archives

2. **Preprocessing improves results**:
   - Deskew for rotated scans
   - Denoise for low-quality images
   - Enhance contrast for faded documents

3. **Use confidence filtering**:
   - High confidence (>0.9): Very reliable
   - Medium (0.7-0.9): Likely correct
   - Low (<0.7): May need manual review

4. **Batch processing tips**:
   - Use 2-4 workers for optimal performance
   - Use `recursive=True` for nested directories
   - Monitor progress with `show_progress=True`
