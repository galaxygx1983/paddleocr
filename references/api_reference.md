# API Reference

## PaddleOCRWrapper

Main class for OCR operations.

### Constructor

```python
PaddleOCRWrapper(language: str = "ch", performance: str = "balanced", **kwargs)
```

**Parameters:**
- `language` (str): Language code
  - `'ch'`: Chinese (default)
  - `'en'`: English
  - `'ml'`: Multilingual
- `performance` (str): Performance mode
  - `'accuracy'`: Best accuracy, slower
  - `'balanced'`: Balanced speed/accuracy (default)
  - `'speed'`: Fastest, lower accuracy
- `**kwargs`: Additional PaddleOCR parameters

**Example:**
```python
ocr = PaddleOCRWrapper(language='ch')
ocr = PaddleOCRWrapper(language='en')
ocr = PaddleOCRWrapper(language='ml')
```

**Notes:**
- Automatic handling of Windows Chinese/non-ASCII paths
- Uses `use_textline_orientation=True` for text rotation detection
- Uses PaddleOCR 3.x `predict()` API (not deprecated `ocr()` method)

### Methods

#### recognize(image_path: str) -> OCRPageResult

Performs OCR on a single image.

**Parameters:**
- `image_path` (str): Path to image file

**Returns:**
- `OCRPageResult`: OCR results with text, positions, and confidence

**Example:**
```python
result = ocr.recognize('document.png')
for item in result.results:
    print(f"[{item.confidence:.2f}] {item.text}")
```

#### get_text_lines(result: OCRPageResult, confidence_threshold: float = 0.5) -> List[str]

Extracts text organized by lines in reading order (top to bottom).

**Parameters:**
- `result` (OCRPageResult): OCR result from `recognize()`
- `confidence_threshold` (float): Minimum confidence for text inclusion. Default: 0.5

**Returns:**
- `List[str]`: List of text lines in reading order

**Example:**
```python
result = ocr.recognize('certificate.jpg')
lines = ocr.get_text_lines(result)

for line in lines:
    print(line)

# With higher confidence threshold
lines = ocr.get_text_lines(result, confidence_threshold=0.7)
```

#### get_high_confidence_results(result: OCRPageResult, threshold: float = 0.85) -> List[OCRResult]

Gets only high-confidence OCR results.

**Parameters:**
- `result` (OCRPageResult): OCR result from `recognize()`
- `threshold` (float): Minimum confidence threshold. Default: 0.85

**Returns:**
- `List[OCRResult]`: List of high-confidence results

**Example:**
```python
result = ocr.recognize('document.png')
high_conf = ocr.get_high_confidence_results(result, threshold=0.9)

for item in high_conf:
    print(f"[{item.confidence:.2f}] {item.text}")
```

#### batch_recognize(image_paths: List[str]) -> BatchOCRResult

Processes multiple images.

**Parameters:**
- `image_paths` (List[str]): List of image paths

**Returns:**
- `BatchOCRResult`: Combined results for all images

**Example:**
```python
images = ['img1.png', 'img2.jpg', 'img3.jpeg']
result = ocr.batch_recognize(images)
print(f"Processed: {result.successful}/{result.total_files}")
```

#### batch_from_directory(input_dir: str, extensions: tuple = (".png", ".jpg", ".jpeg", ".bmp", ".tiff"), recursive: bool = False) -> BatchOCRResult

Processes all images in a directory.

**Parameters:**
- `input_dir` (str): Input directory path
- `extensions` (tuple): File extensions to process
- `recursive` (bool): Search subdirectories

**Returns:**
- `BatchOCRResult`: All OCR results

**Example:**
```python
# Process all images in directory
result = ocr.batch_from_directory('scans/')

# Process recursively
result = ocr.batch_from_directory('documents/', recursive=True)
```

#### save_results(result: Union[OCRPageResult, BatchOCRResult], output_path: str, output_format: str = "json") -> str

Saves OCR results to file.

**Parameters:**
- `result`: OCR result object
- `output_path` (str): Output file path
- `output_format` (str): `'json'` or `'txt'`. Default: `'json'`

**Returns:**
- `str`: Path to saved file

**Example:**
```python
# Save as JSON
ocr.save_results(result, 'output.json')

# Save as plain text
ocr.save_results(result, 'output.txt', output_format='txt')
```

#### recognize_with_layout(image_path: str) -> Dict

Performs OCR with layout analysis (detects tables, images, text regions). Requires PPStructureV3.

**Parameters:**
- `image_path` (str): Path to image file

**Returns:**
- `Dict` with keys:
  - `layout`: List of detected layout regions
  - `text_regions`: List of extracted text with confidence
  - `tables`: Detected table regions
  - `processing_time`: Processing time in seconds

**Note:** Requires extra dependencies: `pip install paddleocr[all]`

## ImagePreprocessor

### Static Methods

#### load_image(image_path: str) -> np.ndarray

Loads an image from file.

**Parameters:**
- `image_path` (str): Path to image

**Returns:**
- `numpy.ndarray`: Image array

**Raises:**
- `ImportError`: If OpenCV not installed
- `ValueError`: If image cannot be loaded

#### save_image(image_path: str, img) -> bool

Saves an image to file.

**Parameters:**
- `image_path` (str): Output path
- `img`: Image array

**Returns:**
- `bool`: Success status

#### preprocess(image_path: str, output_dir: str = '.', deskew: bool = True, denoise: bool = True, enhance_contrast: bool = True, binarize_flag: bool = False, output_prefix: str = 'processed') -> Dict

Complete preprocessing pipeline.

**Returns:**
- `Dict` with paths to processed images:
  - `original`: Original image path
  - `deskewed`: Deskewed image
  - `denoised`: Denoised image
  - `contrast`: Contrast enhanced image
  - `binary`: Binarized image
  - `final`: Final processed image

## BatchOCRProcessor

### Constructor

```python
BatchOCRProcessor(
    language: str = 'ch',
    performance: str = 'balanced',
    max_workers: int = 4,
    **kwargs
)
```

**Parameters:**
- `language` (str): OCR language
- `performance` (str): Performance mode
- `max_workers` (int): Maximum parallel workers

### Methods

#### process_batch(image_paths: List[str], show_progress: bool = True) -> BatchOCRResult

Processes multiple images with multi-threading.

**Parameters:**
- `image_paths` (List[str]): List of image paths
- `show_progress` (bool): Show progress bar

**Returns:**
- `BatchOCRResult`: Combined results

#### process_directory(input_dir: str, extensions: tuple = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff'), recursive: bool = False, show_progress: bool = True) -> BatchOCRResult

Processes all images in a directory.

**Parameters:**
- `input_dir` (str): Input directory
- `extensions` (tuple): File extensions
- `recursive` (bool): Search subdirectories
- `show_progress` (bool): Show progress bar

**Returns:**
- `BatchOCRResult`: All results

#### get_summary(result: BatchOCRResult) -> str

Generates a summary report.

**Returns:**
- `str`: Formatted summary report

## Data Classes

### OCRResult

```python
@dataclass
class OCRResult:
    text: str                      # Extracted text
    confidence: float              # Recognition confidence (0-1)
    position: List[List[int]]      # Bounding box [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    type: str = "text"            # "text", "table_cell", "handwritten"

    def to_dict(self) -> Dict: ...
```

**Attributes:**
- `text`: Extracted text content
- `confidence`: Recognition confidence score (0.0 - 1.0)
- `position`: Bounding box coordinates (4 points)
- `type`: Result type - `"text"`, `"table_cell"`, or `"handwritten"`

**Methods:**
- `to_dict() -> Dict`: Convert to dictionary

### OCRPageResult

```python
@dataclass
class OCRPageResult:
    input_path: str                     # Original image path
    page_index: Optional[int]           # Page index (for multi-page documents)
    results: List[OCRResult]            # List of OCRResult items
    processing_time: float              # Processing time in seconds
    model_settings: Dict[str, Any]      # Model configuration
    raw_result: Optional[Dict]          # Raw PaddleOCR result (advanced use)

    def to_dict(self) -> Dict: ...
```

**Attributes:**
- `input_path`: Path to the processed image
- `page_index`: Page index (None for single-page images)
- `results`: List of all detected text items
- `processing_time`: Time taken for OCR processing
- `model_settings`: Dictionary with language and performance mode
- `raw_result`: Raw PaddleOCR output (for advanced use cases)

**Methods:**
- `to_dict() -> Dict`: Convert to dictionary

### BatchOCRResult

```python
@dataclass
class BatchOCRResult:
    batch_id: str                       # Unique batch identifier
    timestamp: str                      # Processing timestamp (ISO format)
    total_files: int                    # Total files processed
    successful: int                     # Successfully processed files
    failed: int                         # Failed files
    results: List[OCRPageResult]        # Individual file results
    errors: List[Dict]                  # List of errors
    total_processing_time: float = 0.0  # Total batch processing time

    def to_dict(self) -> Dict: ...
```

**Attributes:**
- `batch_id`: Unique identifier for the batch
- `timestamp`: ISO format timestamp
- `total_files`: Total files in the batch
- `successful`: Number of successfully processed files
- `failed`: Number of failed files
- `results`: List of OCRPageResult for each file
- `errors`: List of error dictionaries with file path and error message
- `total_processing_time`: Total processing time in seconds

**Methods:**
- `to_dict() -> Dict`: Convert to dictionary
