# Troubleshooting

## Installation Issues

### NumPy Version Conflict (Important)

**Problem:**
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x as it may crash.
AttributeError: _ARRAY_API not found
```

**Solution:**
PaddleOCR requires NumPy < 2.x. Install the compatible version first:

```bash
pip install "numpy<2"
```

Then install PaddleOCR:
```bash
pip install paddlepaddle paddleocr opencv-python
```

**Note:** Some other packages (like reme-ai) may require NumPy 2.x. Consider using virtual environments to isolate dependencies.

