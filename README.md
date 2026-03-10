# PaddleOCR Skill

> Comprehensive OCR wrapper for PaddleOCR 3.x

## 功能特性

- 支持中文、英文及多语言文档
- Windows 路径编码处理
- 批量处理
- JSON 输出（含文本位置和置信度）
- 高精度/快速模式切换

## 安装

```bash
# 注意: NumPy < 2.x (PaddleOCR 兼容性)
pip install "numpy<2"

# 安装 PaddleOCR
pip install paddlepaddle paddleocr opencv-python
```

## 快速开始

```python
from paddleocr import PaddleOCR

# 初始化
ocr = PaddleOCR(use_angle_cls=True, lang='ch')

# 识别图片
result = ocr.ocr('image.jpg', cls=True)

# 输出结果
for line in result:
    print(line)
```

## 支持语言

| 语言代码 | 说明 |
|----------|------|
| ch | 中文 |
| en | 英文 |
| korean | 韩文 |
| japan | 日文 |
| german | 德文 |
| french | 法文 |

## 详细文档

查看 [SKILL.md](SKILL.md) 获取完整使用指南。

## 许可证

MIT License