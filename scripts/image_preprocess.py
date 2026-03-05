#!/usr/bin/env python3
"""
Image Preprocessing Utilities for OCR

Provides image preprocessing functions:
- Deskewing (rotation correction)
- Denoising
- Contrast enhancement (CLAHE)
- Binarization
- Complete preprocessing pipeline
"""

import os
from typing import Optional, Tuple
from pathlib import Path

# Optional imports
OPENCV_AVAILABLE = False
try:
    import cv2
    import numpy as np

    OPENCV_AVAILABLE = True
except ImportError:
    print("Warning: OpenCV not installed. Install with: pip install opencv-python")


class ImagePreprocessor:
    """Image preprocessing utilities for OCR"""

    @staticmethod
    def load_image(image_path: str):
        """Load image from file path

        Args:
            image_path: Path to image file

        Returns:
            Image as numpy array or None if failed
        """
        if not OPENCV_AVAILABLE:
            raise ImportError("OpenCV required for image processing")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return img

    @staticmethod
    def save_image(image_path: str, img) -> bool:
        """Save image to file

        Args:
            image_path: Output file path
            img: Image to save

        Returns:
            True if successful
        """
        if not OPENCV_AVAILABLE:
            raise ImportError("OpenCV required")
        return cv2.imwrite(image_path, img)

    @staticmethod
    def get_image_info(image_path: str) -> dict:
        """Get image information

        Args:
            image_path: Path to image

        Returns:
            Dictionary with image info
        """
        img = ImagePreprocessor.load_image(image_path)
        return {
            "path": image_path,
            "width": img.shape[1],
            "height": img.shape[0],
            "channels": img.shape[2] if len(img.shape) == 3 else 1,
            "dtype": str(img.dtype),
        }

    @staticmethod
    def deskew(image_path: str, output_path: Optional[str] = None) -> str:
        """Correct image skew/rotation

        Args:
            image_path: Input image path
            output_path: Optional output path

        Returns:
            Path to corrected image
        """
        if not OPENCV_AVAILABLE:
            raise ImportError("OpenCV required")

        import math

        img = ImagePreprocessor.load_image(image_path)
        gray = (
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        )

        # Calculate skew angle using Hough transform
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, math.pi / 180, 100, minLineLength=100, maxLineGap=10
        )

        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = math.atan2(y2 - y1, x2 - x1) * 180.0 / math.pi
                angles.append(angle)

            if angles:
                median_angle = float(np.median(angles))
                (h, w) = gray.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                corrected = cv2.warpAffine(
                    img,
                    M,
                    (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                )

                output = output_path or f"deskewed_{os.path.basename(image_path)}"
                ImagePreprocessor.save_image(output, corrected)
                return output

        # No correction needed
        if output_path:
            ImagePreprocessor.save_image(output_path, img)
            return output_path
        return image_path

    @staticmethod
    def denoise(
        image_path: str, strength: int = 3, output_path: Optional[str] = None
    ) -> str:
        """Apply denoising to image

        Args:
            image_path: Input image path
            strength: Denoising strength (1-10)
            output_path: Optional output path

        Returns:
            Path to denoised image
        """
        if not OPENCV_AVAILABLE:
            raise ImportError("OpenCV required")

        img = ImagePreprocessor.load_image(image_path)

        if len(img.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(
                img, None, strength, strength, 7, 21
            )
        else:
            denoised = cv2.fastNlMeansDenoising(img, None, strength, 7, 21)

        output = output_path or f"denoised_{os.path.basename(image_path)}"
        ImagePreprocessor.save_image(output, denoised)
        return output

    @staticmethod
    def enhance_contrast(
        image_path: str,
        clip_limit: float = 2.0,
        tile_grid_size: int = 8,
        output_path: Optional[str] = None,
    ) -> str:
        """Enhance image contrast using CLAHE

        Args:
            image_path: Input image path
            clip_limit: Contrast enhancement limit
            tile_grid_size: Grid size for CLAHE
            output_path: Optional output path

        Returns:
            Path to enhanced image
        """
        if not OPENCV_AVAILABLE:
            raise ImportError("OpenCV required")

        img = ImagePreprocessor.load_image(image_path)

        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(
                clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size)
            )
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(
                clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size)
            )
            enhanced = clahe.apply(img)

        output = output_path or f"enhanced_{os.path.basename(image_path)}"
        ImagePreprocessor.save_image(output, enhanced)
        return output

    @staticmethod
    def binarize(
        image_path: str, threshold: int = 0, output_path: Optional[str] = None
    ) -> str:
        """Binarize image using thresholding

        Args:
            image_path: Input image path
            threshold: Fixed threshold (0 for adaptive)
            output_path: Optional output path

        Returns:
            Path to binarized image
        """
        if not OPENCV_AVAILABLE:
            raise ImportError("OpenCV required")

        img = ImagePreprocessor.load_image(image_path)
        gray = (
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        )

        if threshold == 0:
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
        else:
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

        output = output_path or f"binary_{os.path.basename(image_path)}"
        ImagePreprocessor.save_image(output, binary)
        return output

    @staticmethod
    def preprocess(
        image_path: str,
        output_dir: str = ".",
        deskew: bool = True,
        denoise: bool = True,
        enhance_contrast: bool = True,
        binarize_flag: bool = False,
        output_prefix: str = "processed",
    ) -> dict:
        """
        Complete preprocessing pipeline

        Args:
            image_path: Input image path
            output_dir: Output directory
            deskew: Correct rotation/skew
            denoise: Apply denoising
            enhance_contrast: Apply CLAHE
            binarize_flag: Apply thresholding
            output_prefix: Prefix for output files

        Returns:
            Dictionary with paths to processed images
        """
        import math

        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(image_path).stem

        outputs = {"original": image_path}
        img = ImagePreprocessor.load_image(image_path)
        current_img = img

        if deskew:
            gray = (
                cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
                if len(current_img.shape) == 3
                else current_img.copy()
            )
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(
                edges, 1, math.pi / 180, 100, minLineLength=100, maxLineGap=10
            )

            if lines is not None:
                angles = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = math.atan2(y2 - y1, x2 - x1) * 180.0 / math.pi
                    angles.append(angle)

                if angles:
                    median_angle = float(np.median(angles))
                    (h, w) = gray.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    current_img = cv2.warpAffine(
                        current_img,
                        M,
                        (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE,
                    )

            deskew_path = os.path.join(output_dir, f"{output_prefix}_deskewed.jpg")
            ImagePreprocessor.save_image(deskew_path, current_img)
            outputs["deskewed"] = deskew_path

        if denoise:
            if len(current_img.shape) == 3:
                denoised = cv2.fastNlMeansDenoisingColored(
                    current_img, None, 3, 3, 7, 21
                )
            else:
                denoised = cv2.fastNlMeansDenoising(current_img, None, 3, 7, 21)
            current_img = denoised
            denoise_path = os.path.join(output_dir, f"{output_prefix}_denoised.jpg")
            ImagePreprocessor.save_image(denoise_path, current_img)
            outputs["denoised"] = denoise_path

        if enhance_contrast:
            if len(current_img.shape) == 3:
                lab = cv2.cvtColor(current_img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                lab = cv2.merge([l, a, b])
                current_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                current_img = clahe.apply(current_img)
            contrast_path = os.path.join(output_dir, f"{output_prefix}_contrast.jpg")
            ImagePreprocessor.save_image(contrast_path, current_img)
            outputs["contrast"] = contrast_path

        if binarize_flag:
            gray = (
                cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
                if len(current_img.shape) == 3
                else current_img.copy()
            )
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            binary_path = os.path.join(output_dir, f"{output_prefix}_binary.jpg")
            ImagePreprocessor.save_image(binary_path, binary)
            outputs["binary"] = binary_path

        # Final output
        final_path = os.path.join(output_dir, f"{output_prefix}_final.jpg")
        ImagePreprocessor.save_image(final_path, current_img)
        outputs["final"] = final_path

        return outputs


def main():
    """CLI entry point for image preprocessing"""
    import argparse

    parser = argparse.ArgumentParser(description="Image Preprocessing for OCR")
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("-o", "--output", default=".", help="Output directory")
    parser.add_argument("--deskew", action="store_true", help="Correct rotation/skew")
    parser.add_argument("--denoise", action="store_true", help="Apply denoising")
    parser.add_argument("--contrast", action="store_true", help="Enhance contrast")
    parser.add_argument("--binarize", action="store_true", help="Apply binarization")
    parser.add_argument("--all", action="store_true", help="Apply all preprocessing")

    args = parser.parse_args()

    if not OPENCV_AVAILABLE:
        print("Error: OpenCV required. Install: pip install opencv-python")
        return

    if os.path.isdir(args.input):
        # Process all images in directory
        for f in Path(args.input).glob("*"):
            if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
                print(f"Processing: {f.name}")
                try:
                    outputs = ImagePreprocessor.preprocess(
                        str(f),
                        args.output,
                        deskew=args.deskew or args.all,
                        denoise=args.denoise or args.all,
                        enhance_contrast=args.contrast or args.all,
                        binarize_flag=args.binarize or args.all,
                        output_prefix=f.stem,
                    )
                    print(f"  Final: {outputs.get('final', 'N/A')}")
                except Exception as e:
                    print(f"  Error: {e}")
    else:
        outputs = ImagePreprocessor.preprocess(
            args.input,
            args.output,
            deskew=args.deskew or args.all,
            denoise=args.denoise or args.all,
            enhance_contrast=args.contrast or args.all,
            binarize_flag=args.binarize or args.all,
        )
        print(f"Processed: {outputs}")


if __name__ == "__main__":
    main()
