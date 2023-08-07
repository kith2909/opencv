import cv2
import numpy as np
import argparse
import math
import sys


class ImageRecognition:
    """
    This class handles image processing tasks such as reading an image,
    finding patches of maximum brightness, calculating area and drawing quadrilaterals.
    """

    def __init__(self, file_path):
        """
        Initialize ImageRecognition with an image file.

        Args:
        file_path: str. Path to the image file.
        """
        self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        self.height, self.width = self.image.shape

    def get_patch_average(self, x, y):
        """
        Calculate the average brightness of a 5x5 patch at given coordinates.

        Args:
        x: int. x-coordinate of the top-left point of the patch.
        y: int. y-coordinate of the top-left point of the patch.

        Returns:
        float. The average brightness of the patch.
        """
        patch = self.image[x:x + 5, y:y + 5]
        if patch.size and not np.isnan(patch).all():
            return np.average(patch)
        else:
            return 0
        return np.average(self.image[x:x + 5, y:y + 5])

    def find_max_patches(self):
        """
        Find the four non-overlapping 5x5 patches with the highest average brightness.

        Returns:
        list. A list of tuples representing the center coordinates of the four patches.
        """
        averages = []
        for i in range(0, self.width - 5):
            for j in range(0, self.height - 5):
                averages.append(((i + 2, j + 2), self.get_patch_average(i, j)))

        sorted_patches = sorted(averages, key=lambda x: -x[1])
        top_patches = []

        for patch in sorted_patches:
            if not any(abs(patch[0][0] - p[0]) < 5 and abs(patch[0][1] - p[1]) < 5 for p in top_patches):
                top_patches.append(patch[0])
                if len(top_patches) == 4:
                    break

        return top_patches

    @staticmethod
    def calculate_area(points):
        """
        Calculate the area of the quadrilateral formed by given points.

        Args:
        points: list. A list of tuples representing the coordinates of the corners of the quadrilateral.

        Returns:
        float. The area of the quadrilateral.
        """
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        area = 0.5 * abs(
            x[0] * y[1] + x[1] * y[2] + x[2] * y[3] + x[3] * y[0] - x[1] * y[0] - x[2] * y[1] - x[3] * y[2] - x[0] * y[
                3])
        return area

    def draw_quadrilateral(self, points):
        """
        Draw a red quadrilateral on the image using given points as corners.

        Args:
        points: list. A list of tuples representing the coordinates of the corners of the quadrilateral.

        Returns:
        ndarray. The image with the quadrilateral drawn on it.
        """
        img_color = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        return cv2.polylines(img_color, [pts], True, (0, 0, 255), 2)

    @staticmethod
    def save_image(image, filename):
        """
        Save an image to a file.

        Args:
        image: ndarray. The image to be saved.
        filename: str. The name of the file to save the image to.
        """
        cv2.imwrite(filename, image)

    def recognize(self, output_filename):
        """
        The main processing function. Finds the patches, calculates the area, draws the quadrilateral, and saves the image.

        Args:
        output_filename: str. The name of the file to save the output image to.
        """
        patches = self.find_max_patches()
        print(f"Area of quadrilateral: {self.calculate_area(patches)}")
        output_image = self.draw_quadrilateral(patches)
        self.save_image(output_image, output_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="Path to the input image")
    parser.add_argument("output_filename", help="Filename for the output image")
    args = parser.parse_args()

    process = ImageRecognition(args.file_path)
    process.recognize(args.output_filename)
