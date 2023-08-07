import unittest
from io import BytesIO
import numpy as np
import cv2
from main import ImageRecognition


class TestImageProcessor(unittest.TestCase):

    def setUp(self):
        # Create a simple 10x10 image with all pixel values set to 0
        self.test_image = np.zeros((10, 10), dtype=np.uint8)

    def test_patch_average(self):
        processor = ImageRecognition('test.png')
        processor.image = self.test_image

        avg = processor.get_patch_average(0, 0)
        self.assertEqual(avg, 0)

        # Set a 5x5 patch to 255
        self.test_image[0:5, 0:5] = 255
        avg = processor.get_patch_average(0, 0)
        self.assertEqual(avg, 255)

    def test_get_patch_average(self):
        processor = ImageRecognition("test.jpg")
        assert isinstance(processor.get_patch_average(0, 0), float), "get_patch_average should return a float"

    def test_max_patches_not_overlapping(self):
        processor = ImageRecognition("test.jpg")
        patches = processor.find_max_patches()
        for i in range(len(patches)):
            for j in range(i + 1, len(patches)):
                assert not (abs(patches[i][0] - patches[j][0]) < 5 and abs(
                    patches[i][1] - patches[j][1]) < 5), "Patches should not overlap"

    def test_find_max_patches_returns_correct_number_of_patches(self):
        processor = ImageRecognition("test.jpg")
        patches = processor.find_max_patches()
        assert len(patches) == 4, "find_max_patches should return 4 patches"

    def test_calculate_area_is_correct(self):
        processor = ImageRecognition("test.jpg")
        points = [(0, 0), (5, 0), (5, 5), (0, 5)]
        assert processor.calculate_area(points) == 25, "calculate_area should return the correct area"

    def test_save_image_creates_file(self):
        import os
        processor = ImageRecognition("test.jpg")
        processor.save_image(processor.image, "output_test.jpg")
        assert os.path.isfile("output_test.jpg"), "save_image should create a file"

    def test_process_outputs_correct_area(self):
        import io
        import sys
        processor = ImageRecognition("test.jpg")
        sys.stdout = io.StringIO()
        processor.recognize("output_test.jpg")
        assert "Area of quadrilateral: " in sys.stdout.getvalue(), "process should print the area of the quadrilateral"

    def test_process_creates_file(self):
        import os
        processor = ImageRecognition("test.jpg")
        processor.recognize("output_test.jpg")
        assert os.path.isfile("output_test.jpg"), "process should create a file"


if __name__ == '__main__':
    unittest.main()
