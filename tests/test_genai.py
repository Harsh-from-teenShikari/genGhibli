from unittest import TestCase
from imgprocalgs.algorithms.genai import GenAIEnhancement, GenAIGhibliConverter
import os
from PIL import Image

class TestGenAI(TestCase):
    TEST_IMAGE = "tests/data/bird.jpg"
    DEST_PATH = "tests/data"

    def tearDown(self):
        if os.path.exists(os.path.join(self.DEST_PATH, "output_genai_enhanced.jpg")):
            os.remove(os.path.join(self.DEST_PATH, "output_genai_enhanced.jpg"))
        if os.path.exists(os.path.join(self.DEST_PATH, "output_genai_ghibli.jpg")):
            os.remove(os.path.join(self.DEST_PATH, "output_genai_ghibli.jpg"))

    def test_enhancement(self):
        output_file = os.path.join(self.DEST_PATH, "output_genai_enhanced.jpg")
        algo = GenAIEnhancement(self.TEST_IMAGE, output_file, "make it sharper")
        algo.process()
        self.assertTrue(os.path.exists(output_file))

    def test_ghibli(self):
        output_file = os.path.join(self.DEST_PATH, "output_genai_ghibli.jpg")
        algo = GenAIGhibliConverter(self.TEST_IMAGE, output_file, "totoro style")
        algo.process()
        self.assertTrue(os.path.exists(output_file))
