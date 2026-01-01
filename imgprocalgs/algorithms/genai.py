""" Module including GenAI based algorithms """
import argparse
import time
import os
import requests
from io import BytesIO
from imgprocalgs.algorithms.base import BaseAlgorithm
from imgprocalgs.algorithms.utilities import ImageData
from imgprocalgs.visualisation.server import App
from PIL import Image as PillowImage, ImageFilter


class GenAIAlgorithm(BaseAlgorithm):
    """ Base class for GenAI algorithms """
    def __init__(self, image_path: str, destination_path: str, user_text: str = None):
        super().__init__(image_path, destination_path)
        self.user_text = user_text

    def mock_genai_process(self, action: str):
        """
        Mocking GenAI processing since we don't have access to an external API.
        In a real scenario, this would call an API like OpenAI DALL-E or Midjourney.
        """
        print(f"Uploading image to GenAI Service...")
        print(f"Calling GenAI service for: {action}")
        if self.user_text:
            print(f"With user instruction: {self.user_text}")

        # Simulating processing time and result
        time.sleep(1)

        return self.image.image.copy()

    def download_and_save_image(self, url: str):
        """ Helper to download image from URL and save it """
        print(f"Downloading result from {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            image_data = BytesIO(response.content)
            img = PillowImage.open(image_data)
            img.save(self.destination_path)
            print(f"Image saved to {self.destination_path}")
        else:
            raise Exception(f"Failed to download image from {url}")


class GenAIEnhancement(GenAIAlgorithm):
    """ GenAI based image enhancement """
    def process(self):
        try:
            import replicate
            if os.environ.get("REPLICATE_API_TOKEN"):
                print("Using Replicate API for Image Enhancement...")
                # Using Real-ESRGAN for face/general enhancement
                model = "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73ab41b2ee43ad4095a1c"

                output = replicate.run(
                    model,
                    input={
                        "image": open(self.image_path, "rb"),
                        "scale": 2,
                        "face_enhance": True
                    }
                )

                # Output is a URI string
                if output:
                    self.download_and_save_image(output)
                    return
        except ImportError:
            print("Replicate library not found. Falling back to mock.")
        except Exception as e:
            print(f"Replicate API failed: {e}. Falling back to mock.")

        # Fallback to mock
        output_image = self.mock_genai_process("Image Enhancement")
        output_image = output_image.filter(ImageFilter.SHARPEN)
        output_image.save(self.destination_path)
        print(f"Enhanced image saved to {self.destination_path}")


class GenAIGhibliConverter(GenAIAlgorithm):
    """ GenAI based image converter to Ghibli style """
    def process(self):
        try:
            import replicate
            if os.environ.get("REPLICATE_API_TOKEN"):
                print("Using Replicate API for Ghibli Conversion...")
                # Using mirage-ghibli model
                model = "aaronaftab/mirage-ghibli:166efd159b4138da932522bc5af40d39194033f587d9bdbab1e594119eae3e7f"

                prompt = "ghibli style"
                if self.user_text:
                    prompt = f"{prompt}, {self.user_text}"

                output = replicate.run(
                    model,
                    input={
                        "image": open(self.image_path, "rb"),
                        "prompt": prompt,
                        "prompt_strength": 0.8
                    }
                )

                # Output is list of URIs or single URI depending on model
                if isinstance(output, list):
                    url = output[0]
                else:
                    url = output

                if url:
                    self.download_and_save_image(url)
                    return
        except ImportError:
             print("Replicate library not found. Falling back to mock.")
        except Exception as e:
            print(f"Replicate API failed: {e}. Falling back to mock.")

        # Fallback to mock
        output_image = self.mock_genai_process("Convert to Ghibli Style")
        output_image = output_image.filter(ImageFilter.SMOOTH_MORE)
        output_image.save(self.destination_path)
        print(f"Ghibli style image saved to {self.destination_path}")


def example(app: App):
    # Example usage for the web app
    # We need to ensure directory exists if we use default data path
    if not os.path.exists('data'):
        os.makedirs('data')

    enhancer = GenAIEnhancement('tests/data/bird.jpg', 'data/genai_enhanced.jpg', "make it pop")
    enhancer.process()

    ghibli = GenAIGhibliConverter('tests/data/bird.jpg', 'data/genai_ghibli.jpg', "ghibli style")
    ghibli.process()

    data = {
        'title': 'GenAI Algorithms',
        'header': 'GenAI Enhancement and Style Transfer',
        'image_data': [
            ImageData("Enhanced", "genai_enhanced.jpg"),
            ImageData("Ghibli Style", "genai_ghibli.jpg"),
        ]
    }
    app.register_route("/genai", template_name="main_page.html", **data)


def parse_args():
    parser = argparse.ArgumentParser(description='GenAI algorithms')
    parser.add_argument("--src", type=str, help="Source file path.")
    parser.add_argument("--dest", type=str, help="Destination file path.", default='data/output.jpg')
    parser.add_argument("--method", type=str, choices=['enhance', 'ghibli'], help="GenAI method to use")
    parser.add_argument("--prompt", type=str, help="User text prompt", default="")
    parser.add_argument("--example", action='store_true', help="Show example")
    parser.add_argument("--visualize", action='store_true', help="Open visualization in webbrowser")
    return parser.parse_args()


def main():
    args = parse_args()
    app = App()

    if args.example:
        example(app)
        # Assuming the app server needs to run to see results if visualized
        if args.visualize:
            app.run_server('127.0.0.1', 8000, open_webiste=True)
    elif args.src and args.method:
        if args.method == 'enhance':
            algo = GenAIEnhancement(args.src, args.dest, args.prompt)
        elif args.method == 'ghibli':
            algo = GenAIGhibliConverter(args.src, args.dest, args.prompt)

        algo.process()
    else:
        print("Please provide --src and --method, or use --example")


if __name__ == "__main__":
    main()
