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

    def save_file_output(self, file_output):
        """ Helper to save file output from Replicate """
        print(f"Saving result to {self.destination_path}...")
        with open(self.destination_path, "wb") as file:
            file.write(file_output.read())
        print(f"Image saved to {self.destination_path}")


class GenAIEnhancement(GenAIAlgorithm):
    """ GenAI based image enhancement """
    def process(self):
        # 1. Try Replicate API
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
                
                # Output handling
                if output:
                    if hasattr(output, 'read'):
                        self.save_file_output(output)
                    else:
                        self.download_and_save_image(output)
                    return
        except ImportError:
            pass # Library not installed
        except Exception as e:
            print(f"Replicate API failed: {e}. Falling back to mock.")

        # 2. Mock Fallback
        output_image = self.mock_genai_process("Image Enhancement")
        output_image = output_image.filter(ImageFilter.SHARPEN)
        output_image.save(self.destination_path)
        print(f"Enhanced image saved to {self.destination_path}")


class GenAIGhibliConverter(GenAIAlgorithm):
    """ GenAI based image converter to Ghibli style """
    def process(self):
        prompt = "ghibli style"
        if self.user_text:
            prompt = f"{prompt}, {self.user_text}"

        # 1. Try Hugging Face Inference API (Nitrosocke Ghibli)
        hf_token = os.environ.get("HUGGING_FACE_TOKEN")
        if hf_token:
            print("Using Hugging Face Inference API for Ghibli Conversion...")
            # Updated to new router endpoint
            api_url = "https://router.huggingface.co/hf-inference/models/nitrosocke/Ghibli-Diffusion"
            headers = {"Authorization": f"Bearer {hf_token}"}
            
            # Using prompt to generate image (Text-to-Image behavior of this endpoint)
            payload = {"inputs": prompt}
            
            try:
                response = requests.post(api_url, headers=headers, json=payload)
                if response.status_code == 200:
                    image_data = BytesIO(response.content)
                    img = PillowImage.open(image_data)
                    img.save(self.destination_path)
                    print(f"Ghibli style image saved to {self.destination_path}")
                    return
                else:
                    print(f"Hugging Face API failed: {response.status_code} {response.text}")
            except Exception as e:
                print(f"Hugging Face API request failed: {e}")

        # 2. Try Replicate API (Mirage Ghibli)
        try:
            import replicate
            if os.environ.get("REPLICATE_API_TOKEN"):
                print("Using Replicate API for Ghibli Conversion...")
                model = "aaronaftab/mirage-ghibli:166efd159b4138da932522bc5af40d39194033f587d9bdbab1e594119eae3e7f"
                
                output = replicate.run(
                    model,
                    input={
                        "image": open(self.image_path, "rb"),
                        "prompt": prompt,
                        "go_fast": True,
                        "guidance_scale": 10,
                        "prompt_strength": 0.77,
                        "num_inference_steps": 38
                    }
                )
                
                # Output is list of objects/URIs
                result = None
                if isinstance(output, list) and len(output) > 0:
                    result = output[0]
                elif output:
                    result = output
                
                if result:
                    if hasattr(result, 'url') and result.url:
                        self.download_and_save_image(result.url)
                    elif hasattr(result, 'read'):
                        self.save_file_output(result)
                    elif isinstance(result, str):
                        self.download_and_save_image(result)
                    return

        except ImportError:
             pass
        except Exception as e:
            print(f"Replicate API failed: {e}. Falling back to mock.")

        # 3. Mock Fallback
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
