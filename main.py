import os
import argparse
import numpy
from utils import load_image, plot_sample
from model.common import resolve_single
import matplotlib.pyplot as plt

from model.edsr import edsr
from model.wdsr import wdsr_b
from model.srgan import generator


class Generator:
    def __init__(self, image_PATH):
        self.image_PATH = image_PATH
        self.image_name = os.path.basename(image_PATH)
        self.number_image = self.generate_unique_number()
        
        self.edsr = edsr(scale=4, num_res_blocks=16)
        self.wdsr = wdsr_b(scale=4, num_res_blocks=32)
        self.srgan = generator()
        
        self.edsr.load_weights('weights/edsr-16-x4/weights.h5')
        self.wdsr.load_weights('weights/wdsr-b-32-x4/weights.h5')
        self.srgan.load_weights('weights/srgan/gan_generator.h5')
    
    def resolve_and_save(self, model, output_path):
        lr = load_image(self.image_PATH)
        sr = resolve_single(model, lr)
        sr_array = sr.numpy()
        plt.imsave(output_path, sr_array)

    def use_all(self):
        self.resolve_and_save(self.edsr, f"rescaled_images/{self.image_name}_edsr_4x_{self.number_image}.png")
        self.resolve_and_save(self.wdsr, f"rescaled_images/{self.image_name}_wdsr_4x_{self.number_image}.png")
        self.resolve_and_save(self.srgan, f"rescaled_images/{self.image_name}_sr_4x_{self.number_image}.png")

    def generate_unique_number(self):
        for number in range(1, 99):
            filename = f"rescaled_images/{self.image_name}_edsr_4x_{number}.png"
            if not os.path.exists(filename):
                return number


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Increase_Image_Resolution")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    args = parser.parse_args()
    img_path = os.path.join('demo/', args.image_path)

    if not os.path.exists(img_path):
        print(f"Error: The specified input image '{img_path}' does not exist.")
    else:
        print(f'{args.image_path} was founded!')

        generator = Generator(img_path)
        generator.use_all()


