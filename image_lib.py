import argparse
from PIL import Image, ImageDraw, ImageFont

'''
Run this file from the terminal (from VS Code, Terminal -> New Terminal) with the command
python .\image_lib.py --text 'REPLACE THIS WITH YOUR OWN TEXT'
if the image appears with your text on it, the Python configuration for Pillow & argpase is correct
'''

parser = argparse.ArgumentParser(description='Test use of image and arg parsing.')
parser.add_argument('-t', '--text', help='Text to display on image', required=True)
args = vars(parser.parse_args())

arg_txt = args['text']
txt_to_display = ''
b_w = 15
while len(arg_txt) > b_w:
    space_found = False
    for i in range(b_w, -1, -1):
        if arg_txt[i] == ' ':
            txt_to_display += arg_txt[:i] + '\n'
            arg_txt = arg_txt[i + 1:]
            space_found = True
            break
    if not space_found:
        i = 0
        any_spaces = False
        while i < len(arg_txt):
            if arg_txt[i] == ' ':
                txt_to_display += arg_txt[:i] + '\n'
                arg_txt = arg_txt[i + 1:]
                any_spaces = True
                break
            i += 1
        if not any_spaces:
            break
txt_to_display += arg_txt

base_img = Image.open("test_img.jpg")
base_img = base_img.resize((720, 844))
d1 = ImageDraw.Draw(base_img)
font = ImageFont.truetype("cour.ttf", 24)
d1.multiline_text((65, 75), txt_to_display, font= font, fill=(0, 0, 0))
base_img.show()


