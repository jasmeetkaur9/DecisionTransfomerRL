from PIL import Image

target_size = (200, 200)  # width, height

for i, path in enumerate(["imgs/ant.png", "imgs/hc.png", "imgs/hopper.png"]):
    img = Image.open(path)
    # Center crop
    w, h = img.size
    left = (w - target_size[0]) // 2
    top = (h - target_size[1]) // 2
    cropped = img.crop((left, top, left + target_size[0], top + target_size[1]))
    cropped.save(f"{i+1}.png")