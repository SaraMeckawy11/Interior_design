import cv2

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at '{image_path}'")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

image_path = 'images/23.jpg'
room_image = load_image(image_path)  # RGB, HWC, uint8

# Only preview when running loader.py directly, so importing it never blocks.
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.imshow(room_image)
    plt.title("Original Room")
    plt.axis("off")
    plt.show()
