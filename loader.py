import cv2
import matplotlib.pyplot as plt

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

image_path = 'images/image.png' 
room_image = load_image(image_path)

plt.imshow(room_image)
plt.title("Original Room")
plt.axis("off")
plt.show()
