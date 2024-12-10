import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms as T
import xml.etree.ElementTree as ET
import json
from skimage.metrics import structural_similarity as SSIM


class LandmarkDataset(Dataset):
    def __init__(self, file_names, transform=None, mode='train'):
        self.file_names = file_names

        if transform:
            self.transform = transform
        else:
            self.transform = T.ToTensor()

        self.mode = mode

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)
        mask = load_mask(img_file_name)
        depth = load_depth(img_file_name)
        image = self.transform(image)
        mask = self.transform(mask.transpose(1, 2, 0))
        depth = self.transform(depth)

        if self.mode == 'train':
            return image, depth, mask, str(img_file_name)
        else:
            return image, depth, mask, str(img_file_name)


def load_image(path):
    img = cv2.imread(str(path))
    img = cv2.resize(img, (1024, 1024))

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_depth(path):
    img = cv2.imread(str(path).replace('images', 'depth_AdelaiDepth').replace('jpg', 'png'), 0)
    img = cv2.resize(img, (1024, 1024))

    return img.astype(np.uint8)


def load_mask(path):
    mask_path = str(path).replace('images', 'labels').replace('jpg', 'json')
    mask = load_json(mask_path)
    mask = cv2.resize(mask, (1024, 1024))
    masks = np.zeros(shape=(4, 1024, 1024), dtype=np.uint8)
    masks[0][mask == 0] = 255
    masks[1][mask == 1] = 255
    masks[2][mask == 2] = 255
    masks[3][mask == 3] = 255

    return masks


def load_xml(path):
    img = np.zeros((1080, 1920), dtype=np.uint8)
    tree = ET.parse(path)
    root = tree.getroot()

    # Loop through each contour
    for contour in root.findall('contour'):
        ctype = contour.find('contourType').text

        # Set the color based on contourType
        if ctype == 'Ridge':
            color = 1  # Red
        elif ctype == 'Silhouette':
            color = 2  # Green
        else:
            color = 3  # Blue

        # Get the x and y coordinates
        x_coords = [float(x) for x in contour.find('imagePoints/x').text.split(',')]
        y_coords = [float(y) for y in contour.find('imagePoints/y').text.split(',')]

        # Mark the points on the image
        for x, y in zip(x_coords, y_coords):
            cv2.circle(img, (int(x), int(y)), 3, color)

    return img


def load_json(path):
    image = np.zeros((1080, 1920), dtype=np.uint8)
    if '_31' in path or '_36' in path or '_25' in path or '_29' in path:
        image = np.zeros((2160, 3840), dtype=np.uint8)

    # Load JSON file
    with open(path, 'r') as f:
        data = json.load(f)

    # Iterate over shapes in data
    for shape in data['shapes']:
        points = shape['points']
        label = shape['label']

        # Choose line color based on label
        if label.startswith('r'):
            color = 1
        elif label.startswith('s'):
            color = 2
        elif label.startswith('l'):
            color = 3
        else:
            color = 0

        # Iterate over points in shape
        for i in range(1, len(points)):
            pt1 = tuple(map(int, points[i - 1]))
            pt2 = tuple(map(int, points[i]))

            # Draw line on image
            cv2.line(image, pt1, pt2, color, 30)

    return image


def save_img(img):
    color_map = {0: (0, 0, 0),  # Black
                 1: (0, 0, 255),  # Red
                 2: (0, 255, 0),  # Green
                 3: (255, 0, 0)}  # Blue

    # Get the height, width and channels
    height, width = img.shape
    channels = 3

    # Create a blank RGB image
    image = np.zeros((height, width, channels), np.uint8)

    # Map each value in the array to a color
    for i in range(height):
        for j in range(width):
            value = img[i, j]
            image[i, j] = color_map[value]
    return image


def ssim(img1, img2):
    img1 = cv2.resize(img1, (224, 224))
    img2 = cv2.resize(img2, (224, 224))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ss = SSIM(img1, img2)
    return ss
