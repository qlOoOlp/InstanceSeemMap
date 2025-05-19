#%%
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#%%
class RoomSegProcessor:
    def __init__(self, root_dir, scene_id, version, save_dir):
        self.root_dir = root_dir
        self.scene_id = scene_id
        self.version = version
        self.save_dir = save_dir
        self.data_dir = os.path.join(root_dir, f"{scene_id}/map/{scene_id}_{version}")
        self.xmin = 0
        self.xmax = 0
    
    def load_map(self, load_path):
        with open(load_path, "rb") as f:
            map = np.load(f, allow_pickle=True)
        return map
    
    def crop(self, npy):
        return npy[self.xmin:self.xmax+1, self.ymin:self.ymax+1]
    
    def convert_to_255(self, image):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j]==0:
                    image[i][j] = 255
                else:
                    image[i][j] = 0
        return image
    
    def save_image(self, image, name):
        path = os.path.join(self.save_dir, name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        cv2.imwrite(path, image)

    def process(self):
        obs_npy = self.load_map(os.path.join(self.data_dir,f'obstacles_{self.version}.npy'))
        x_indices, y_indices = np.where(obs_npy == 1)
        self.xmin = np.min(x_indices)
        self.xmax = np.max(x_indices)
        self.ymin = np.min(y_indices)
        self.ymax = np.max(y_indices)
        
        # 1. Load mask maps
        obs_npy = self.crop(obs_npy)
        wall_npy = self.crop(self.load_map(os.path.join(self.data_dir,f'wall_mask_{self.version}.npy')))
        door_npy = self.crop(self.load_map(os.path.join(self.data_dir,f'door_mask_{self.version}.npy')))
        window_npy = self.crop(self.load_map(os.path.join(self.data_dir,f'window_mask_{self.version}.npy')))
        others_npy = self.crop(self.load_map(os.path.join(self.data_dir,f'others_mask_{self.version}.npy')))

        # 2. Wall candidates for room segmentation
        intersection = np.logical_and(wall_npy, obs_npy).astype(np.uint8)
        all_wall = np.logical_or(wall_npy, np.logical_or(door_npy, window_npy)).astype(np.uint8)

        # 3. Save results
        self.save_image(self.convert_to_255(all_wall), 'all_wall.png')
        self.save_image(self.convert_to_255(intersection), 'intersection.png')
        self.save_image(self.convert_to_255(wall_npy.astype(np.uint8)), 'wall.png')

class RoomSegmentation:
    def __init__(self, m=1, b=5, min_dist=1600, min_l=100, kernel_size=5, image_dir='', image_name=''):
        self.m_thresh = m
        self.b_thresh = b
        self.l_thresh = min_dist
        self.first_line_thresh = min_l # 10 / 50 / 100
        self.kernel_size = kernel_size
        
        self.dir = os.path.join(image_dir, image_name.split(".")[0])  # '/home/home/1126/2t7WUuJeko7_2/B'
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)
        print(self.dir)
        self.img = cv2.imread(os.path.join(image_dir, image_name), 0)
        print(np.unique(self.img))

        self.original_img = self.img.copy()
            
    @staticmethod
    def convert_to_sqr(img):
        row, col = img.shape
        print(row, col)
        if row > col:
            padding = (row - col) // 2
            zero_side_image = np.zeros((row, (row - col) // 2), np.uint8)
            new_image = np.concatenate((zero_side_image, img, zero_side_image), axis=1)
            padding_info = ('horizontal', padding)
        elif col > row:
            padding = (col - row) // 2
            zero_side_image = np.zeros(((col - row) // 2, col), np.uint8)
            new_image = np.concatenate((zero_side_image, img, zero_side_image), axis=0)
            padding_info = ('vertical', padding)
        else:
            new_image = img
            padding_info = None
        return new_image, padding_info

    def run(self):

        # Step1. 정사각형으로 맞춰주는 작업
        self.img, padding_info = self.convert_to_sqr(self.img)
        # self.img = cv2.bitwise_not(self.img)
        print(f"{np.unique(self.img)}")
        cv2.imwrite(self.dir + "/0-SquareImage.png", self.img)
        print(self.img.shape)

        detected_lines = self.line_segmentation(self.img, '/1-DetectedLines')
        extended_lines = self.extend_lines(detected_lines)
        extended_lined_img = cv2.createLineSegmentDetector(1).drawSegments(self.img, extended_lines)
        cv2.imwrite(self.dir + "/2-ExtendedLines.png", extended_lined_img)

        lined_img = cv2.cvtColor(extended_lined_img, cv2.COLOR_BGR2GRAY)

        rotated_img = self.rotate_img(lined_img, 90)
        detected_lines = self.line_segmentation(rotated_img, '/3-DetectedLines')
        extended2_lines = self.extend_lines(detected_lines)
        extended2_lined_img = cv2.createLineSegmentDetector(1).drawSegments(rotated_img, extended2_lines)
        cv2.imwrite(self.dir + "/4-ExtendedLines.png", extended2_lined_img)

        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        lined_img = cv2.cvtColor(extended2_lined_img, cv2.COLOR_BGR2GRAY)
        # lined_img = self.convert_to_bw(lined_img, 200)
        lined_img = cv2.erode(lined_img, kernel, iterations=1)
        lined_img = cv2.morphologyEx(lined_img, cv2.MORPH_OPEN, kernel, iterations=5)
        cv2.imwrite(self.dir + "/5-MorphologyErodeOpen.png", lined_img)

        # circles = self.circle_detection(lined_img)

        second_rotated_img = self.rotate_img(lined_img, 270)
        components = self.connected_component(second_rotated_img)

        # original = self.transfer_to_original(components)
        cv2.imwrite(self.dir + '/6-output.png', components)

        def revert_to_original(img, padding_info):
            if padding_info is None:
                return img  # No padding was added, so no changes are needed
            
            padding_type, padding = padding_info
            if padding_type == 'horizontal':
                return img[:, padding:-padding]
            elif padding_type == 'vertical':
                return img[padding:-padding, :]

        restored_img = revert_to_original(components, padding_info)
        cv2.imwrite(self.dir + '/7-restore_output.png', restored_img)

    @staticmethod
    def circle_detection(image):
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20,
                                param1=50, param2=30, minRadius=50, maxRadius=0)
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 0), 2)
        return image

    @staticmethod
    def convert_to_bw(image, thresh):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j] > thresh:
                    image[i][j] = 255
                else:
                    image[i][j] = 0
        return image

    def line_segmentation(self, img, name):
        lsd = cv2.createLineSegmentDetector(1)
        lines = lsd.detect(img)[0]
        cv2.imwrite(self.dir + name + '.png', lsd.drawSegments(img, lines))
        detected_lines = []
        for line in lines:
            x1 = line[0][0]
            y1 = line[0][1]
            x2 = line[0][2]
            y2 = line[0][3]
            if (x2 - x1) ** 2 + (y2 - y1) ** 2 < self.first_line_thresh:
                continue
            if x2 - x1 == 0:
                continue
            m = (y2 - y1) / (x2 - x1)
            b = y2 - m * x2
            detected_lines.append([m, b, (x1, y1), (x2, y2)])
        return detected_lines

    def extend_lines(self, lines):
        temp_lines = []
        final_lines = []
        for i in range(len(lines)):
            for j in range(i, len(lines)):
                l1 = lines[i]
                l2 = lines[j]
                if abs(l1[0] - l2[0]) < self.m_thresh and \
                        abs(l1[1] - l2[1]) < self.b_thresh:
                    temp_lines.append([[l1[2][0], l1[2][1], l2[2][0], l2[2][1]]])
                    temp_lines.append([[l1[2][0], l1[2][1], l2[3][0], l2[3][1]]])
                    temp_lines.append([[l1[3][0], l1[3][1], l2[2][0], l2[2][1]]])
                    temp_lines.append([[l1[3][0], l1[3][1], l2[3][0], l2[3][1]]])
                    temp_lines.sort(key=lambda k: (k[0][0] - k[0][2]) ** 2 + (k[0][1] - k[0][3]) ** 2)
                    k = temp_lines[0]
                    if (k[0][0] - k[0][2]) ** 2 + (k[0][1] - k[0][3]) ** 2 < self.l_thresh:
                        final_lines.append(temp_lines[0])
                    temp_lines = []
        return np.asarray(final_lines)

    @staticmethod
    def draw_lines(img, lines, name):
        tmp = cv2.createLineSegmentDetector(1).drawSegments(img, lines)
        cv2.imshow(name, tmp)

    @staticmethod
    def connected_component(img):
        ret, labels = cv2.connectedComponents(img, connectivity=8, ltype=cv2.CV_16U)# 4)

        # Map component labels to hue val
        label_hue = np.uint8(360 * labels / np.max(labels))
        blank_ch = 255 * np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

        # cvt to BGR for display
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

        # set bg label to black
        labeled_img[label_hue == 0] = 0
        return labeled_img

    @staticmethod
    def rotate_img(img, theta):
        rows, cols = img.shape
        m = cv2.getRotationMatrix2D((cols / 2, rows / 2), theta, 1)
        dst = cv2.warpAffine(img, m, (cols, rows))
        return dst

    def transfer_to_original(self, final_img):
        sqr_org_img = self.convert_to_sqr(self.original_img)
        w, h = sqr_org_img.shape
        blank_ch = np.zeros((w, h, 3))
        for i in range(w):
            for j in range(h):
                if np.sum(final_img[i][j]) != 0:
                    blank_ch[i][j] = final_img[i][j]
                else:
                    val = sqr_org_img[i][j]
                    blank_ch[i][j] = [val, val, val]
        return blank_ch
    
if __name__ == '__main__':
    ROOT_DIR = '/nvme0n1/vlmaps/Data/habitat_sim/mp3d'
    SCENE_ID = '2t7WUuJeko7_2' # '2t7WUuJeko7'
    VERSION = 'room_seg1_floor_prior' # 'room_seg1_gt_floor_t'
    SAVE_DIR = f'/nvme0n1/eunae/room_seg/room_seg_mp3d/{SCENE_ID}'
    IMAGE_NAME = 'all_wall.png' # 'intersection.png' # 'all_wall.png'

    # Extract Wall candidates
    RoomSegProcessor(ROOT_DIR, SCENE_ID, VERSION, SAVE_DIR).process()

    # Make Room Segmentation
    RoomSegmentation(m=1, b=70, min_dist=1600, min_l=100, kernel_size=5, image_dir=SAVE_DIR,
                               image_name=IMAGE_NAME).run()
