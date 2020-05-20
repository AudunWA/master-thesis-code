import cv2
import numpy as np
import matplotlib.pyplot as plt


def generate_shadow_coordinates(imshape, no_of_shadows=1):
    vertices_list = []
    for index in range(no_of_shadows):
        vertex = []
        for dimensions in range(np.random.randint(3, 15)):  ## Dimensionality of the shadow polygon
            vertex.append((imshape[1] * np.random.uniform(), imshape[0] // 3 + imshape[0] * np.random.uniform()))
        vertices = np.array([vertex], dtype=np.int32)  ## single shadow vertices
        vertices_list.append(vertices)
    return vertices_list  ## List of shadow vertices


def generate_random_lines(imshape, slant, drop_length):
    drops = []
    for i in range(1500):
        ## If You want heavy rain, try increasing this
        if slant < 0:
            x = np.random.randint(slant, imshape[1])
        else:
            x = np.random.randint(0, imshape[1] - slant)
        y = np.random.randint(0, imshape[0] - drop_length)

        drops.append((x, y))
    return drops


def change_light(bgr_img, light_coeff, debug=False):
    lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    lab_img[:, :, 0] = np.minimum(lab_img[:, :, 0] * light_coeff, 255)
    bgr_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
    if debug:
        show_image(bgr_img)
    return np.array(bgr_img, dtype=np.uint8)


def change_hue(bgr_img, hue_coeff, debug=False):
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    hsv_img[:, :, 0] = (hsv_img[:, :, 0] * hue_coeff) % 255

    bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    if debug:
        show_image(bgr_img)
    return np.array(bgr_img, dtype=np.uint8)


def add_rain(bgr_img, debug=False):
    bgr_img = np.copy(bgr_img)
    imshape = bgr_img.shape
    slant_extreme = 10
    slant = 0  # np.random.randint(-slant_extreme,slant_extreme)
    drop_length = 7
    drop_width = 1
    drop_color = (200, 200, 200)  ## a shade of gray
    rain_drops = generate_random_lines(imshape, slant, drop_length)
    for rain_drop in rain_drops:
        cv2.line(bgr_img, (rain_drop[0], rain_drop[1]), (rain_drop[0] + slant, rain_drop[1] + drop_length), drop_color,
                 drop_width)
    bgr_img = cv2.blur(bgr_img, (3, 3))  ## rainy view are blurry
    if debug:
        show_image(bgr_img)
    return np.array(bgr_img, dtype=np.uint8)


def add_shadow(bgr_img, no_of_shadows=1, debug=False):
    lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    mask = np.zeros_like(lab_img)
    img_shape = lab_img.shape
    vertices_list = generate_shadow_coordinates(img_shape, no_of_shadows)  # 3 getting list of shadow vertices
    for vertices in vertices_list:
        cv2.fillPoly(mask, vertices,
                     255)  ## adding all shadow polygons on empty mask, single 255 denotes only red channel

    lab_img[:, :, 0][mask[:, :, 0] == 255] = lab_img[:, :, 0][mask[:, :,
                                                              0] == 255] * 0.2  ## if red channel is hot, image's "Lightness" channel's brightness is lowered
    bgr_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
    if debug:
        show_image(bgr_img)
    return np.array(bgr_img, dtype=np.uint8)


def gaussian_blur(bgr_img, amount=3, debug=False):
    bgr_img = cv2.GaussianBlur(bgr_img, (amount, amount), 0)
    if debug:
        show_image(bgr_img)
    return np.array(bgr_img, dtype=np.uint8)

def show_image(bgr_img):
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_img)
    plt.show()

if __name__ == '__main__':
    test_path = "/hdd/audun/master-thesis-code/prediction_pipeline/data/carla_data/easy_traffic_lights_clear/2020-02-26_11-50-08/imgs/forward_center_rgb_00000019.png"
    bgr_img = cv2.imread(test_path)
    print("Original image")
    show_image(bgr_img)

    print("Gaussian")
    blur_img = gaussian_blur(bgr_img, amount=11, debug=True)

    print("Rain")
    rain_img = add_rain(bgr_img, debug=True)

    print("Shadows")
    shadow_img = add_shadow(bgr_img, debug=True)

    print("Lightness")
    light_img = change_light(bgr_img, 1.1, debug=True)

    print("Hue")
    hue_img = change_hue(bgr_img, 1.5, debug=True)