import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.animation as animation
import scipy.signal as ss


def gaussian_kernel(sz=5, sigma=1) -> np.ndarray:
    """
    Функция для формирования ядра Гаусса
    :param sz: Размерность матрицы ядра
    :param sigma: СКО
    :return: np.array(sz,sz)  Искомая матрица
    """
    ax = np.linspace(-(sz - 1) / 2., (sz - 1) / 2., sz)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


# def sift(img_gray, print_octave=True):
#     '''
#     SIFT algorithm implementation
#     :param img_gray:
#     :param print_octave:
#     :return:
#     '''
#     # Detect keypoints
#     sigma = 1.6
#     k = np.sqrt(2)
#     n_scales = 5
#     g_kernel_sz = 10
#     octave_scales = []
#     DoGs = []
#     if print_octave:
#         fig_octave, axes = plt.subplots(2, 2)
#         fig_octave.canvas.manager.set_window_title("DoGs 1")
#         fig_octave.suptitle("DoGs 1", fontsize=20)
#
#     for i in range(n_scales):
#         sigma_octave = sigma * k * 2**i
#         octave_scales.append(ss.fftconvolve(img_gray, gaussian_kernel(g_kernel_sz, sigma_octave), mode='valid'))
#         if i > 0:
#             DoGs.append(np.subtract(octave_scales[i], octave_scales[i-1]))
#             if print_octave:
#                 if i-1 < 2:
#                     row_n = 0
#                     col_n = i - 1
#                 else:
#                     row_n = 1
#                     col_n = i - 1 - 2
#                 axes[row_n][col_n].imshow(DoGs[i-1], interpolation="none", norm=None, filternorm=False, cmap='gray')
#                 axes[row_n][col_n].title.set_text(r'$\sigma={0}$'.format(sigma_octave))
#     if print_octave:
#         plt.show()
#     pass


def extract_features(img, feature_extractor=None):
    '''
    Extract features using SIFT algo
    :param feature_extractor: python object of feature extractor class (e.g. SIFT)
    :param img: original image
    :return: kp1, des1 - keypoints and descriptors for each keypoint
    '''
    if feature_extractor is None:
        feature_extractor = cv2.SIFT_create()
    kp1, des1 = feature_extractor.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), None)
    return kp1, des1


def get_matches(des1, des2):
    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    matches = matcher.knnMatch(des1, des2, k=2)
    good_points = []
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_points.append(m)
            good_matches.append((m, n))
    return good_points, good_matches


def estimate_homography(kp1, kp2, matches, threshold=3):
    src_points = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_points = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, threshold)
    return H, mask


def warp_images(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners2 = cv2.perspectiveTransform(corners2, H)

    corners = np.concatenate((corners1, warped_corners2), axis=0)
    [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)

    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    warped_img2 = cv2.warpPerspective(img2, Ht @ H, (xmax - xmin, ymax - ymin))
    warped_img2[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img1

    return warped_img2


def stitch_2_imgs(img_1, img_2, kp1, kp2, des1, des2):
    good_points, good_matches = get_matches(des1, des2)
    H, mask = estimate_homography(kp1, kp2, good_points, threshold=5)
    warped_img = warp_images(img_2, img_1, H)
    return warped_img, good_matches


def stitch_images(imgs):
    sift = cv2.SIFT_create()
    keypoints = []
    descriptors = []
    for img in imgs:
        kp, des = extract_features(img, sift)
        keypoints.append(kp)
        descriptors.append(des)

    stitched_image, good_matches = stitch_2_imgs(imgs[0], imgs[1], keypoints[0], keypoints[1],
                                                 descriptors[0], descriptors[1])

    return stitched_image, good_matches, keypoints, descriptors
def main():
    image_paths = ['imgs/IMG_20240526_121013.jpg', 'imgs/IMG_20240526_121018.jpg', 'imgs/IMG_20240526_121021.jpg', 'imgs/IMG_20240526_121023.jpg']
    # initialized a list of images
    imgs = []

    for i in range(len(image_paths)):
        imgs.append(cv2.imread(image_paths[i]))
        imgs[i] = cv2.resize(imgs[i], (0, 0), fx=0.2, fy=0.2)
        imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)

    stitched_image, good_matches,  keypoints, descriptors = stitch_images([imgs[0], imgs[1]])

    fig, axes = plt.subplots(2, 3)
    fig.canvas.manager.set_window_title("")
    fig.suptitle("", fontsize=20)
    axes[0][0].imshow(cv2.drawMatchesKnn(imgs[0], keypoints[0], imgs[1], keypoints[1], good_matches[0:50], None, flags=2),
                   interpolation="none", norm=None, filternorm=False, cmap='gray')
    axes[1][0].imshow(stitched_image, interpolation="none", norm=None, filternorm=False)

    stitched_image_2, good_matches, keypoints, descriptors = stitch_images([stitched_image, imgs[2]])

    axes[0][1].imshow(cv2.drawMatchesKnn(stitched_image, keypoints[0], imgs[2], keypoints[1], good_matches[0:50], None, flags=2),
                    interpolation="none", norm=None, filternorm=False, cmap='gray')
    axes[1][1].imshow(stitched_image_2, interpolation="none", norm=None, filternorm=False)

    # При перемене двух изображений местами, не удаётся нормально склеить
    stitched_image_3, good_matches, keypoints, descriptors = stitch_images([imgs[3], stitched_image_2])

    axes[0][2].imshow(
        cv2.drawMatchesKnn(imgs[3], keypoints[0],stitched_image_2, keypoints[1], good_matches[0:50], None, flags=2),
        interpolation="none", norm=None, filternorm=False, cmap='gray')
    axes[1][2].imshow(stitched_image_3, interpolation="none", norm=None, filternorm=False)
    fig.set_size_inches(10, 7)
    plt.show()
    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
