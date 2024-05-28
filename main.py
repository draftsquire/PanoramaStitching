import numpy as np
import cv2
import matplotlib.pyplot as plt


def extract_features(img, feature_extractor=None):
    '''
    Extract features using SIFT by default
    :param feature_extractor: python object of feature extractor class (e.g. SIFT)
    :param img: original image
    :return: kp1, des1 - keypoints and descriptors for each keypoint respectively
    '''
    if feature_extractor is None:
        feature_extractor = cv2.SIFT_create()
    kp1, des1=feature_extractor.detectAndCompute(img, None)
    # kp1, des1 = feature_extractor.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), None)
    return kp1, des1


def get_matches(des1, des2):
    '''
    Get matches between two descriptor lists using k-nearest neighbour matcher and ratio test as per Lowe's paper
    :param des1: descriptors of 1st image
    :param des2: descriptors of 2nd image
    :return: "good" matches, tuple of matches in respect for each descriptor (for debug purposes)
    '''
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
    '''
    Estimate homography matrix to transform a 2nd image to 1st's image plane using RANSAC-based method
    :param kp1: Keypoints of 1st image
    :param kp2: Keypoints of 2nd image
    :param matches: mathces between two images descriptors
    :param threshold: Maximum allowed reprojection error to treat a point pair as an inlier, used by RANSAC
    :return: Homography matrix, mask for "good" keypoints (filtered by RANSAC)
    '''
    src_points = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_points = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, threshold)
    return H, mask


def warp_images(img1, img2, H):
    '''
    Warp two images and stich them together
    :param img1: 1st image
    :param img2: 2nd image
    :param H: Homography matrix
    :return: stitched image
    '''
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
    '''
    Stitch 2 images together using keypoints and descriptors
    :param img_1: first image
    :param img_2: second image
    :param kp1: 1st image's keypoints
    :param kp2: 2nd image's keypoints
    :param des1: 1st image's descriptors
    :param des2: 2nd image's descriptors
    :return: (stitched image, list of "good" matches between descriptors)
    '''
    good_points, good_matches = get_matches(des1, des2)
    H, mask = estimate_homography(kp1, kp2, good_points, threshold=5)
    mask = np.reshape(mask, (len(mask))).astype(bool)
    warped_img = warp_images(img_2, img_1, H)

    return warped_img, good_matches


def stitch_images(imgs):
    '''
    Stitch list of images together using SIFT to extract features
    :param imgs: list of images
    :return: stitched image, list of "good" matches,
    list of keypoints for each image, list of descriptors for each image
    '''
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
    image_paths = ['imgs/IMG_20240526_121013.jpg', 'imgs/IMG_20240526_121018.jpg', 'imgs/IMG_20240526_121021.jpg',
                   'imgs/IMG_20240526_121023.jpg']
    # initialized a list of images
    imgs = []

    for i in range(len(image_paths)):
        imgs.append(cv2.imread(image_paths[i]))
        imgs[i] = cv2.resize(imgs[i], (0, 0), fx=0.2, fy=0.2)
        imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)

    stitched_image, good_matches, keypoints, descriptors = stitch_images([imgs[0], imgs[1]])

    fig, axes = plt.subplots(2, 3)
    fig.canvas.manager.set_window_title("")
    fig.suptitle("", fontsize=20)
    axes[0][0].imshow(
        cv2.drawMatchesKnn(imgs[0], keypoints[0], imgs[1], keypoints[1], good_matches[0:50], None, flags=2),
        interpolation="none", norm=None, filternorm=False, cmap='gray')
    axes[1][0].imshow(stitched_image, interpolation="none", norm=None, filternorm=False)

    stitched_image_2, good_matches, keypoints, descriptors = stitch_images([stitched_image, imgs[2]])

    axes[0][1].imshow(
        cv2.drawMatchesKnn(stitched_image, keypoints[0], imgs[2], keypoints[1], good_matches[0:50], None, flags=2),
        interpolation="none", norm=None, filternorm=False, cmap='gray')
    axes[1][1].imshow(stitched_image_2, interpolation="none", norm=None, filternorm=False)

    # При перемене двух изображений местами, не удаётся нормально склеить
    stitched_image_3, good_matches, keypoints, descriptors = stitch_images([imgs[3], stitched_image_2])

    axes[0][2].imshow(
        cv2.drawMatchesKnn(imgs[3], keypoints[0], stitched_image_2, keypoints[1], good_matches[0:50], None, flags=2),
        interpolation="none", norm=None, filternorm=False, cmap='gray')
    axes[1][2].imshow(stitched_image_3, interpolation="none", norm=None, filternorm=False)
    fig.set_size_inches(10, 7)
    plt.show()
    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
