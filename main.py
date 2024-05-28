import numpy
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
    kp1, des1 = feature_extractor.detectAndCompute(img, None)
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


def stitch_2_images(img1, img2):
    '''
    Stitch list of images together using SIFT to extract features
    :param imgs: list of images
    :return: stitched image, list of "good" matches,
    list of keypoints for each image, list of descriptors for each image
    '''
    sift = cv2.SIFT_create()

    kp1, des1 = extract_features(img1, sift)
    kp2, des2 = extract_features(img2, sift)

    good_points, good_matches = get_matches(des1, des2)
    H, mask = estimate_homography(kp1, kp2, good_points, threshold=5)
    mask = np.reshape(mask, (len(mask))).astype(bool)
    stitched_image = warp_images(img2, img1, H)

    return stitched_image, good_matches


def main():
    image_paths = ['imgs/IMG_20240526_121013.jpg', 'imgs/IMG_20240526_121018.jpg', 'imgs/IMG_20240526_121021.jpg']
    # initialized a list of images
    imgs = []

    for i in range(len(image_paths)):
        imgs.append(cv2.imread(image_paths[i]))
        imgs[i] = cv2.resize(imgs[i], (0, 0), fx=0.2, fy=0.2)
        imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)

    for i in range(1, len(imgs)):
        stitched_image, good_matches = stitch_2_images(imgs[i-1], imgs[i])
        imgs[i] = stitched_image
    plt.figure()
    plt.imshow(stitched_image, interpolation="none", norm=None, filternorm=False)
    plt.title("Image stitching example")
    plt.show()

    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
