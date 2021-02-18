import cv2
import numpy as np


def sift_feature(modern_img_filename, historical_img_filename, return_type, image_show):

    # Many thanks to Pysource tech tutorial for publishing tutorials and the majority of the below code example.
    # https://pysource.com/2018/07/19/check-if-two-images-are-equal-with-opencv-and-python/
    # https://pysource.com/2018/07/20/find-similarities-between-two-images-with-opencv-and-python/

    # load images
    # https://pysource.com/2018/03/23/feature-matching-brute-force-opencv-3-4-with-python-3-tutorial-26/
    modern = cv2.imread(modern_img_filename, cv2.IMREAD_GRAYSCALE)
    historical = cv2.imread(historical_img_filename, cv2.IMREAD_GRAYSCALE)

    # show images
    if image_show:
        cv2.imshow("modern", modern)
        cv2.imshow("historical", historical)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # resample the modern image so it shares the same dimension as the historical photo
    # source: https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
    h, w = historical.shape
    modern = cv2.resize(modern, (w, h), interpolation=cv2.INTER_AREA)

    # calculate color difference and show image
    difference = cv2.subtract(historical, modern)
    if image_show:
        cv2.imshow("difference", difference)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # extract features
    sift = cv2.SIFT_create()
    kp_historical, desc_historical = sift.detectAndCompute(historical, None)
    kp_modern, desc_modern = sift .detectAndCompute(modern, None)

    # show image features
    # https://www.programcreek.com/python/example/89309/cv2.drawKeypoints
    historical_kp = cv2.drawKeypoints(historical, kp_historical, None)
    modern_kp = cv2.drawKeypoints(modern, kp_modern, None)
    if image_show:
        cv2.imshow("key points - historical", historical_kp)
        cv2.imshow("key points - modern", modern_kp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Brute force matches
    # http://datahacker.rs/feature-matching-methods-comparison-in-opencv/
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_historical, desc_modern, k=2)
    # print(len(matches))
    # print(matches)

    # Flann based image similarity
    # index_params = dict(algorithm=0, trees=5)
    # search_params = dict()
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(desc_historical, desc_modern, k=2)

    # reduce the number of matches to only those where the difference between match 1 and match 2 is large.
    # If the difference between match 1 and match 2 is large then there is increased confidence that match 1 is correct.
    # If match 1 and match 2 are close to each other, there is less certainty about whether match 1 or match 2 is more
    # accurate
    # http://datahacker.rs/feature-matching-methods-comparison-in-opencv/
    good_matches = []

    for m1, m2 in matches:
        if m1.distance < 0.85*m2.distance:
            good_matches.append([m1])

    img_match = cv2.drawMatchesKnn(historical, kp_historical, modern, kp_modern, good_matches, None, flags=2)
    if image_show:
        cv2.imshow("matches", img_match)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if return_type == "absolute":
        return len(good_matches)
    elif return_type == "percent":
        return len(good_matches) / len(kp_modern)


image_set = [["GrayNew", "GrayOld"], ["BaseSame", "BaseDiff"], ["LibertyNew", "LibertyOld"], ["MotoNew", "MotoOld"],
             ["NewRedHouse", "OldYellowHouse"], ["RailNew", "RailOld"], ["BaseSame", "BaseSame"]]

for i in image_set:
    similarity = sift_feature("images/"+i[0]+".png", "images/"+i[1]+".png", "absolute", False)
    i.append(similarity)

# https://stackoverflow.com/questions/17555218/python-how-to-sort-a-list-of-lists-by-the-fourth-element-in-each-list
image_set.sort(key=lambda x: x[2])

print(image_set)