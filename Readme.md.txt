Steps for camera calibration

1)Take multiple images of a chessboard from various angle and various distance and store them in a single folder(remove images with extreme angle or distance or blurred images as they will work as outliers and may increase the reprojection error).
2)Hardcode the dimension of chessboard inner corners if its differ from current one i:e 6,9.
3)After calculating all the coeffecients of distortion, they will be applied and the image will be undistorted.
4)To apply the undistortion on baseplate add the images to buildlayerimages