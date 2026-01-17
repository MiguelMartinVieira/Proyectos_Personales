import numpy as np
import cv2
import glob
import argparse
import os

def calibrate_camera(image_dir, output_file="calibration_data.npz", grid_size=(9, 6), square_size=1.0):
    """
    Calibrates the camera using a set of checkerboard images.
    
    Args:
        image_dir (str): Directory containing the calibration images.
        output_file (str): Path to save the calibration data.
        grid_size (tuple): Number of inner corners per a chessboard row and column (cols, rows).
        square_size (float): Size of a square in your defined unit (e.g., mm, cm).
    """
    
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)
    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(os.path.join(image_dir, '*.jpg'))
    
    if not images:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(images)} images in {image_dir}. Starting processing...")
    
    valid_images = 0
    img_shape = None

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if img_shape is None:
            img_shape = gray.shape[::-1]

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, grid_size, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            valid_images += 1
            print(f"Processed {fname} - OK")
        else:
            print(f"Processed {fname} - FAILED to detect corners")

    if valid_images > 0:
        print(f"Calibrating with {valid_images} valid images...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

        print(f"Calibration successful. RMS Error: {ret}")
        print("Camera Matrix:\n", mtx)
        print("Distortion Coefficients:\n", dist)

        np.savez(output_file, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
        print(f"Calibration data saved to {output_file}")
    else:
        print("Not enough valid images for calibration.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='captured_images', help='Directory with images')
    parser.add_argument('--out', type=str, default='calibration_data.npz', help='Output file')
    args = parser.parse_args()
    
    calibrate_camera(args.dir, args.out)
