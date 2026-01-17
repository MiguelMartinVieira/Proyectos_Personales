import cv2
import os
import argparse

def capture_images(output_dir, quantity=20, grid_size=(9, 6)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    count = 0
    print(f"Press 'C' to capture image. Need {quantity} images.")
    print("Press 'Q' to quit.")

    while count < quantity:
        ret, frame = cap.read()
        if not ret:
            break
        
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find chess board corners
        ret_corners, corners = cv2.findChessboardCorners(gray, grid_size, None)

        if ret_corners:
            cv2.drawChessboardCorners(display_frame, grid_size, corners, ret_corners)
            cv2.putText(display_frame, "PATTERN DETECTED! Press 'C'", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Pattern NOT detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(display_frame, f"Images Captured: {count}/{quantity}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('Calibration Capture', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if ret_corners:
                filename = os.path.join(output_dir, f"calib_{count:02d}.jpg")
                cv2.imwrite(filename, frame) # Save original frame, not the one with drawn corners
                print(f"Saved {filename}")
                count += 1
                # Visual feedback
                cv2.rectangle(display_frame, (0,0), (640,480), (255,255,255), 10)
                cv2.imshow('Calibration Capture', display_frame)
                cv2.waitKey(200)
            else:
                print("Cannot capture: Pattern not detected.")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    if count >= quantity:
        print("Capture complete!")
    else:
        print(f"Capture incomplete. Saved {count} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Capture images for camera calibration.')
    parser.add_argument('--dir', type=str, default='captured_images', help='Output directory for images')
    args = parser.parse_args()
    
    capture_images(args.dir)
