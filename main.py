import os
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation

# Load the background images
bg_images_path = "images"
bg_image_files = os.listdir(bg_images_path)
bg_images = [cv2.imread(os.path.join(bg_images_path, img)) for img in bg_image_files]

# Initialize video capture and segmentation
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
sugmentor = SelfiSegmentation()

indexImage = 0

while True:
    print("Index: ", indexImage)
    success, img = cap.read()
    imgOut = sugmentor.removeBG(img, bg_images[indexImage], cutThreshold=0.4)

    # Calculate FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    imgsStacked = cvzone.stackImages([img, imgOut], 2, 1)
    cv2.imshow("Images Stacked", imgsStacked)
    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):  # Press 'q' to exit the loop
        break
    elif key == ord('p'):  # Press 'p' to have following bg
        indexImage = (indexImage + 1) % len(bg_images)
    elif key == ord('m'):  # Press 'm' to have previous bg
        indexImage = (indexImage - 1) % len(bg_images)

cap.release()
cv2.destroyAllWindows()
