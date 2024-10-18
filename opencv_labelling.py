import cv2
import pytesseract
import os

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load Haar Cascade for number plates
plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# Open video
cap = cv2.VideoCapture('In-Transit Vehicle Plate Captures [001]_3.mp4')

# Dictionary to store extracted frames for each car
car_frames = {}
car_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect number plates
    plates = plate_cascade.detectMultiScale(blurred, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in plates:
        # Crop the number plate from the frame
        plate_img = frame[y:y + h, x:x + w]

        # Use Tesseract to extract text
        text = pytesseract.image_to_string(plate_img, config='--psm 8').strip()

        if text:
            # Create folder for the detected car if it doesn't exist
            car_id = f"car_{car_counter}"
            if car_id not in car_frames:
                os.makedirs(car_id)
                car_frames[car_id] = 0  # Initialize frame count for this car

            # Save frame only if we have less than 10 for this car
            if car_frames[car_id] < 10:
                filename = f"{car_id}/frame_{car_frames[car_id]}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved {filename}")
                car_frames[car_id] += 1

            # Move to the next car after 10 frames are saved
            if car_frames[car_id] == 10:
                car_counter += 1

    # Display the current frame (Optional)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
