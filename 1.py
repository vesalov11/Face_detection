import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

# Initialize a counter for the number of people
people_count = 0

while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Update the people count based on the number of faces detected
    people_count = len(faces)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the people count on the image
    cv2.putText(img, f'People Count: {people_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                cv2.LINE_AA)

    # Display the image
    cv2.imshow('img', img)

    # Check for the 'Esc' key to exit the loop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
