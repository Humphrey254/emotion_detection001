import cv2
from deepface import DeepFace

path = r'C:\Users\user\Downloads\emotion detector\IMG-20211119-WA0004.jpg'


# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml'
)

# read image
img = cv2.imread(path)

# Define the new width and height (e.g., 800 pixels width, 600 pixels height)
new_width = 300
new_height = 450

# Resize the image
resized_image = cv2.resize(img, (new_width, new_height))
#for saving  resized image 
cv2.imwrite('image1.jpg', resized_image)
# Convert into grayscale
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255,0), 2)

# Detect faces
faces = face_cascade.detectMultiScale(resized_image, 1.1, 4)
# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255,0), 2)



# show emoton 
def detect_emotion(path):
    emotion = DeepFace.analyze(path,actions=[ 'emotion'])
    
    return emotion
if __name__ == '__main__':
    print(detect_emotion('IMG-20211119-WA0004.jpg'))
 
cv2.imshow('my pic' ,resized_image)

cv2.waitKey(0)

cv2.destroyAllWindows()