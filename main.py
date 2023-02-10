import cv2
import face_recognition
import os


# drawing a rectangle with right color
def draw_box(frame, left, top, right, bottom, color, name):
    font_color = (0, 255, 0) if color == "green" else (0, 0, 255)
    # Draw a box around the face
    cv2.rectangle(frame, (left, top), (right, bottom), font_color, 2)
    # Draw a label with a name below the face
    cv2.rectangle(frame, (left, bottom - 25), (right, bottom), font_color, cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (0, 0, 0), 1)


dataset_faces = []
dataset = os.listdir("images")
names = {}

# encode the faces from dataset
i = 0
for img in dataset:
    if img == '.DS_Store':
        continue
    human_recognition = face_recognition.load_image_file(
        "/Users/daniel5577/PycharmProjects/Face_Recognition/images/" + img)
    human_encoding = face_recognition.face_encodings(human_recognition)[0]
    dataset_faces.append(human_encoding)
    names[i] = img[:img.find(".")]
    i += 1

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # cv2.imshow("frame", cv2.flip(frame, 1))

    # recognize the faces
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # comparing faces with dataset
    finded_human_names = []
    for face_encoding in face_encodings:
        results = face_recognition.compare_faces(dataset_faces, face_encoding, tolerance=0.50)
        if True in results:
            for j in range(len(results)):
                if results[j] and (j in names.keys()):
                    finded_human_names.append(names[j])
                    print(names[j])
                    break
        else:
            finded_human_names.append(None)
            print("Unknown person")

    # drawing rectangle with name
    for (top, right, bottom, left), name in zip(face_locations, finded_human_names):

        if name == None:
            draw_box(frame, left, top, right, bottom, "red", "Unknown person")
        else:
            draw_box(frame, left, top, right, bottom, "green", name)
        cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exec(0)
cap.release()
cv2.destroyAllWindows()
