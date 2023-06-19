from main import *


def test():
    """
    Yüz tespiti sınıfını kullanarak test işlemi gerçekleştirildi.
    """

    detect = FaceDetectionLayer()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        _, test_frame = cap.read()

        face = detect.face_det(test_frame)

        for (x, y, w, h) in face:
            cv2.rectangle(test_frame, (x, y), (x + w, y + h), (0, 200, 50), 2)

        cv2.imshow("Test", test_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test()
