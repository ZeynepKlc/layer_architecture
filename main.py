import cv2


class UserInterfaceLayer:

    """
    Kullanıcı arayüzü katmanında webcami çalıştırmak için gerekli kodlar yazıldı.
    Ekran açma kapama, serbest bırakma gibi kullanıcıyla etkileşimde bulunduğumuz katman
     olarak videoyu yakalama ve okuma adımları işlendi.
    """

    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    def load_webcam(self):
        ret, frame = self.cap.read()
        return frame

    def show(self, frame):
        cv2.imshow("Webcam", frame)

    def close_screen(self):
        self.cap.release()
        cv2.destroyAllWindows()


class FaceDetectionLayer:

    """
    Yüz tespiti için hazır bir cascade kullanılarak webcamde yüz algılandığında etrafına
    bir dikdörtgen çizilecek senaryo oluşturuldu.

    """

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier("frontalface.xml")

    def face_det(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 2)
        return faces


class EyeDetectionLayer:

    """
    Webcamde algılanan yüzdeki gözleri tespit eden katman oluşturul.
    """

    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier("eye.xml")

    def eye_det(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray, 1.3, 3)

        return eyes


if __name__ == "__main__":
    ui = UserInterfaceLayer()
    face_detect = FaceDetectionLayer()
    eye_detect = EyeDetectionLayer()

    while True:
        frame = ui.load_webcam()

        faces = face_detect.face_det(frame)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            roi_gray = frame[y:y + h, x:x + w]
            eyes = eye_detect.eye_det(roi_gray)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)

        ui.show(frame)

        if cv2.waitKey(1) == ord('q'):
            break

    ui.close_screen()
