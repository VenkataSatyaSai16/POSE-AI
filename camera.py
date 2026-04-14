import cv2


class CameraManager:
    """Simple webcam wrapper for consistent frame capture and cleanup."""

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam. Check camera permissions or index.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        return frame

    def release(self):
        if self.cap:
            self.cap.release()
