from cail.calibrator import Calibration

c = Calibration("C:\\rasp")

print(c.distortion)
print(c.cameraMatrix)