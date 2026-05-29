import cv2

# Load your image
img = cv2.imread("B.png", cv2.IMREAD_GRAYSCALE)

# Set up the detector exactly like your robot
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(aruco_dict)

# Scan the image
corners, ids, _ = detector.detectMarkers(img)

if ids is not None:
    val = int(ids[0][0])
    x = (val >> 4) & 0x0F
    y = val & 0x0F
    print(f"Tag ID Detected: {val} (Hex: 0x{val:02X})")
    print(f"Location Cell: ({x}, {y})")
else:
    print("Could not read the AR tag.")