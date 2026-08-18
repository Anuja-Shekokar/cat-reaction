import cv2
import mediapipe as mp
import math
from collections import deque, Counter


# ==========================================
# MEDIAPIPE SETUP
# ==========================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def distance(a, b):
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2
    )


def finger_extended(hand, tip, pip):
    return (
        distance(hand[tip], hand[0])
        > distance(hand[pip], hand[0]) * 1.15
    )


# ==========================================
# GESTURE RECOGNITION
# ==========================================

def recognize_gesture(hand):

    index = finger_extended(hand, 8, 6)
    middle = finger_extended(hand, 12, 10)
    ring = finger_extended(hand, 16, 14)
    pinky = finger_extended(hand, 20, 18)

    thumb = (
        distance(hand[4], hand[0])
        > distance(hand[3], hand[0]) * 1.20
    )

    # --------------------------------------
    # SHAKA 🤙
    # --------------------------------------

    if (
        thumb
        and pinky
        and not index
        and not middle
        and not ring
    ):
        return "SHAKA"


    # --------------------------------------
    # INDEX-ONLY GESTURES
    # --------------------------------------

    if (
        index
        and not middle
        and not ring
        and not pinky
    ):

        index_tip = hand[8]
        index_mcp = hand[5]

        dx = abs(index_tip.x - index_mcp.x)
        dy = abs(index_tip.y - index_mcp.y)

        # Index pointing UP
        if dy > dx * 1.20:
            return "SHUSH"

        # Index pointing SIDEWAYS
        if dx > dy * 1.20:
            return "POINT"


    # --------------------------------------
    # THUMBS UP 👍
    # --------------------------------------

    if (
        thumb
        and not index
        and not middle
        and not ring
        and not pinky
    ):

        if hand[4].y < hand[3].y:
            return "THUMBS UP"


    # --------------------------------------
    # NO GESTURE
    # --------------------------------------

    return "NO GESTURE"


# ==========================================
# CAT IMAGE PATHS
# ==========================================

cat_paths = {
    "NO GESTURE": "cats/default.png",
    "SHUSH": "cats/shush.png",
    "SHAKA": "cats/shaka.png",
    "POINT": "cats/point.png",
    "THUMBS UP": "cats/thumbs_up.png"
}


# ==========================================
# LOAD ALL CAT IMAGES ONCE
# ==========================================

cat_images = {}

for gesture, path in cat_paths.items():

    image = cv2.imread(path)

    if image is None:
        print("ERROR: Could not load:", path)
        exit()

    cat_images[gesture] = image


# Start with default cat
reaction_image = cat_images["NO GESTURE"]


# ==========================================
# CREATE WINDOWS
# ==========================================

cv2.namedWindow(
    "Camera",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Camera",
    640,
    480
)

cv2.moveWindow(
    "Camera",
    50,
    100
)


cv2.namedWindow(
    "Cat Reaction",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Cat Reaction",
    640,
    640
)

cv2.moveWindow(
    "Cat Reaction",
    750,
    50
)


# ==========================================
# CAMERA
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not access camera.")
    exit()


# ==========================================
# GESTURE SMOOTHING
# ==========================================

gesture_history = deque(maxlen=5)

stable_gesture = "NO GESTURE"


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    # --------------------------------------
    # READ CAMERA
    # --------------------------------------

    success, frame = camera.read()

    if not success:
        print("ERROR: Could not read camera frame.")
        break


    # Mirror camera
    frame = cv2.flip(frame, 1)


    # --------------------------------------
    # CONVERT CAMERA TO RGB
    # --------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------
    # CREATE MEDIAPIPE IMAGE
    # --------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------------
    # DETECT HAND
    # --------------------------------------

    result = landmarker.detect(mp_image)


    current_gesture = "NO GESTURE"


    # --------------------------------------
    # PROCESS HAND
    # --------------------------------------

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        current_gesture = recognize_gesture(hand)

        # NO GREEN DOTS HERE
        # Hand detection happens invisibly.


    # --------------------------------------
    # GESTURE SMOOTHING
    # --------------------------------------

    gesture_history.append(
        current_gesture
    )

    counts = Counter(
        gesture_history
    )

    most_common_gesture, count = (
        counts.most_common(1)[0]
    )


    # Change only when 3 of the last 5
    # frames agree

    if count >= 3:

        stable_gesture = most_common_gesture


    # --------------------------------------
    # UPDATE CAT
    # --------------------------------------

    reaction_image = cat_images[
        stable_gesture
    ]


    # ======================================
    # CAMERA WINDOW
    # ======================================

    # Clean camera feed.
    # No text.
    # No green dots.

    cv2.imshow(
        "Camera",
        frame
    )


    # ======================================
    # CAT REACTION WINDOW
    # ======================================

    cv2.imshow(
        "Cat Reaction",
        reaction_image
    )


    # ======================================
    # QUIT
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

camera.release()

cv2.destroyAllWindows()