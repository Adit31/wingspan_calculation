import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import streamlit as st
import tempfile

pose_landmark_model_path = 'pose_landmarker_full.task'
hand_landmark_model_path = 'hand_landmarker.task'

def draw_pose_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

def calculate_palm(image_path_palm):
    try:
        base_options = python.BaseOptions(model_asset_path=hand_landmark_model_path)
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                            num_hands=2)
        detector = vision.HandLandmarker.create_from_options(options)
    
        image = mp.Image.create_from_file(image_path_palm)
        height, width = image.height, image.width
    
        detection_result = detector.detect(image)
    
        middle_finger_tip = detection_result.hand_landmarks[0][12]
        wrist = detection_result.hand_landmarks[0][0]
    
        middle_finger_tip_x = middle_finger_tip.x * width
        middle_finger_tip_y = middle_finger_tip.y * height
    
        wrist_x = wrist.x * width
        wrist_y  = wrist.y * height
        palm_size_pixels = np.sqrt((middle_finger_tip_x - wrist_x)**2 + (middle_finger_tip_y - wrist_y)**2)
    
        return palm_size_pixels
    except IndexError:
        st.write("The application requires at least one arm to be visible in the frame")
        return 0

def calculate_wingspan(image_path_wingspan, length_inches, length_pixels):
    try:
        base_options = python.BaseOptions(model_asset_path=pose_landmark_model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True)
        detector = vision.PoseLandmarker.create_from_options(options)
    
        image = mp.Image.create_from_file(image_path_wingspan)
        height, width = image.height, image.width
    
        detection_result = detector.detect(image)
    
        annotated_image = draw_pose_landmarks_on_image(image.numpy_view(), detection_result)
    
        left_wrist = detection_result.pose_landmarks[0][15]
        right_wrist = detection_result.pose_landmarks[0][16]
    
        x_left_wrist = left_wrist.x *width
        y_left_wrist = left_wrist.y *height
    
        x_right_wrist = right_wrist.x *width
        y_right_wrist = right_wrist.y *height
    
        wingspan_pixels = np.sqrt((x_right_wrist - x_left_wrist)**2 + (y_right_wrist - y_left_wrist)**2 )
    
        palm_pixels = calculate_palm(image_path_wingspan)
    
        total_wingspan_pixels = wingspan_pixels + 2*palm_pixels

        nose = detection_result.pose_landmarks[0][0]
        left_eye = detection_result.pose_landmarks[0][1]
        right_eye = detection_result.pose_landmarks[0][4]
        left_heel = detection_result.pose_landmarks[0][29]
        right_heel = detection_result.pose_landmarks[0][30]

        y_eye = (left_eye.y + right_eye.y)/2 *height
        y_heel = (left_heel.y + right_heel.y)/2 *height
        forehead = 1.5 * abs(nose.y*height - y_eye)

        total_height_pixels = abs(y_eye - y_heel) + forehead
    
        pixels_to_inches = length_inches / length_pixels
        total_wingspan_inches = total_wingspan_pixels * pixels_to_inches
        total_height_inches = total_height_pixels * pixels_to_inches
    
        return total_wingspan_inches, total_height_inches, annotated_image
    except IndexError:
        st.write("The application requires at least one arm to be visible in the frame")
        return 0, 0, image_path_wingspan

st.title("Calculate Wingspan")
uploaded_file = st.camera_input("Take a picture")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name
    length_inches = st.number_input("Enter the static height in inches:", value=1.0)
    length_pixels = st.number_input("Enter the static height in pixels:", value=2.5)

    wingspan, height, annotated_image = calculate_wingspan(temp_path, length_inches, length_pixels)
    
    st.image(annotated_image, caption="Annotated Image", use_column_width=True)
    st.write("Wingspan is:", wingspan, "inches")
    st.write("Height is:", height, "inches")
