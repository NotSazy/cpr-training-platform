import sys
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sys import platform

# Configuration
file_location = "C:/School/Thesis/Females/221/Front/"
act = "3"
file_name = "221.mp4"
direction = "Right"

# Action and direction based point selection
if act == "1":
    if direction == "Left":
        p1, p2, p3, p4 = 5, 6, 7, 0
    else:
        p1, p2, p3, p4 = 2, 3, 4, 0
elif act == "3":
    if direction == "Left":
        p1, p2, p3, p4 = 5, 6, 7, 12
    else:
        p1, p2, p3, p4 = 2, 3, 4, 9
elif act in ["4", "5", "6", "7", "8"]:
    if direction == "Left":
        p1, p2, p3, p4 = 5, 6, 7, 0
    else:
        p1, p2, p3, p4 = 2, 3, 4, 0

# OpenPose import and setup
openpose_dir = "C:/School/openpose-master"
sys.path.append(openpose_dir + '/build/python/openpose/Release')
os.environ['PATH'] += ';' + openpose_dir + '/build/x64/Release;' + openpose_dir + '/bin;'

try:
    import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found.')
    raise e

params = {
    "model_folder": openpose_dir + "/models/",
    "net_resolution": "320x320",
    "number_people_max": 1
}

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()


# Helper functions
def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth


def calculate_angle(vector_a, vector_b):
    ang_a = np.arctan2(*vector_a[::-1])
    ang_b = np.arctan2(*vector_b[::-1])
    ans = np.rad2deg((ang_a - ang_b) % (2 * np.pi))
    return ans if ans < 180 else 360 - ans


# Start processing
try:
    datum = op.Datum()
    cap = cv2.VideoCapture(os.path.join(file_location, file_name))

    if not cap.isOpened():
        raise Exception(f"Error: Could not open video file: {os.path.join(file_location, file_name)}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video_path = os.path.join(file_location, "output_video.avi")
    if not os.path.exists(file_location):
        os.makedirs(file_location)

    # Try different codecs if 'XVID' fails
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

    if not out.isOpened():
        print(f"Error: VideoWriter could not be opened with codec 'XVID'. Trying 'MJPG' codec.")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

    if not out.isOpened():
        raise Exception(f"Error: VideoWriter could not be opened with path: {output_video_path}")

    body_keypoints_df = pd.DataFrame()

    while cap.isOpened():
        has_frame, frame = cap.read()
        if not has_frame:
            break

        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        if datum.poseKeypoints is not None and len(datum.poseKeypoints) > 0:
            temp_df = pd.DataFrame(datum.poseKeypoints[0])
            body_keypoints_df = pd.concat([body_keypoints_df, temp_df], ignore_index=True)
            out.write(datum.cvOutputData)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Process DataFrame
    column_names = ['x', 'y', 'accuracy']
    if not body_keypoints_df.empty:
        body_keypoints_df.columns = column_names
        body_keypoints_df.to_csv(os.path.join(file_location, "body_keypoints.csv"), index=False)

    # Further processing for specific points
    point1 = body_keypoints_df.iloc[p1::25].copy()
    point2 = body_keypoints_df.iloc[p2::25].copy()
    point3 = body_keypoints_df.iloc[p3::25].copy()
    point4 = body_keypoints_df.iloc[p4::25].copy() if p4 != 0 else pd.DataFrame(columns=column_names)

    point1 = point1.reset_index(drop=True)
    point2 = point2.reset_index(drop=True)
    point3 = point3.reset_index(drop=True)
    point4 = point4.reset_index(drop=True)

    # Example of plotting the points
    plt.figure()
    plt.plot(point1['x'], height - point1['y'], 'r')
    plt.plot(point2['x'], height - point2['y'], 'g')
    plt.plot(point3['x'], height - point3['y'], 'b')
    if not point4.empty:
        plt.plot(point4['x'], height - point4['y'], 'y')
    plt.savefig(os.path.join(file_location, 'keypoints_plot.png'))
    plt.close()

except Exception as e:
    print(e)
    sys.exit(-1)

# Additional processing and scoring logic
try:
    # Pick the exact points
    point1 = body_keypoints_df.iloc[p1::25].copy()
    point2 = body_keypoints_df.iloc[p2::25].copy()
    point3 = body_keypoints_df.iloc[p3::25].copy()
    point4 = body_keypoints_df.iloc[p4::25].copy() if p4 != 0 else pd.DataFrame(columns=column_names)

    point1 = point1.reset_index(drop=True)
    point2 = point2.reset_index(drop=True)
    point3 = point3.reset_index(drop=True)
    point4 = point4.reset_index(drop=True)

    print("point1 columns:", point1.columns)
    print("point2 columns:", point2.columns)
    print("point3 columns:", point3.columns)
    print("point4 columns:", point4.columns)

    # Smooth the points
    S = 4
    pp1 = point1[S:][:-S].copy()
    pp2 = point2[S:][:-S].copy()
    pp3 = point3[S:][:-S].copy()
    pp4 = point4[S:][:-S].copy()
    pp1['x'] = smooth(point1['x'], 2 * S + 1)
    pp1['y'] = smooth(point1['y'], 2 * S + 1)
    pp2['x'] = smooth(point2['x'], 2 * S + 1)
    pp2['y'] = smooth(point2['y'], 2 * S + 1)
    pp3['x'] = smooth(point3['x'], 2 * S + 1)
    pp3['y'] = smooth(point3['y'], 2 * S + 1)
    if not pp4.empty:
        pp4['x'] = smooth(point4['x'], 2 * S + 1)
        pp4['y'] = smooth(point4['y'], 2 * S + 1)

    # Calculate angles and vectors
    e_angles = pd.DataFrame()
    for i in range(min(point1.shape[0], point2.shape[0], point3.shape[0])):
        if not point1.iloc[i].empty and not point2.iloc[i].empty and not point3.iloc[i].empty:
            es = np.array([point1.iloc[i].x, point1.iloc[i].y]) - np.array([point2.iloc[i].x, point2.iloc[i].y])
            eh = np.array([point3.iloc[i].x, point3.iloc[i].y]) - np.array([point2.iloc[i].x, point2.iloc[i].y])
            angle = calculate_angle(es, eh)
            e_angles = e_angles.append(pd.DataFrame([angle]))
    e_angles = e_angles.reset_index(drop=True)
    e_angles.columns = ['angle']

    # Plot elbow angleso
    plt.figure()
    plt.plot(e_angles.index, e_angles.angle, 'r')
    plt.savefig(os.path.join(file_location, 'e_angles.png'))
    plt.close()

    # Calculate shoulder to elbow vectors
    s_vect = pd.DataFrame()
    for i in range(min(point1.shape[0], point2.shape[0], point3.shape[0], point4.shape[0])):
        if not point1.iloc[i].empty and not point2.iloc[i].empty and not point3.iloc[i].empty and not point4.iloc[
            i].empty:
            es = np.array([point2.iloc[i].x, point2.iloc[i].y]) - np.array([point1.iloc[i].x, point1.iloc[i].y])
            vertical = np.array([0, 1])
            angle = calculate_angle(es, vertical)
            s_vect = s_vect.append(pd.DataFrame([angle]))
    s_vect = s_vect.reset_index(drop=True)
    s_vect.columns = ['angle']

    # Plot shoulder vectors
    plt.figure()
    plt.plot(s_vect.index, s_vect.angle, 'g')
    plt.savefig(os.path.join(file_location, 's_vect.png'))
    plt.close()


    # Calculate speeds
    def calculate_speed(point, label):
        speed = []
        for i in range(point.shape[0] - 1):
            dx = (point.iloc[i + 1].x - point.iloc[i].x) / (point.index[i + 1] - point.index[i])
            dy = (point.iloc[i + 1].y - point.iloc[i].y) / (point.index[i + 1] - point.index[i])
            spd = np.sqrt(dx ** 2 + dy ** 2)
            speed.append([point.index[i], spd, label])
        return pd.DataFrame(speed, columns=['frame', 'speed', 'label'])


    Speed1 = calculate_speed(point1, 'shoulder')
    Speed2 = calculate_speed(point2, 'elbow')
    Speed3 = calculate_speed(point3, 'wrist')
    Speed4 = calculate_speed(point4, 'hip')

    # Plot speeds
    plt.figure()
    plt.plot(Speed1.frame, Speed1.speed, 'r', label='shoulder')
    plt.plot(Speed2.frame, Speed2.speed, 'g', label='elbow')
    plt.plot(Speed3.frame, Speed3.speed, 'b', label='wrist')
    plt.plot(Speed4.frame, Speed4.speed, 'y', label='hip')
    plt.legend()
    plt.savefig(os.path.join(file_location, 'v-t_speeds.png'))
    plt.close()


    # Calculate accelerations
    def calculate_acceleration(speed_df, label):
        accel = []
        for i in range(speed_df.shape[0] - 1):
            acc = (speed_df.iloc[i + 1].speed - speed_df.iloc[i].speed) / (
                        speed_df.iloc[i + 1].frame - speed_df.iloc[i].frame)
            accel.append([speed_df.iloc[i].frame, acc, label])
        return pd.DataFrame(accel, columns=['frame', 'acceleration', 'label'])


    A1 = calculate_acceleration(Speed1, 'shoulder')
    A2 = calculate_acceleration(Speed2, 'elbow')
    A3 = calculate_acceleration(Speed3, 'wrist')
    A4 = calculate_acceleration(Speed4, 'hip')

    # Plot accelerations
    plt.figure()
    plt.plot(A1.frame, A1.acceleration, 'r', label='shoulder')
    plt.plot(A2.frame, A2.acceleration, 'g', label='elbow')
    plt.plot(A3.frame, A3.acceleration, 'b', label='wrist')
    plt.plot(A4.frame, A4.acceleration, 'y', label='hip')
    plt.legend()
    plt.savefig(os.path.join(file_location, 'a-t_accelerations.png'))
    plt.close()


    # Calculate jerks
    def calculate_jerk(accel_df, label):
        jerk = []
        for i in range(accel_df.shape[0] - 1):
            jrk = (accel_df.iloc[i + 1].acceleration - accel_df.iloc[i].acceleration) / (
                        accel_df.iloc[i + 1].frame - accel_df.iloc[i].frame)
            jerk.append([accel_df.iloc[i].frame, jrk, label])
        return pd.DataFrame(jerk, columns=['frame', 'jerk', 'label'])


    J1 = calculate_jerk(A1, 'shoulder')
    J2 = calculate_jerk(A2, 'elbow')
    J3 = calculate_jerk(A3, 'wrist')
    J4 = calculate_jerk(A4, 'hip')

    # Plot jerks
    plt.figure()
    plt.plot(J1.frame, J1.jerk, 'r', label='shoulder')
    plt.plot(J2.frame, J2.jerk, 'g', label='elbow')
    plt.plot(J3.frame, J3.jerk, 'b', label='wrist')
    plt.plot(J4.frame, J4.jerk, 'y', label='hip')
    plt.legend()
    plt.savefig(os.path.join(file_location, 'j-t_jerks.png'))
    plt.close()

    # Ensure all points DataFrames have the same columns before renaming
    for point_df in [point1, point2, point3, point4]:
        print(point_df.columns)

    # Prepare final dataframe
    point1.columns = ['x_shoulder', 'y_shoulder', 'accuracy_shoulder']
    point2.columns = ['x_elbow', 'y_elbow', 'accuracy_elbow']
    point3.columns = ['x_wrist', 'y_wrist', 'accuracy_wrist']
    if not point4.empty:
        point4.columns = ['x_hip', 'y_hip', 'accuracy_hip']

    point1['frame'] = point1.index
    point2['frame'] = point2.index
    point3['frame'] = point3.index
    if not point4.empty:
        point4['frame'] = point4.index

    point1['y_shoulder'] = height - point1['y_shoulder']
    point2['y_elbow'] = height - point2['y_elbow']
    point3['y_wrist'] = height - point3['y_wrist']
    if not point4.empty:
        point4['y_hip'] = height - point4['y_hip']

    # Concatenate all dataframes with proper labels
    dfs_to_concat = [
        point1,
        Speed1[['frame', 'speed']].rename(columns={'frame': 'frame_speed_shoulder', 'speed': 'speed_shoulder'}),
        A1[['frame', 'acceleration']].rename(
            columns={'frame': 'frame_acceleration_shoulder', 'acceleration': 'acceleration_shoulder'}),
        J1[['frame', 'jerk']].rename(columns={'frame': 'frame_jerk_shoulder', 'jerk': 'jerk_shoulder'}),

        point2,
        Speed2[['frame', 'speed']].rename(columns={'frame': 'frame_speed_elbow', 'speed': 'speed_elbow'}),
        A2[['frame', 'acceleration']].rename(
            columns={'frame': 'frame_acceleration_elbow', 'acceleration': 'acceleration_elbow'}),
        J2[['frame', 'jerk']].rename(columns={'frame': 'frame_jerk_elbow', 'jerk': 'jerk_elbow'}),

        point3,
        Speed3[['frame', 'speed']].rename(columns={'frame': 'frame_speed_wrist', 'speed': 'speed_wrist'}),
        A3[['frame', 'acceleration']].rename(
            columns={'frame': 'frame_acceleration_wrist', 'acceleration': 'acceleration_wrist'}),
        J3[['frame', 'jerk']].rename(columns={'frame': 'frame_jerk_wrist', 'jerk': 'jerk_wrist'})
    ]

    if not point4.empty:
        dfs_to_concat.extend([
            point4,
            Speed4[['frame', 'speed']].rename(columns={'frame': 'frame_speed_hip', 'speed': 'speed_hip'}),
            A4[['frame', 'acceleration']].rename(
                columns={'frame': 'frame_acceleration_hip', 'acceleration': 'acceleration_hip'}),
            J4[['frame', 'jerk']].rename(columns={'frame': 'frame_jerk_hip', 'jerk': 'jerk_hip'})
        ])

    final_df = pd.concat(dfs_to_concat, axis=1)

    # Save final DataFrame
    final_csv_path = os.path.join(file_location, "Final.csv")
    print(f"Saving final DataFrame to {final_csv_path}")
    final_df.to_csv(final_csv_path, index=False, encoding='utf_8_sig')
    print(f"Final DataFrame saved successfully to {final_csv_path}")

except Exception as e:
    print(e)
    sys.exit(-1)
