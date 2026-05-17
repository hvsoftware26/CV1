import cv2
import numpy as np
import torch


def preprocess_frame(frame, img_size):
    #Vì cv mặc định đọc ảnh định dạng BGR -> Cần chuyển qua RGB
    #Resize cho đồng bộ size ảnh

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (img_size, img_size))

    return frame


def read_video_frames(video_path, num_frames=30, img_size=128):
    """
    Đọc video và lấy cố định num_frames frame.

    Output:
        Tensor shape: [T, C, H, W]
        Ví dụ: [30, 3, 128, 128]
    """

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return torch.zeros(num_frames, 3, img_size, img_size)
        
    #Cắt đều frames từ tổng frames -> làm tròn 
    frame_indices = np.linspace(0, total_frames - 1, num_frames)
    frame_indices = np.round(frame_indices).astype(int)
    #Tăng tốc tìm bằng set
    target_indices = set(frame_indices)

    frames = []
    current_index = 0

    while True:
        #Đọc từng frame
        ret, frame = cap.read()

        if not ret:
            break

        if current_index in target_indices:
            frame = preprocess_frame(frame, img_size)
            #Đổi từ 0 - 255 sang 0 - 1
            frame = frame.astype(np.float32) / 255.0
            # chuyển sang tensor [C, H, W]
            frame = torch.tensor(frame).permute(2, 0, 1)

            frames.append(frame)

        current_index += 1

    cap.release()

    if len(frames) == 0:
        frames = [torch.zeros(3, img_size, img_size)]

    #Tránh TH frames 1 video không đủ 30 -> lặp lại frame cuối cùng cho đủ
    while len(frames) < num_frames:
        frames.append(frames[-1])

    frames = frames[:num_frames]
    #Gộp list frame thành tensor [T, C, H, W]
    frames = torch.stack(frames)

    return frames