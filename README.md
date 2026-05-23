# Video Hand Gesture Recognition

Dự án nhận diện cử chỉ tay từ video. Tool dùng PyTorch để train mô hình phân loại video, sau đó chạy inference trên tập test và xuất file submission dạng CSV.

Mô hình hiện tại sử dụng kiến trúc `SmallCNN_GRU`: CNN trích xuất đặc trưng từng frame, GRU học quan hệ theo thời gian giữa các frame, cuối cùng phân loại ra nhãn cử chỉ.

## Chức năng chính

- Đọc video `.mp4` từ thư mục train theo từng nhãn.
- Tự động tạo `label_map.json` từ tên thư mục nhãn.
- Chia dữ liệu train và validation theo tỉ lệ 80/20.
- Train mô hình CNN + GRU bằng PyTorch.
- Tính validation Macro-F1 sau mỗi epoch.
- Lưu model tốt nhất vào `outputs/best_model.pth`.
- Chạy dự đoán trên thư mục public test.
- Xuất file submission `outputs/cv_submission.csv`.
- Kiểm tra cấu trúc dữ liệu train/test.
- Kiểm tra file submission có đúng định dạng không.

## Cấu trúc dự án

```text
video-hand-gesture-recognition/
    train.py
    inference.py
    check_data.py
    check_submission.py
    check_diff.py
    requirements.txt
    services/
        config.py
        data_service.py
        video_service.py
        model_service.py
        train_service.py
        inference_service.py
        submission_service.py
```

## Yêu cầu môi trường

- Python 3.10 trở lên.
- Windows, Linux hoặc macOS.
- Khuyến nghị có GPU CUDA nếu train nhiều video.
- Dữ liệu video định dạng `.mp4`.

## Cài đặt

Tạo môi trường ảo:

```powershell
py -m venv .venv
```

Kích hoạt môi trường ảo bằng PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Hoặc kích hoạt bằng CMD:

```cmd
.venv\Scripts\activate.bat
```

Cài thư viện:

```powershell
pip install -r requirements.txt
```

Nếu dùng GPU, cần đảm bảo bản `torch`, `torchvision`, `torchaudio` trong `requirements.txt` phù hợp với CUDA trên máy.

## Chuẩn bị dữ liệu

Mặc định tool đọc dữ liệu theo cấu trúc:

```text
dataset/
    train/
        label_1/
            video_001.mp4
            video_002.mp4
        label_2/
            video_003.mp4
            video_004.mp4
    public_test/
        test_001.mp4
        test_002.mp4
```

Trong đó:

- `dataset/train`: chứa dữ liệu train.
- Mỗi thư mục con trong `dataset/train` là một nhãn cử chỉ.
- Mỗi file `.mp4` trong thư mục nhãn là một video train.
- `dataset/public_test`: chứa các video test cần dự đoán.

Ví dụ nếu có ba nhãn `hello`, `thanks`, `yes`:

```text
dataset/train/hello/a.mp4
dataset/train/thanks/b.mp4
dataset/train/yes/c.mp4
dataset/public_test/0001.mp4
```

## Cấu hình

Các cấu hình nằm trong `services/config.py` và có thể ghi đè bằng biến môi trường hoặc file `.env`.

Giá trị mặc định:

```text
TRAIN_DIR=./dataset/train
TEST_DIR=./dataset/public_test
OUTPUT_DIR=./outputs
NUM_FRAMES=30
IMG_SIZE=128
BATCH_SIZE=2
EPOCHS=50
LR=0.001
SEED=42
```

Có thể tạo file `.env` ở thư mục gốc để chỉnh cấu hình:

```env
TRAIN_DIR=./dataset/train
TEST_DIR=./dataset/public_test
OUTPUT_DIR=./outputs
NUM_FRAMES=30
IMG_SIZE=128
BATCH_SIZE=4
EPOCHS=30
LR=0.001
SEED=42
```

Ý nghĩa cấu hình:

- `TRAIN_DIR`: thư mục chứa dữ liệu train.
- `TEST_DIR`: thư mục chứa video test.
- `OUTPUT_DIR`: thư mục lưu model, label map và submission.
- `NUM_FRAMES`: số frame lấy mẫu từ mỗi video.
- `IMG_SIZE`: kích thước resize mỗi frame.
- `BATCH_SIZE`: số video trong mỗi batch.
- `EPOCHS`: số vòng train.
- `LR`: learning rate.
- `SEED`: seed để chia dữ liệu và train ổn định hơn.

## Kiểm tra dữ liệu

Trước khi train, chạy:

```powershell
python check_data.py
```

Lệnh này sẽ in ra:

- Đường dẫn cấu hình đang dùng.
- Danh sách nhãn trong `TRAIN_DIR`.
- Số video của từng nhãn.
- Tổng số class.
- Tổng số video train.
- Số video trong `TEST_DIR`.

Nếu thiếu thư mục train hoặc cấu trúc dữ liệu sai, script sẽ báo lỗi để sửa trước khi train.

## Train model

Chạy:

```powershell
python train.py
```

Quá trình train sẽ:

1. Đọc cấu hình.
2. Tạo thư mục `outputs` nếu chưa có.
3. Đọc toàn bộ video train.
4. Tạo label map và lưu vào `outputs/label_map.json`.
5. Chia train/validation theo `SEED`.
6. Tính class weight để giảm ảnh hưởng lệch lớp.
7. Train mô hình CNN + GRU.
8. Đánh giá validation Macro-F1 sau mỗi epoch.
9. Lưu model tốt nhất vào `outputs/best_model.pth`.

Nếu đã có `outputs/best_model.pth`, script sẽ load model cũ và tiếp tục train từ checkpoint đó.

## Chạy inference

Sau khi train xong và có đủ hai file:

```text
outputs/best_model.pth
outputs/label_map.json
```

Chạy:

```powershell
python inference.py
```

Script sẽ:

1. Load `outputs/label_map.json`.
2. Load model từ `outputs/best_model.pth`.
3. Đọc toàn bộ video `.mp4` trong `TEST_DIR`.
4. Dự đoán nhãn cho từng video.
5. Lưu kết quả vào `outputs/cv_submission.csv`.

File submission có dạng:

```csv
video_name,label
test_001.mp4,hello
test_002.mp4,thanks
```

## Kiểm tra submission

Sau khi chạy inference, kiểm tra file submission:

```powershell
python check_submission.py
```

Script sẽ kiểm tra:

- File có đúng hai cột `video_name` và `label`.
- Không có giá trị rỗng.
- Không có video bị trùng.
- Danh sách video trong submission khớp với thư mục test.

Nếu hợp lệ, chương trình sẽ in `Submission hợp lệ.`

## So sánh hai file submission

File `check_diff.py` dùng để so sánh hai file CSV và tính các chỉ số như accuracy, Macro-F1 nếu một file được xem là đáp án đúng.

Trước khi chạy, cần sửa hai biến trong file:

```python
file_1_path = "duong_dan_file_1.csv"
file_2_path = "duong_dan_file_2.csv"
```

Sau đó chạy:

```powershell
python check_diff.py
```

## Luồng sử dụng đề xuất

1. Chuẩn bị dữ liệu theo đúng cấu trúc `dataset/train/<label>/*.mp4` và `dataset/public_test/*.mp4`.
2. Cài thư viện bằng `pip install -r requirements.txt`.
3. Chạy `python check_data.py` để kiểm tra dữ liệu.
4. Chạy `python train.py` để train model.
5. Chạy `python inference.py` để tạo submission.
6. Chạy `python check_submission.py` để kiểm tra file submission.
7. Nộp file `outputs/cv_submission.csv`.

## Kết quả đầu ra

Sau khi train và inference, thư mục `outputs` sẽ có:

```text
outputs/
    best_model.pth
    label_map.json
    cv_submission.csv
```

Trong đó:

- `best_model.pth`: trọng số model tốt nhất theo validation Macro-F1.
- `label_map.json`: ánh xạ giữa tên nhãn và ID số.
- `cv_submission.csv`: file kết quả để nộp.

## Ghi chú kỹ thuật

- Mỗi video được lấy mẫu đều `NUM_FRAMES` frame.
- Mỗi frame được resize về `IMG_SIZE x IMG_SIZE`.
- Frame được chuẩn hóa về khoảng `[0, 1]`.
- Nếu video lỗi hoặc không đọc được frame, tool trả về tensor zero để tránh crash.
- DataLoader đang dùng `num_workers=0`, phù hợp với Windows và dễ debug hơn.
- Metric chính trong train là Macro-F1, phù hợp khi dữ liệu bị lệch lớp.

## Lỗi thường gặp

### Không tìm thấy thư mục train

Kiểm tra lại `TRAIN_DIR` trong `.env` hoặc dùng cấu trúc mặc định `dataset/train`.

### Không có video trong một nhãn

Đảm bảo mỗi thư mục nhãn có ít nhất một file `.mp4`.

### Lỗi khi chia train/validation với stratify

Mỗi nhãn nên có đủ số video để chia train và validation. Nếu một nhãn có quá ít video, cần bổ sung dữ liệu hoặc chỉnh logic chia dữ liệu.

### Không tìm thấy model khi inference

Cần chạy `python train.py` trước để tạo `outputs/best_model.pth`.

### Không tìm thấy label map

Cần chạy train trước để tạo `outputs/label_map.json`.

### Train quá chậm

Giảm `NUM_FRAMES`, giảm `IMG_SIZE`, giảm `BATCH_SIZE`, hoặc dùng GPU CUDA.

### Hết bộ nhớ GPU

Giảm `BATCH_SIZE`, `IMG_SIZE` hoặc `NUM_FRAMES`.
