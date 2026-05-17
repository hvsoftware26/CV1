import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Sửa lại path theo file của bạn
file_1_path = "D:\AI CONTEST\CV_B1\outputs\cv_submission.csv"
file_2_path = "D:\AI CONTEST\CV_B1\outputs\sample_submission.csv"

df1 = pd.read_csv(file_1_path, encoding="utf-8")
df2 = pd.read_csv(file_2_path, encoding="utf-8")

print("===== INFO =====")
print("File 1 shape:", df1.shape)
print("File 2 shape:", df2.shape)

print("File 1 columns:", list(df1.columns))
print("File 2 columns:", list(df2.columns))

# Chuẩn hóa text để tránh lỗi khoảng trắng
df1["video_name"] = df1["video_name"].astype(str).str.strip()
df1["label"] = df1["label"].astype(str).str.strip()

df2["video_name"] = df2["video_name"].astype(str).str.strip()
df2["label"] = df2["label"].astype(str).str.strip()

# Merge theo video_name
merged = df1.merge(
    df2,
    on="video_name",
    how="outer",
    suffixes=("_file1", "_file2"),
    indicator=True
)

print("\n===== MERGE RESULT =====")
print(merged["_merge"].value_counts())

# Những dòng có ở cả 2 file
common = merged[merged["_merge"] == "both"].copy()

# So sánh label
common["is_same_label"] = common["label_file1"] == common["label_file2"]

same_count = common["is_same_label"].sum()
diff_count = len(common) - same_count

print("\n===== LABEL COMPARE =====")
print("Số video có ở cả 2 file:", len(common))
print("Số dòng label giống nhau:", same_count)
print("Số dòng label khác nhau:", diff_count)

if len(common) > 0:
    print("Tỷ lệ giống label:", round(same_count / len(common), 4))

# Nếu file_1 là ground truth, có thể tính metric
accuracy = accuracy_score(common["label_file1"], common["label_file2"])
macro_f1 = f1_score(common["label_file1"], common["label_file2"], average="macro")

print("\n===== METRIC nếu file_1 là đáp án đúng =====")
print("Accuracy:", round(accuracy, 4))
print("Macro-F1:", round(macro_f1, 4))

print("\n===== CÁC DÒNG KHÁC LABEL =====")
diff_rows = common[common["is_same_label"] == False]
print(diff_rows[["video_name", "label_file1", "label_file2"]].head(30))

# Lưu file kết quả so sánh
output_path = "/content/drive/MyDrive/CV1/compare_labels_result.csv"
merged.to_csv(output_path, index=False, encoding="utf-8-sig")

print("\nĐã lưu file so sánh tại:", output_path)