# PHÂN TÍCH CẢM XÚC DỰA TRÊN PHẢN HỒI SINH VIÊN

<div align="center">

<p align="center">
  <img src="./img/logoDaiNam.png" alt="DaiNam University Logo" width="200"/>
  <img src="./img/LogoAIoTLab.png" alt="AIoTLab Logo" width="170"/>
</p>

[![Made by AIoTLab](https://img.shields.io/badge/Made%20by%20AIoTLab-blue?style=for-the-badge)](https://www.facebook.com/DNUAIoTLab)  
[![Fit DNU](https://img.shields.io/badge/Fit%20DNU-green?style=for-the-badge)](https://fitdnu.net/)  
[![DaiNam University](https://img.shields.io/badge/DaiNam%20University-red?style=for-the-badge)](https://dainam.edu.vn)

</div>

---

## 1. Giới thiệu

Hệ thống **phân tích phản hồi sinh viên** giúp:

- Phân loại **cảm xúc**: 😡 Tiêu cực – 😐 Trung lập – 😊 Tích cực  
- Nhận diện **chủ đề góp ý**:
  - 🧑‍🏫 Giảng viên  
  - 📘 Chương trình học  
  - 🏫 Cơ sở vật chất  
  - 💻 Học liệu / Website  

Hệ thống sử dụng:

- **PHoBERT** để sinh embedding tiếng Việt.
- **CNN + GRU** cho phân tích cảm xúc.
- **GRU** cho phân tích chủ đề.
- Tích hợp vào **API FastAPI**, hỗ trợ phân tích:
  - Một câu phản hồi.
  - Nhiều phản hồi trong file Excel/CSV.
  - Dữ liệu khảo sát (Likert + câu hỏi mở).

---

## 2. Kiến trúc hệ thống

```bash
PhanTichPhanHoi/
├── models/
│   ├── sent_phobert_hybrid_best.pth      # Mô hình cảm xúc
│   └── topic_phobert_gru_best.pth        # Mô hình chủ đề
├── app.py                                # FastAPI app (REST API)
├── data_processing.py                    # Load PhoBERT, model, tiền xử lý & suy luận
├── model_classes.py                      # Định nghĩa kiến trúc CNN–GRU & GRU
├── analyze_demo.html                     # Giao diện demo (frontend)
└── requirements.txt                      # Thư viện cần cài đặt
```

**Luồng chính:**

- `model_classes.py`  
  - Định nghĩa:
    - `PhoBERT_CNN_GRU_Sentiment` (3 nhãn cảm xúc).
    - `PhoBERT_GRU_Topic` (4 nhãn chủ đề).

- `data_processing.py`  
  - Load `vinai/phobert-base` và tokenizer.  
  - Load trọng số từ `models/*.pth`.  
  - Cung cấp các hàm:
    - `split_feedback_text(text)`
    - `predict_feedback(text)`
    - `analyze_feedback_text(full_text)`
    - `analyze_many_texts(text_list, batch_size)`

- `app.py`  
  - Tạo FastAPI app, CORS.  
  - Endpoint cho phân tích văn bản, file, khảo sát.  
  - Quản lý cache phân tích và thống kê.

---

## 3. Cài đặt môi trường

```bash
conda create -n sentiment-dev python=3.9
conda activate sentiment-dev
pip install -r requirements.txt
```

`requirements.txt`:

```text
fastapi
uvicorn[standard]
torch
transformers
pandas
scikit-learn
openpyxl
```

---

## 4. Chạy API

Từ thư mục `PhanTichPhanHoi`:

```bash
uvicorn app:app --reload
```

- API: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`

---

## 5. Các endpoint chính

### 5.1. Phân tích một đoạn phản hồi

- **URL**: `POST /analyze_text/`  
- **Body (JSON)**:

```json
{
  "text": "Môn học rất hay, nhưng cơ sở vật chất còn kém"
}
```

- **Kết quả** (rút gọn):

```json
{
  "original_text": "Môn học rất hay, nhưng cơ sở vật chất còn kém",
  "analysis_parts": [
    {
      "part": "Môn học rất hay",
      "sentiment": "😊 Tích cực",
      "topic": "📘 Chương trình học"
    },
    {
      "part": "cơ sở vật chất còn kém",
      "sentiment": "😡 Tiêu cực",
      "topic": "🏫 Cơ sở vật chất"
    }
  ]
}
```

---

### 5.2. Phân tích file Excel/CSV

- **URL**: `POST /analyze_file`  
- **Form-data**:
  - `file`: file `.xlsx`, `.xls` hoặc `.csv`
  - `text_column` (mặc định: `"Phản hồi"`)
  - `student_id_column` (mặc định: `"Mã sinh viên"`)
  - `batch_size` (mặc định: `64`)

Kết quả:

- `total_rows`: số dòng phản hồi.  
- `summary.topic_sentiment`: thống kê số câu **pos/neu/neg** theo từng chủ đề.  
- `rows`: chi tiết từng dòng, kèm `analysis_parts`, `student_id`, `sheet`.

---

### 5.3. Khảo sát sinh viên

- **Gửi khảo sát**: `POST /submit_survey`  
  - Body: `SurveyResponse` gồm:
    - `student_id`, `class_name`.
    - Các câu Likert `q1..q23`.
    - Các câu mở: `q15_gvcn_improve`, `q20_teacher_improve`, `q24_leader_improve`, `q25_satisfied`, `q26_unsatisfied`, `q27_suggestions`.

- **Thống kê khảo sát**: `GET /survey_stats`  
  - Trả về:
    - `total_responses`
    - `likert_statistics` (average + distribution 1–5)
    - `open_feedback_analysis` (phân tích AI cho câu mở)
    - Top chủ đề được khen/chê/đề xuất cải thiện.

- **Lấy toàn bộ bản ghi**: `GET /survey_records`

---

### 5.4. Kiểm tra trạng thái server

- **URL**: `GET /health`  
- **Trả về**:

```json
{
  "status": "ok",
  "device": "cpu"
}
```

(hoặc `"cuda:0"` nếu chạy được trên GPU)

---

## 6. Công nghệ sử dụng

- **FastAPI**: xây dựng REST API phân tích phản hồi.  
- **PyTorch** + **Transformers (PHoBERT)**: mô hình hoá ngôn ngữ và học sâu.  
- **CNN + GRU**: phân tích cảm xúc.  
- **GRU**: phân tích chủ đề.  
- **Pandas, OpenPyXL**: đọc và xử lý Excel/CSV.  

---

## 7. Hướng phát triển

- Mở rộng thêm lớp cảm xúc (rất tích cực, hơi tiêu cực,…).  
- Tối ưu mô hình cho dữ liệu chuyên ngành từng khoa.  
- Tích hợp trực tiếp với LMS để phân tích phản hồi theo thời gian thực.  
- Xây dựng dashboard trực quan cho phòng đào tạo / ban giám hiệu.
