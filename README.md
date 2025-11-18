<h1 align="center">PHÂN TÍCH CẢM XÚC DỰA TRÊN PHẢN HỒI SINH VIÊN </h1>

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

## Giới thiệu

- Student Sentiment Analysis là hệ thống phân tích cảm xúc trong phản hồi của sinh viên, giúp nhà trường đánh giá mức độ hài lòng và phát hiện sớm các vấn đề trong quá trình học tập.

- Hệ thống kết hợp sức mạnh của PHoBERT – mô hình ngôn ngữ mạnh mẽ cho tiếng Việt – cùng CNN và GRU để phát hiện cụm từ mang tính cảm xúc (“chưa hiểu”, “rất hay”, “khó tiếp thu”…), đồng thời nắm bắt ngữ cảnh của toàn câu để phân loại cảm xúc chính xác hơn.

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

## Kiến trúc hệ thống

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

## Công nghệ sử dụng

- **FastAPI**: xây dựng REST API phân tích phản hồi.  
- **PyTorch** + **Transformers (PHoBERT)**: mô hình hoá ngôn ngữ và học sâu.  
- **CNN + GRU**: phân tích cảm xúc.  
- **GRU**: phân tích chủ đề.  
- **Pandas, OpenPyXL**: đọc và xử lý Excel/CSV.  

