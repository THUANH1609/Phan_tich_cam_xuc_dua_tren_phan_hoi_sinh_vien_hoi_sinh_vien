import torch
import re
import os
from transformers import AutoTokenizer, AutoModel
from model_classes import PhoBERT_CNN_GRU_Sentiment, PhoBERT_GRU_Topic # Cần đảm bảo file này tồn tại

# ----------------------------------------------------
# 1. THIẾT LẬP VÀ LOAD MODEL (Chỉ chạy 1 lần)
# ----------------------------------------------------
MAX_LEN = 96
# 🚨 QUAN TRỌNG: Buộc chạy trên CPU (hoặc GPU nếu có)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print(f"Loading models on device: {device}")
VERBOSE = False

# Load Tokenizer và PhoBERT Base
try:
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    phobert_base = AutoModel.from_pretrained("vinai/phobert-base")
except Exception as e:
    print(f"❌ LỖI KHỞI TẠO: Không thể tải PhoBERT/Tokenizer. {e}")
    raise

# Khởi tạo cấu trúc mô hình
model_sent = PhoBERT_CNN_GRU_Sentiment(phobert_base, n_classes=3)
model_topic = PhoBERT_GRU_Topic(phobert_base, n_classes=4)

# Load trọng số đã lưu
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models') 

try:
    # 🚨 Dùng map_location để đảm bảo tải đúng thiết bị (CPU hoặc CUDA)
    model_sent.load_state_dict(torch.load(
        os.path.join(MODEL_PATH, 'sent_phobert_hybrid_best.pth'), 
        map_location=device
    ))
    model_topic.load_state_dict(torch.load(
        os.path.join(MODEL_PATH, 'topic_phobert_gru_best.pth'), 
        map_location=device
    ))
    
    # Chuyển mô hình sang thiết bị và đặt ở chế độ đánh giá
    model_sent.to(device).eval()
    model_topic.to(device).eval()
    print("✅ Model weights loaded and models set to evaluation mode.")

except Exception as e:
    print(f"❌ LỖI LOAD TRỌNG SỐ: Kiểm tra thư mục 'models/' và file .pth. {e}")
    # Đưa ra lỗi để Uvicorn hiển thị Traceback chi tiết
    raise

# Định nghĩa các ánh xạ nhãn (đã kiểm tra và sửa lỗi)
sentiment_map = {0:"😡 Tiêu cực", 1:"😐 Trung lập", 2:"😊 Tích cực"}
topic_map = {0:"🧑‍🏫 Giảng viên", 1:"📘 Chương trình học", 2:"🏫 Cơ sở vật chất", 3:"💻 Học liệu/Website"}


# ----------------------------------------------------
# 2. HÀM XỬ LÝ DỮ LIỆU
# ----------------------------------------------------

# Hàm Tách Câu theo từ nối và dấu câu
def split_feedback_text(text):
    """
    Tách văn bản thành các phần dựa trên:
    - Từ nối đối lập: nhưng, tuy nhiên, còn, song
    - Từ nối bổ sung: và, còn
    - Dấu câu: . ! ?
    
    Ví dụ: "thầy dạy hay, nhưng cơ sở vật chất kém" 
           → ["thầy dạy hay", "cơ sở vật chất kém"]
    """
    text = text.strip()
    
    # Tách theo dấu câu trước
    sentences = re.split(r'[.!?]+', text)
    
    all_parts = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Tách mỗi câu theo từ nối (bỏ dấu phẩy trước từ nối)
        parts = re.split(
            r',?\s*(?:\bnhưng mà\b|\bnhưng\b|\btuy nhiên\b|\bmà\b|\bcòn\b|\bsong\b|\bvà\b)', 
            sentence, 
            flags=re.IGNORECASE
        )
        
        # Lọc và làm sạch các phần
        cleaned_parts = [p.strip(" ,.") for p in parts if p.strip()]
        all_parts.extend(cleaned_parts)
    
    return all_parts if all_parts else [text]

# Hàm Dự đoán Chính
def preprocess_text(text):
    """Tiền xử lý văn bản trước khi đưa vào mô hình"""
    # Xóa các ký tự đặc biệt không cần thiết
    text = re.sub(r'[^\w\sÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠ-ỹ]', ' ', text)
    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_feedback(text):
    """
    Dự đoán cảm xúc và chủ đề cho văn bản đầu vào
    
    Args:
        text (str): Văn bản cần phân tích
        
    Returns:
        tuple: (sentiment, topic, confidence_sent, confidence_topic)
    """
    try:
        # Tiền xử lý văn bản
        text = preprocess_text(text)
        if not text:
            return "Không xác định", "Không xác định", 0.0, 0.0
            
        # Mã hóa văn bản
        enc = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=MAX_LEN,
            return_tensors='pt'
        )
        
        # Chuyển dữ liệu vào device phù hợp
        ids = enc['input_ids'].to(device)
        mask = enc['attention_mask'].to(device)
        
        # Dự đoán
        with torch.no_grad():
            model_sent.eval()
            model_topic.eval()
            
            # Dự đoán cảm xúc
            s_out = model_sent(ids, mask)
            s_probs = torch.softmax(s_out, dim=1)
            s_label = torch.argmax(s_probs, dim=1).item()
            s_confidence = s_probs[0][s_label].item()
            
            # Dự đoán chủ đề
            t_out = model_topic(ids, mask)
            t_probs = torch.softmax(t_out, dim=1)
            t_label = torch.argmax(t_probs, dim=1).item()
            t_confidence = t_probs[0][t_label].item()
            
            # Lấy nhãn tương ứng
            sentiment = sentiment_map.get(s_label, "Không xác định")
            topic = topic_map.get(t_label, "Không xác định")

        # -----------------------------
        # Lớp sửa luật (rule-based) nhẹ
        # -----------------------------
        txt_lower = text.lower()
        pos_words = [
            "hay", "tốt", "tuyệt vời", "hài lòng", "nhiệt tình", "thân thiện",
            "dễ hiểu", "ổn", "ok", "đẹp", "sạch", "chất lượng", "tận tâm"
        ]
        neg_words = [
            "tệ", "kém", "kém chất lượng", "đắt", "chán", "ồn", "bẩn",
            "cũ", "khó hiểu", "chậm", "lâu", "quá tải", "thiếu"
        ]
        teacher_words = [
            "thầy", "cô", "giảng viên", "gv", "dạy", "giảng dạy", 
            "gvcn", "chủ nhiệm", "giáo viên", "thầy cô", "đội ngũ giảng viên",
            "phương pháp giảng dạy", "kỹ năng sư phạm", "nhiệt tình"
        ]
        facility_words = [
            "cơ sở vật chất", "phòng học", "bàn ghế", "wifi", "máy chiếu", 
            "phòng", "csdl", "thư viện", "phòng thí nghiệm", "phòng lab",
            "ký túc xá", "ktx", "căn tin", "sân chơi", "thiết bị", "trang thiết bị"
        ]
        program_words = [
            "chương trình", "học phần", "môn", "khung chương trình", 
            "tín chỉ", "lịch học", "chất lượng đào tạo", "nội dung học", 
            "kiến thức", "kỹ năng", "thực hành", "lý thuyết", "bài tập"
        ]
        material_words = [
            "học liệu", "tài liệu", "website", "web", "lms", "moodle", 
            "bài giảng", "slides", "giáo trình", "sách", "tư liệu",
            "học phí", "chi phí", "lệ phí", "tiền học", "đóng học"
        ]

        def contains_any(words):
            return any(w in txt_lower for w in words)

        # Xử lý phủ định đơn giản: "không/chưa/chẳng + tích cực" => tiêu cực
        negation_tokens = ["không", "chưa", "chẳng", "chả"]
        has_negation = any(tok in txt_lower for tok in negation_tokens)
        pos_hit = contains_any(pos_words)
        neg_hit = contains_any(neg_words)

        # Sửa cảm xúc CHỈ KHI mô hình không chắc chắn (trung lập hoặc confidence thấp)
        if sentiment == "😐 Trung lập" or s_confidence < 0.6:
            # Phủ định + tích cực → tiêu cực (ví dụ: "không hay")
            if has_negation and pos_hit and not neg_hit:
                sentiment = "😡 Tiêu cực"
                s_confidence = 0.75
            # Chỉ có từ tiêu cực → tiêu cực
            elif neg_hit and not pos_hit:
                sentiment = "😡 Tiêu cực"
                s_confidence = 0.75
            # Chỉ có từ tích cực → tích cực
            elif pos_hit and not neg_hit and not has_negation:
                sentiment = "😊 Tích cực"
                s_confidence = 0.75

        # Sửa chủ đề bằng chỉ báo từ khoá rõ ràng
        if contains_any(teacher_words):
            topic = topic_map.get(0, topic)  # Giảng viên
            t_confidence = max(t_confidence, 0.75)
        elif contains_any(program_words):
            topic = topic_map.get(1, topic)  # Chương trình
            t_confidence = max(t_confidence, 0.75)
        elif contains_any(facility_words):
            topic = topic_map.get(2, topic)  # Cơ sở vật chất
            t_confidence = max(t_confidence, 0.75)
        elif contains_any(material_words):
            topic = topic_map.get(3, topic)  # Học liệu/Website
            t_confidence = max(t_confidence, 0.75)
        
        # In thông tin debug
        if VERBOSE:
            print(f"\n📝 Văn bản: {text}")
            print(f"😊 Cảm xúc: {sentiment} (Độ tin cậy: {s_confidence*100:.1f}%)")
            print(f"🏷️ Chủ đề: {topic} (Độ tin cậy: {t_confidence*100:.1f}%)")
        
        return sentiment, topic, s_confidence, t_confidence
            
    except Exception as e:
        print(f"❌ Lỗi khi dự đoán: {str(e)}")
        return "Lỗi", "Lỗi", 0.0, 0.0

# Hàm xử lý phân tích toàn bộ phản hồi
def analyze_feedback_text(full_text):
    """
    Phân tích toàn bộ văn bản phản hồi, tách thành các câu và phân tích từng câu
    
    Args:
        full_text (str): Toàn bộ văn bản phản hồi
        
    Returns:
        list: Danh sách kết quả phân tích cho từng câu
    """
    # Tách câu dựa trên từ nối và dấu câu để xử lý các đánh giá trái ngược
    # Ví dụ: "thầy dạy hay, nhưng cơ sở vật chất kém" → ["thầy dạy hay", "cơ sở vật chất kém"]
    sentences = split_feedback_text(full_text)

    # Batch tokenize để tăng tốc
    cleaned = [preprocess_text(s) for s in sentences]
    if not cleaned:
        cleaned = [preprocess_text(full_text)]
    enc = tokenizer(
        cleaned,
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN,
        return_tensors='pt'
    )
    ids = enc['input_ids'].to(device)
    mask = enc['attention_mask'].to(device)

    with torch.no_grad():
        model_sent.eval()
        model_topic.eval()
        s_out = model_sent(ids, mask)
        t_out = model_topic(ids, mask)
        s_probs = torch.softmax(s_out, dim=1)
        t_probs = torch.softmax(t_out, dim=1)
        s_labels = torch.argmax(s_probs, dim=1)
        t_labels = torch.argmax(t_probs, dim=1)
        s_conf = s_probs.gather(1, s_labels.view(-1,1)).squeeze(1)
        t_conf = t_probs.gather(1, t_labels.view(-1,1)).squeeze(1)

    results = []
    for i, sentence in enumerate(cleaned, 1):
        s_label = int(s_labels[i-1].item())
        t_label = int(t_labels[i-1].item())
        s_confidence = float(s_conf[i-1].item())
        t_confidence = float(t_conf[i-1].item())
        sentiment = sentiment_map.get(s_label, "Không xác định")
        topic = topic_map.get(t_label, "Không xác định")

        # Rule-based điều chỉnh nhẹ (giữ nguyên như predict_feedback)
        txt_lower = sentence.lower()
        pos_words = [
            "hay", "tốt", "tuyệt vời", "hài lòng", "nhiệt tình", "thân thiện",
            "dễ hiểu", "ổn", "ok", "đẹp", "sạch", "chất lượng", "tận tâm"
        ]
        neg_words = [
            "tệ", "kém", "kém chất lượng", "đắt", "chán", "ồn", "bẩn",
            "cũ", "khó hiểu", "chậm", "lâu", "quá tải", "thiếu"
        ]
        teacher_words = [
            "thầy", "cô", "giảng viên", "gv", "dạy", "giảng dạy", 
            "gvcn", "chủ nhiệm", "giáo viên", "thầy cô", "đội ngũ giảng viên",
            "phương pháp giảng dạy", "kỹ năng sư phạm", "nhiệt tình"
        ]
        facility_words = [
            "cơ sở vật chất", "phòng học", "bàn ghế", "wifi", "máy chiếu", 
            "phòng", "csdl", "thư viện", "phòng thí nghiệm", "phòng lab",
            "ký túc xá", "ktx", "căn tin", "sân chơi", "thiết bị", "trang thiết bị"
        ]
        program_words = [
            "chương trình", "học phần", "môn", "khung chương trình", 
            "tín chỉ", "lịch học", "chất lượng đào tạo", "nội dung học", 
            "kiến thức", "kỹ năng", "thực hành", "lý thuyết", "bài tập"
        ]
        material_words = [
            "học liệu", "tài liệu", "website", "web", "lms", "moodle", 
            "bài giảng", "slides", "giáo trình", "sách", "tư liệu",
            "học phí", "chi phí", "lệ phí", "tiền học", "đóng học"
        ]
        def contains_any(words):
            return any(w in txt_lower for w in words)
        negation_tokens = ["không", "chưa", "chẳng", "chả"]
        has_negation = any(tok in txt_lower for tok in negation_tokens)
        pos_hit = contains_any(pos_words)
        neg_hit = contains_any(neg_words)
        if sentiment == "😐 Trung lập" or s_confidence < 0.6:
            if has_negation and pos_hit and not neg_hit:
                sentiment = "😡 Tiêu cực"; s_confidence = 0.75
            elif neg_hit and not pos_hit:
                sentiment = "😡 Tiêu cực"; s_confidence = 0.75
            elif pos_hit and not neg_hit and not has_negation:
                sentiment = "😊 Tích cực"; s_confidence = 0.75
        if contains_any(teacher_words):
            topic = topic_map.get(0, topic); t_confidence = max(t_confidence, 0.75)
        elif contains_any(program_words):
            topic = topic_map.get(1, topic); t_confidence = max(t_confidence, 0.75)
        elif contains_any(facility_words):
            topic = topic_map.get(2, topic); t_confidence = max(t_confidence, 0.75)
        elif contains_any(material_words):
            topic = topic_map.get(3, topic); t_confidence = max(t_confidence, 0.75)

        results.append({
            'sentence_id': i,
            'text': sentences[i-1],
            'sentiment': sentiment,
            'topic': topic,
            'sentiment_confidence': round(float(s_confidence) * 100, 1),
            'topic_confidence': round(float(t_confidence) * 100, 1)
        })
    
    # In kết quả tổng hợp
    if VERBOSE:
        print("\n📊 KẾT QUẢ PHÂN TÍCH CHI TIẾT:")
        for i, result in enumerate(results, 1):
            print(f"\n🔍 Câu {i}:")
            print(f"   - Nội dung: {result['text']}")
            print(f"   - Cảm xúc: {result['sentiment']} ({result['sentiment_confidence']}%)")
            print(f"   - Chủ đề: {result['topic']} ({result['topic_confidence']}%)")
    
    return results

# ----------------------------------------------------
# Phân tích nhiều phản hồi hàng loạt (tối ưu cho /analyze_file)
# ----------------------------------------------------
def analyze_many_texts(text_list, batch_size: int = 64):
    """Phân tích hàng loạt nhiều phản hồi.
    Trả về: List[List[dict]] tương ứng với từng feedback ban đầu.
    """
    # 1) Tách câu cho từng feedback và flatten
    per_text_sentences = [split_feedback_text(t or '') for t in text_list]
    flat_sentences = []
    owners = []  # (text_idx, local_sentence_id)
    for idx, sents in enumerate(per_text_sentences):
        if not sents:
            continue
        for j, s in enumerate(sents, 1):
            flat_sentences.append(preprocess_text(s))
            owners.append((idx, j))

    if not flat_sentences:
        return [[] for _ in text_list]

    # 2) Chạy mô hình theo lô
    all_results = [None] * len(flat_sentences)
    for start in range(0, len(flat_sentences), batch_size):
        chunk = flat_sentences[start:start+batch_size]
        enc = tokenizer(
            chunk,
            truncation=True,
            padding='max_length',
            max_length=MAX_LEN,
            return_tensors='pt'
        )
        ids = enc['input_ids'].to(device)
        mask = enc['attention_mask'].to(device)
        with torch.no_grad():
            model_sent.eval(); model_topic.eval()
            s_out = model_sent(ids, mask)
            t_out = model_topic(ids, mask)
            s_probs = torch.softmax(s_out, dim=1)
            t_probs = torch.softmax(t_out, dim=1)
            s_labels = torch.argmax(s_probs, dim=1)
            t_labels = torch.argmax(t_probs, dim=1)
            s_conf = s_probs.gather(1, s_labels.view(-1,1)).squeeze(1)
            t_conf = t_probs.gather(1, t_labels.view(-1,1)).squeeze(1)

        for i in range(len(chunk)):
            global_idx = start + i
            s_label = int(s_labels[i].item()); t_label = int(t_labels[i].item())
            s_confidence = float(s_conf[i].item()); t_confidence = float(t_conf[i].item())
            sentiment = sentiment_map.get(s_label, "Không xác định")
            topic = topic_map.get(t_label, "Không xác định")

            # Rule-based điều chỉnh nhẹ (giống ở trên)
            txt_lower = chunk[i].lower()
            pos_words = [
                "hay", "tốt", "tuyệt vời", "hài lòng", "nhiệt tình", "thân thiện",
                "dễ hiểu", "ổn", "ok", "đẹp", "sạch", "chất lượng", "tận tâm"
            ]
            neg_words = [
                "tệ", "kém", "kém chất lượng", "đắt", "chán", "ồn", "bẩn",
                "cũ", "khó hiểu", "chậm", "lâu", "quá tải", "thiếu"
            ]
            teacher_words = [
                "thầy", "cô", "giảng viên", "gv", "dạy", "giảng dạy", 
                "gvcn", "chủ nhiệm", "giáo viên", "thầy cô", "đội ngũ giảng viên",
                "phương pháp giảng dạy", "kỹ năng sư phạm", "nhiệt tình"
            ]
            facility_words = [
                "cơ sở vật chất", "phòng học", "bàn ghế", "wifi", "máy chiếu", 
                "phòng", "csdl", "thư viện", "phòng thí nghiệm", "phòng lab",
                "ký túc xá", "ktx", "căn tin", "sân chơi", "thiết bị", "trang thiết bị"
            ]
            program_words = [
                "chương trình", "học phần", "môn", "khung chương trình", 
                "tín chỉ", "lịch học", "chất lượng đào tạo", "nội dung học", 
                "kiến thức", "kỹ năng", "thực hành", "lý thuyết", "bài tập"
            ]
            material_words = [
                "học liệu", "tài liệu", "website", "web", "lms", "moodle", 
                "bài giảng", "slides", "giáo trình", "sách", "tư liệu",
                "học phí", "chi phí", "lệ phí", "tiền học", "đóng học"
            ]
            def contains_any(words):
                return any(w in txt_lower for w in words)
            negation_tokens = ["không", "chưa", "chẳng", "chả"]
            has_negation = any(tok in txt_lower for tok in negation_tokens)
            pos_hit = contains_any(pos_words); neg_hit = contains_any(neg_words)
            if sentiment == "😐 Trung lập" or s_confidence < 0.6:
                if has_negation and pos_hit and not neg_hit:
                    sentiment = "😡 Tiêu cực"; s_confidence = 0.75
                elif neg_hit and not pos_hit:
                    sentiment = "😡 Tiêu cực"; s_confidence = 0.75
                elif pos_hit and not neg_hit and not has_negation:
                    sentiment = "😊 Tích cực"; s_confidence = 0.75
            if contains_any(teacher_words):
                topic = topic_map.get(0, topic); t_confidence = max(t_confidence, 0.75)
            elif contains_any(program_words):
                topic = topic_map.get(1, topic); t_confidence = max(t_confidence, 0.75)
            elif contains_any(facility_words):
                topic = topic_map.get(2, topic); t_confidence = max(t_confidence, 0.75)
            elif contains_any(material_words):
                topic = topic_map.get(3, topic); t_confidence = max(t_confidence, 0.75)

            all_results[global_idx] = {
                'text': chunk[i],
                'sentiment': sentiment,
                'topic': topic,
                'sentiment_confidence': round(float(s_confidence) * 100, 1),
                'topic_confidence': round(float(t_confidence) * 100, 1)
            }

    # 3) Gom lại theo feedback ban đầu
    grouped = [[] for _ in text_list]
    for (owner_idx, local_id), res in zip(owners, all_results):
        grouped[owner_idx].append(res)
    return grouped