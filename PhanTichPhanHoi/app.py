from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from data_processing import split_feedback_text, predict_feedback, analyze_feedback_text, analyze_many_texts
from typing import List, Optional, Dict
from fastapi.middleware.cors import CORSMiddleware # CẦN IMPORT NÀY CHO CORS
import torch
import json
import os
from datetime import datetime
from collections import defaultdict
from io import BytesIO
try:
    import pandas as pd  # optional, for Excel parsing
except Exception:  # pragma: no cover
    pd = None

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="Đại Nam Feedback Analysis API", 
    version="1.0",
    description="REST API để phân tích Cảm xúc và Chủ đề theo từng khía cạnh (Aspect-Level) sử dụng mô hình PhoBERT-Hybrid."
)

# ----------------------------------------------------
# 1. CẤU HÌNH MIDDLEWARE CORS (GIẢI QUYẾT LỖI 'Failed to fetch')
# ----------------------------------------------------
# Cho phép tất cả các nguồn gốc (origins) trong quá trình phát triển
# Bao gồm file:// (khi chạy index.html trực tiếp) và localhost
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # Cho phép tất cả các nguồn gốc
    allow_credentials=False,     # Không bật khi dùng '*' để tránh lỗi CORS
    allow_methods=["*"],         # Cho phép tất cả các phương thức (POST, GET)
    allow_headers=["*"],         # Cho phép tất cả các header
)
# ----------------------------------------------------

# ----------------------------------------------------
# 2. ĐỊNH NGHĨA CẤU TRÚC DỮ LIỆU (Sử dụng Pydantic)
# ----------------------------------------------------

class FeedbackInput(BaseModel):
    """Cấu trúc input: đoạn văn bản phản hồi."""
    text: str

class PartResult(BaseModel):
    """Cấu trúc output: kết quả phân tích cho từng phần câu."""
    part: str
    sentiment: str
    topic: str

class AnalysisResult(BaseModel):
    """Cấu trúc output tổng thể."""
    original_text: str
    analysis_parts: List[PartResult]

class FileAnalysisRow(BaseModel):
    index: int
    original_text: str
    analysis_parts: List[PartResult]
    student_id: Optional[str] = None
    sheet: Optional[str] = None

class FileAnalysisSummary(BaseModel):
    topic_sentiment: Dict[str, Dict[str, int]]  # {topic: {pos, neu, neg}}

class FileAnalysisResponse(BaseModel):
    total_rows: int
    summary: FileAnalysisSummary
    rows: List[FileAnalysisRow]
    sheets: Optional[List[str]] = None

class SurveyResponse(BaseModel):
    """Cấu trúc dữ liệu khảo sát"""
    student_id: str
    class_name: str
    # Câu hỏi Likert (1-5)
    q1: Optional[int] = None
    q2: Optional[int] = None
    q3: Optional[int] = None
    q4: Optional[int] = None
    q5: Optional[int] = None
    q6: Optional[int] = None
    q7: Optional[int] = None
    q8: Optional[int] = None
    q9: Optional[int] = None
    q10: Optional[int] = None
    q11: Optional[int] = None
    q12: Optional[int] = None
    q13: Optional[int] = None
    q14: Optional[int] = None
    q16: Optional[int] = None
    q17: Optional[int] = None
    q18: Optional[int] = None
    q19: Optional[int] = None
    q21: Optional[int] = None
    q22: Optional[int] = None
    q23: Optional[int] = None
    # Câu hỏi mở
    q15_gvcn_improve: Optional[str] = ""
    q20_teacher_improve: Optional[str] = ""
    q24_leader_improve: Optional[str] = ""
    q25_satisfied: Optional[str] = ""
    q26_unsatisfied: Optional[str] = ""
    q27_suggestions: Optional[str] = ""

# File lưu trữ responses và cache phân tích
RESPONSES_FILE = "survey_responses.json"
ANALYSIS_CACHE_FILE = "analysis_cache.json"

def load_responses():
    """Đọc responses từ file JSON"""
    if os.path.exists(RESPONSES_FILE):
        with open(RESPONSES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_response(response_data):
    """Lưu response mới vào file"""
    responses = load_responses()
    response_data['timestamp'] = datetime.now().isoformat()
    response_data['analyzed'] = False  # Đánh dấu chưa phân tích
    responses.append(response_data)
    with open(RESPONSES_FILE, 'w', encoding='utf-8') as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)

def load_analysis_cache():
    """Đọc cache phân tích từ file"""
    if os.path.exists(ANALYSIS_CACHE_FILE):
        with open(ANALYSIS_CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_analysis_cache(cache):
    """Lưu cache phân tích vào file"""
    with open(ANALYSIS_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def analyze_response_texts(response, student_id):
    """Phân tích tất cả câu trả lời mở của 1 response"""
    results = {
        'q25_satisfied': [],
        'q26_unsatisfied': [],
        'q15_gvcn_improve': [],
        'q20_teacher_improve': [],
        'q24_leader_improve': []
    }
    
    questions_to_analyze = ['q25_satisfied', 'q26_unsatisfied', 'q15_gvcn_improve', 
                            'q20_teacher_improve', 'q24_leader_improve']
    
    for q_key in questions_to_analyze:
        text = response.get(q_key, '')
        if text and text.strip():
            try:
                analysis = analyze_feedback_text(text)
                for part in analysis:
                    results[q_key].append({
                        "student_id": student_id,
                        "text": part['text'],
                        "sentiment": part['sentiment'],
                        "topic": part['topic']
                    })
            except:
                pass
    
    return results


# ----------------------------------------------------
# 3. ENDPOINT PHÂN TÍCH
# ----------------------------------------------------

@app.post("/analyze_text/", response_model=AnalysisResult, tags=["Analysis"])
async def analyze_feedback(feedback: FeedbackInput):
    """
    Phân tích một đoạn văn bản phản hồi, tách thành các câu 
    và dự đoán Cảm xúc/Chủ đề cho từng câu.
    """
    try:
        # Sử dụng hàm analyze_feedback_text mới để xử lý
        analysis_results = analyze_feedback_text(feedback.text)
        
        # Chuyển đổi kết quả sang định dạng response
        parts = []
        for result in analysis_results:
            parts.append(PartResult(
                part=result['text'],
                sentiment=result['sentiment'],
                topic=result['topic']
            ))
            
        return AnalysisResult(
            original_text=feedback.text,
            analysis_parts=parts
        )
        
    except Exception as e:
        print(f"Lỗi khi xử lý yêu cầu: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi xử lý yêu cầu: {str(e)}"
        )

# ----------------------------------------------------
# 3.1. ENDPOINT PHÂN TÍCH FILE (EXCEL/CSV)
# ----------------------------------------------------

def _normalize_sentiment(s: str) -> str:
    t = (s or '').lower()
    if 'tích cực' in t or 'positive' in t:
        return 'pos'
    if 'tiêu cực' in t or 'negative' in t:
        return 'neg'
    return 'neu'

def _aggregate_topic_sentiment(rows_parts: List[List[Dict[str, str]]]) -> Dict[str, Dict[str, int]]:
    agg: Dict[str, Dict[str, int]] = {}
    for parts in rows_parts:
        for p in parts:
            topic = p.get('topic') or 'Khác'
            sent = _normalize_sentiment(p.get('sentiment', ''))
            agg.setdefault(topic, {'pos': 0, 'neu': 0, 'neg': 0})
            agg[topic][sent] += 1
    return agg

@app.post("/analyze_file", response_model=FileAnalysisResponse, tags=["Analysis"])
async def analyze_feedback_file(
    file: UploadFile = File(...),
    text_column: str = Form('Phản hồi'),
    batch_size: int = Form(64),
    student_id_column: str = Form('Mã sinh viên')
):
    """Nhận file Excel/CSV chứa phản hồi sinh viên và phân tích theo Chủ đề & Cảm xúc.

    - text_column: tên cột chứa nội dung phản hồi. Mặc định 'Phản hồi'.
    - Hỗ trợ .xlsx/.xls (yêu cầu pandas + openpyxl) và .csv (UTF-8/UTF-8-SIG).
    """
    try:
        filename = (file.filename or '').lower()
        content = await file.read()

        texts: List[str] = []
        student_ids: List[Optional[str]] = []
        row_sheets: List[str] = []
        sheet_names_all: List[str] = []
        if filename.endswith(('.xlsx', '.xls')):
            if pd is None:
                raise HTTPException(status_code=400, detail="Thiếu pandas/openpyxl. Vui lòng cài đặt hoặc upload CSV.")
            # Đọc tất cả sheet
            all_sheets = pd.read_excel(BytesIO(content), sheet_name=None)
            candidate_cols = [text_column, 'Phản hồi', 'phan_hoi', 'feedback', 'noi_dung', 'Góp ý', 'Góp Ý']
            sid_candidates = [student_id_column, 'Mã sinh viên', 'Ma sinh vien', 'mssv', 'student_id', 'MSSV']
            added_any = False
            for sheet_name, df in all_sheets.items():
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue
                if sheet_name not in sheet_names_all:
                    sheet_names_all.append(str(sheet_name))
                found = next((c for c in candidate_cols if c in df.columns), None)
                if not found:
                    continue
                added_any = True
                sheet_texts = df[found].fillna('').astype(str).tolist()
                texts.extend(sheet_texts)
                row_sheets.extend([str(sheet_name)] * len(sheet_texts))
                sid_col = next((c for c in sid_candidates if c in df.columns), None)
                if sid_col:
                    sheet_sids = df[sid_col].astype(str).fillna('').tolist()
                    # canh kích thước
                    if len(sheet_sids) < len(sheet_texts):
                        sheet_sids += [''] * (len(sheet_texts) - len(sheet_sids))
                    student_ids.extend(sheet_sids)
                else:
                    student_ids.extend([''] * len(sheet_texts))
            if not added_any:
                raise HTTPException(status_code=400, detail=f"Không tìm thấy cột phản hồi trong bất kỳ sheet nào. Hãy đặt header là một trong {candidate_cols} hoặc chỉ định text_column.")
        else:
            # Xử lý như CSV
            raw = content.decode('utf-8-sig', errors='ignore').splitlines()
            import csv
            reader = csv.DictReader(raw)
            if not reader.fieldnames:
                raise HTTPException(status_code=400, detail="CSV không có header.")
            candidate_cols = [text_column, 'Phản hồi', 'phan_hoi', 'feedback', 'noi_dung']
            found = next((c for c in candidate_cols if c in reader.fieldnames), None)
            if not found:
                raise HTTPException(status_code=400, detail=f"Không tìm thấy cột phản hồi trong CSV. Thử tham số text_column hoặc đổi header thành một trong {candidate_cols}.")
            sid_candidates = [student_id_column, 'Mã sinh viên', 'Ma sinh vien', 'mssv', 'student_id', 'MSSV']
            sid_col = next((c for c in sid_candidates if c in (reader.fieldnames or [])), None)
            for row in reader:
                texts.append(str(row.get(found, '') or ''))
                student_ids.append(str(row.get(sid_col, '') or '') if sid_col else '')
                row_sheets.append('CSV')
            sheet_names_all = ['CSV']

        # Loại bỏ dòng trống, chuẩn hóa và dedup để tăng tốc
        normalized = [(i, (texts[i] or '').strip()) for i in range(len(texts))]
        index_map: Dict[str, List[int]] = {}
        unique_texts: List[str] = []
        for idx, t in normalized:
            if not t:
                index_map.setdefault('__EMPTY__', []).append(idx)
                continue
            key = t
            if key not in index_map:
                index_map[key] = []
                unique_texts.append(t)
            index_map[key].append(idx)

        # Phân tích theo lô cho danh sách UNIQUE
        unique_results = analyze_many_texts(unique_texts, batch_size=max(8, int(batch_size) if str(batch_size).isdigit() else 64))

        # Map kết quả về thứ tự gốc
        rows: List[FileAnalysisRow] = []
        rows_parts_for_agg: List[List[Dict[str, str]]] = []
        # Kết quả cho empty
        empty_parts: List[Dict[str, str]] = []
        # Duyệt theo thứ tự văn bản duy nhất
        uniq_iter = iter(unique_results)
        uniq_map: Dict[str, List[Dict[str, str]]] = {}
        for key in index_map.keys():
            if key == '__EMPTY__':
                continue
            uniq_map[key] = next(uniq_iter)

        for i, original in enumerate(texts):
            if not (original or '').strip():
                rows.append(FileAnalysisRow(index=i, original_text=original or '', analysis_parts=[], student_id=(student_ids[i] if i < len(student_ids) else ''), sheet=(row_sheets[i] if i < len(row_sheets) else None)))
                rows_parts_for_agg.append(empty_parts)
            else:
                parts_raw = uniq_map.get((original or '').strip(), [])
                parts_struct = [PartResult(part=p['text'], sentiment=p['sentiment'], topic=p['topic']) for p in parts_raw]
                rows.append(FileAnalysisRow(index=i, original_text=original or '', analysis_parts=parts_struct, student_id=(student_ids[i] if i < len(student_ids) else ''), sheet=(row_sheets[i] if i < len(row_sheets) else None)))
                rows_parts_for_agg.append(parts_raw)

        agg = _aggregate_topic_sentiment(rows_parts_for_agg)

        return FileAnalysisResponse(
            total_rows=len(texts),
            summary=FileAnalysisSummary(topic_sentiment=agg),
            rows=rows,
            sheets=sheet_names_all if sheet_names_all else None
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích file: {str(e)}")

# ----------------------------------------------------
# 4. ENDPOINT SURVEY MANAGEMENT
# ----------------------------------------------------

@app.post("/submit_survey", tags=["Survey"])
async def submit_survey(survey: SurveyResponse):
    """Nhận và lưu phản hồi khảo sát từ sinh viên"""
    try:
        save_response(survey.dict())
        return {"status": "success", "message": "Đã lưu phản hồi thành công"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lưu: {str(e)}")

@app.get("/survey_stats", tags=["Survey"])
async def get_survey_stats():
    """Lấy thống kê và phân tích toàn bộ khảo sát (có cache để tăng tốc)"""
    try:
        responses = load_responses()
        
        if not responses:
            return {"total_responses": 0, "message": "Chưa có dữ liệu khảo sát"}
        
        # Thống kê câu hỏi Likert (nhanh, không cần cache)
        likert_stats = {}
        likert_questions = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 
                           'q12', 'q13', 'q14', 'q16', 'q17', 'q18', 'q19', 'q21', 'q22', 'q23']
        
        for q in likert_questions:
            values = [r.get(q) for r in responses if r.get(q)]
            if values:
                likert_stats[q] = {
                    "average": round(sum(values) / len(values), 2),
                    "distribution": {
                        "1": values.count(1),
                        "2": values.count(2),
                        "3": values.count(3),
                        "4": values.count(4),
                        "5": values.count(5)
                    }
                }
        
        # Load cache phân tích AI
        analysis_cache = load_analysis_cache()
        cache_updated = False
        
        # Phân tích câu hỏi mở bằng AI (sử dụng cache)
        open_feedback_analysis = {
            "satisfied": [],
            "unsatisfied": [],
            "suggestions": [],
            "gvcn_improve": [],
            "teacher_improve": [],
            "leader_improve": []
        }
        
        topic_counts = {
            "satisfied": defaultdict(int),
            "unsatisfied": defaultdict(int),
            "gvcn_improve": defaultdict(int),
            "teacher_improve": defaultdict(int),
            "leader_improve": defaultdict(int)
        }
        
        for idx, response in enumerate(responses):
            student_id = response.get('student_id', f'unknown_{idx}')
            cache_key = f"{student_id}_{response.get('timestamp', '')}"
            
            # Kiểm tra cache
            if cache_key in analysis_cache:
                # Dùng kết quả từ cache (NHANH!)
                cached_result = analysis_cache[cache_key]
            else:
                # Chưa có cache, phân tích mới (CHẬM - chỉ chạy 1 lần)
                print(f"⚡ Phân tích mới cho {student_id}...")
                cached_result = analyze_response_texts(response, student_id)
                analysis_cache[cache_key] = cached_result
                cache_updated = True
            
            # Gộp kết quả từ cache vào output
            for key in ['q25_satisfied', 'q26_unsatisfied', 'q15_gvcn_improve', 
                       'q20_teacher_improve', 'q24_leader_improve']:
                output_key = key.replace('q25_', '').replace('q26_', '').replace('q15_', '').replace('q20_', '').replace('q24_', '')
                if 'satisfied' in key:
                    output_key = 'satisfied'
                elif 'unsatisfied' in key:
                    output_key = 'unsatisfied'
                elif 'gvcn' in key:
                    output_key = 'gvcn_improve'
                elif 'teacher' in key:
                    output_key = 'teacher_improve'
                elif 'leader' in key:
                    output_key = 'leader_improve'
                
                for item in cached_result.get(key, []):
                    open_feedback_analysis[output_key].append(item)
                    topic_counts[output_key][item['topic']] += 1
            
            # Đề xuất (câu 27) không cần phân tích AI
            if response.get('q27_suggestions'):
                open_feedback_analysis['suggestions'].append({
                    "student_id": student_id,
                    "text": response['q27_suggestions']
                })
        
        # Lưu cache nếu có cập nhật
        if cache_updated:
            save_analysis_cache(analysis_cache)
            print(f"✅ Đã lưu cache phân tích!")
        
        return {
            "total_responses": len(responses),
            "likert_statistics": likert_stats,
            "open_feedback_analysis": open_feedback_analysis,
            "top_satisfied_topics": dict(sorted(topic_counts['satisfied'].items(), key=lambda x: x[1], reverse=True)[:5]),
            "top_unsatisfied_topics": dict(sorted(topic_counts['unsatisfied'].items(), key=lambda x: x[1], reverse=True)[:5]),
            "top_gvcn_improve": dict(sorted(topic_counts['gvcn_improve'].items(), key=lambda x: x[1], reverse=True)[:5]),
            "top_teacher_improve": dict(sorted(topic_counts['teacher_improve'].items(), key=lambda x: x[1], reverse=True)[:5]),
            "top_leader_improve": dict(sorted(topic_counts['leader_improve'].items(), key=lambda x: x[1], reverse=True)[:5])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi phân tích: {str(e)}")

# Thêm endpoint mới để kiểm tra trạng thái server
@app.get("/health", tags=["System"])
async def health_check():
    """Kiểm tra trạng thái hoạt động của server"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return {"status": "ok", "device": str(device)}

# ----------------------------------------------------
# 6. ENDPOINT PHỤ TRỢ DASHBOARD
# ----------------------------------------------------

@app.get("/survey_records", tags=["Survey"])
async def get_survey_records():
    """Trả về toàn bộ bản ghi khảo sát (phục vụ dashboard lọc theo năm/học kỳ/chủ đề)"""
    try:
        return {"items": load_responses()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi đọc dữ liệu: {str(e)}")