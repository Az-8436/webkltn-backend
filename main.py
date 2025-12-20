import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Form
from ocr import extract_info_from_image
from pydantic import BaseModel
import pickle
import json
import joblib
import numpy as np
from models.Neural_Network import forward_prop
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId # Để xử lý ID của MongoDB
from datetime import datetime
from typing import Optional
import google.generativeai as genai
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import os
from dotenv import load_dotenv

load_dotenv()

my_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=my_api_key)
model = genai.GenerativeModel('gemini-2.5-flash-lite')
# --- CẤU HÌNH MONGODB ---
# Kết nối đến MongoDB (mặc định là cổng 27017)
MONGO_DETAILS = "mongodb+srv://ngothimyha271:ngothimyha271@updatedata.f1pphvr.mongodb.net/?appName=updatedata" 
client = AsyncIOMotorClient(MONGO_DETAILS)


# genai.configure(api_key="")
# model = genai.GenerativeModel('gemini-2.5-flash')

# Tạo database tên là "medical_db"
db = client.medical_db 
# Tạo collection (bảng) tên là "patient_records"
collection = db.patient_records
collection_glucose = db.glucose_records

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



class ChatRequest(BaseModel):
    question: str
    glucose_value: int
    measure_type: str

class PatientInfo(BaseModel):
    id: str
    name: str
    gender: str
    age: int
    height: int
    weight: int
    systolicBloodPressure: int
    diastolicBloodPressure: int
    heartRate: int
    bmi: float

class BloodTests(BaseModel):
    cholesterol: float
    hdl: float
    ldl: float
    triglycerid: float
    creatinin: float
    hba1c: float
    ure: float
    # gender: int
    # age: int
    # bmi: float
    vldl: float

# Định nghĩa khuôn dữ liệu để lưu vào MongoDB
class SaveRecordInput(BaseModel):
    patient_info: dict       # Thông tin bệnh nhân
    blood_tests: dict        # Chỉ số xét nghiệm
    units: dict
    ai_diagnosis: str        # Kết luận của AI
    doctor_diagnosis: str    # Kết luận của Bác sĩ (Mới thêm)
    created_at: Optional[str] = None

# Hàm này giúp chuyển dữ liệu từ MongoDB (dạng thô) sang JSON (để trả về Frontend)
def record_helper(record) -> dict:
    return {
        "id": str(record["_id"]), # ID luôn phải có
        
        # Lấy nguyên cục patient_info (chứa tên, tuổi, giới tính...)
        "patient_info": record.get("patient_info", {}), 
        
        # Lấy nguyên cục xét nghiệm
        "blood_tests": record.get("blood_tests", {}),
        
        # Kết quả chẩn đoán
        "ai_diagnosis": record.get("ai_diagnosis", "Chưa có kết quả"),
        "doctor_diagnosis": record.get("doctor_diagnosis", ""), # Mặc định là chuỗi rỗng nếu bác sĩ chưa nhập
        
        # Ngày giờ khám
        "created_at": record.get("created_at", "")
    }

class PredictionInput(BaseModel):
    patient_info: PatientInfo
    blood_tests: BloodTests



@app.get("/")
def home():
    return {"message": "Hello World"}

@app.post("/ocr")
async def upload_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        result = extract_info_from_image(image_bytes)

        return {"status": "success", "data": result}

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/predict-disease")
async def predict(data: PredictionInput):
        tests = data.blood_tests
        info = data.patient_info
        # return {"status": "success", "data": tests}

        # chuyển gender
        if info.gender in 'Nữ':
            info.gender = 0
        elif info.gender in 'Nam':
            info.gender = 1
        else:
            info.gender = None
        # return {"status": "success", "data": tests}
        # age
        # if info["birth_date"]:
        #     try:
        #         year = int(info["birth_date"][-4:])
        #         tests["age"] = 2025 - year
        #     except:
        #         tests["age"] = None
        # else:
        #     tests["age"] = None

        # default bmi nếu thiếu
        # tests["bmi"] = 31

        # VLDL
        # if tests.triglycerid is not None:
        #     vldl = round(float(tests.triglycerid) / 2.2, 2)
        #     tests.vldl = vldl
        
            
     
        # nếu ure thiếu
        # if tests.ure is None:
        #     tests.ure = 5.0

        data_for_model_dia = np.array([[info.gender, info.age, tests.ure, tests.creatinin, tests.hba1c, tests.cholesterol, tests.triglycerid, tests.hdl, tests.ldl, tests.vldl, info.bmi]])
        # return {"status": "success", "data": data_for_model.tolist}
        scaler_dia =  joblib.load('scaler_cua_be.pkl')
        normalized_data_dia = scaler_dia.transform(data_for_model_dia).T
        # return {"status": "success", "data": normalized_data.tolist()}

        with open('weights_bias_diabetes_with_batch_gradient_descent.pkl', 'rb') as f:
            params_dia = pickle.load(f)

    
        W1_d = params_dia['W1']
        b1_d = params_dia['b1']
        W2_d = params_dia['W2']
        b2_d = params_dia['b2']
        W3_d = params_dia['W3']
        b3_d = params_dia['b3']



        _, _, _, _, _, A3_d = forward_prop(normalized_data_dia, W1_d, b1_d, W2_d, b2_d, W3_d, b3_d)
        pre_d = np.argmax(A3_d, 0)
        if pre_d[0] == 0:
            result_d = "Bệnh nhân đường huyết bình thường"
        elif pre_d[0] == 1: 
            result_d = 'Bệnh nhân bị tiền đái tháo đường'
        elif pre_d[0] == 2:
            result_d = 'Bệnh nhân bị đái tháo đường'

        data_for_model_hyper = np.array([[info.gender, info.age, info.height, info.weight, info.systolicBloodPressure, info.diastolicBloodPressure, info.heartRate, info.bmi]])
        scaler_hyper = joblib.load('scaler_cua_hypertension.pkl')
        normalized_data_hyper = scaler_hyper.transform(data_for_model_hyper).T
        
        with open('weights_bias_hypertension_0.97.pkl', 'rb') as f:
            params_h = pickle.load(f)

        W1_h = params_h['W1']
        b1_h = params_h['b1']
        W2_h = params_h['W2']
        b2_h = params_h['b2']
        W3_h = params_h['W3']
        b3_h = params_h['b3']

        _, _, _, _, _, A3_h = forward_prop(normalized_data_hyper, W1_h, b1_h, W2_h, b2_h, W3_h, b3_h)

        pre_h = np.argmax(A3_h, 0)


        if pre_h[0] == 0:
            result_h = "Bệnh nhân không bị tăng huyết áp"
        elif pre_h[0] == 1:
            result_h = "Bệnh nhân bị tiền tăng huyết áp"
        elif pre_h[0] == 2:
            result_h = "Bệnh nhân bị tăng huyết áp"
        elif pre_h[0] == 3:
            result_h = 'Bệnh nhân bị tăng huyết áp'

        data_for_model_lipid = np.array([[tests.cholesterol, tests.triglycerid, tests.hdl, tests.ldl]])
        # return {"status": "success", "data": data_for_model.tolist}
        scaler_lipid =  joblib.load('scaler_mo_mau.pkl')
        normalized_data_lipid = scaler_lipid.transform(data_for_model_lipid).T
        # return {"status": "success", "data": normalized_data.tolist()}

        with open('weights_bias_mo_mau_with_stochastic_gradient_descent.pkl', 'rb') as f:
            params_dia = pickle.load(f)

    
        W1_l = params_dia['W1']
        b1_l = params_dia['b1']
        W2_l = params_dia['W2']
        b2_l = params_dia['b2']
        W3_l = params_dia['W3']
        b3_l = params_dia['b3']



        _, _, _, _, _, A3_l = forward_prop(normalized_data_lipid, W1_l, b1_l, W2_l, b2_l, W3_l, b3_l)
        pre_l = np.argmax(A3_l, 0)
        if pre_l[0] == 0:
            result_l = "Bệnh nhân có chỉ số lipid máu bình thường"
        elif pre_l[0] == 1: 
            result_l = 'Bệnh nhân bị rối loạn lipid máu mức độ nhẹ'
        elif pre_l[0] == 2:
            result_l = 'Bệnh nhân bị rối loạn lipid máu'

        warning_msg = ""
        if info.systolicBloodPressure >= 180 or info.diastolicBloodPressure >= 110:
            warning_msg = "\n⚠️ CẢNH BÁO: Huyết áp mức nguy hiểm (>=180/110 mmHg). Cần xử trí cấp cứu ngay!"

    # Dùng ký tự \n để xuống dòng

        combined_result = f"{result_d};\n{result_h};\n{result_l}.{warning_msg}"
        return {"status": "success", "data": combined_result}
    
# @app.post('/predict/hypertension')
# async def predict_hypertension(data: PredictionInput):

@app.post("/predict-hypertension")
async def predict_hypertension(data: PredictionInput):
        info = data.patient_info

        if info.gender in 'Nữ':
            info.gender = 0
        elif info.gender in 'Nam':
            info.gender = 1
        else:
            info.gender = None
        data_for_model = np.array([[info.gender, info.age, info. height, info.weight, info.systolicBloodPressure, info.diastolicBloodPressure, info.heartRate, info.bmi]])
        scaler = joblib.load('scaler_cua_hypertension.pkl')
        normalized_data = scaler.transform(data_for_model).T

        with open('weights_bias_hypertension_0.97.pkl', 'rb') as f:
            params = pickle.load(f)

        W1 = params['W1']
        b1 = params['b1']
        W2 = params['W2']
        b2 = params['b2']
        W3 = params['W3']
        b3 = params['b3']

        _, _, _, _, _, A3 = forward_prop(normalized_data, W1, b1, W2, b2, W3, b3)

        pre = np.argmax(A3, 0)

        if pre[0] == 0:
            result = "Bệnh nhân huyết áp bình thường"
        elif pre[0] == 1:
            result = "Benh nhan co nguy co bi tien huyet ap"
        elif pre[0] == 2:
            result = "Benh nhan bi huyet ap loai 1"
        elif pre[0] == 3:
            result = 'Benh nha bi huyet ap loai 2'
        return {"status": "success", "data": result}


# --- API 1: LƯU HỒ SƠ (Dùng ở trang UploadImage) ---
@app.post("/api/save-record")
async def save_record(data: SaveRecordInput):
    record_dict = data.dict()
    
    # Tự động lấy giờ hiện tại nếu không gửi lên
    if not record_dict.get("created_at"):
        record_dict["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    new_record = await collection.insert_one(record_dict)
    
    return {"status": "success", "message": "Đã lưu hồ sơ thành công", "id": str(new_record.inserted_id)}

# --- API 2: LẤY DANH SÁCH ---
@app.get("/api/get-records")
async def get_records():
    records = []
    try:
        # Lấy dữ liệu và sắp xếp mới nhất lên đầu
        async for record in collection.find().sort("_id", -1):
            processed_record = record_helper(record)
            if processed_record: # Chỉ thêm nếu convert thành công
                records.append(processed_record)
                
        return {"status": "success", "data": records}
        
    except Exception as e:
        # In lỗi ra Terminal để bé biết đường sửa
        print(f"🔥 LỖI 500 Ở GET-RECORDS: {e}")
        return {"status": "error", "message": str(e)}

# --- API THỐNG KÊ DASHBOARD ---
@app.get("/api/dashboard")
async def get_dashboard_stats():
    total_patients = 0
    diabetes_count = 0
    hypertension_count = 0
    lipid_count = 0
    
    # Dùng dictionary để gom nhóm theo ngày cho biểu đồ
    # Cấu trúc: { "2025-11-24": { "date": "24/11", "diabetes": 1, "hypertension": 0 } }
    chart_data_dict = {} 

    async for record in collection.find():
        total_patients += 1
        
        # 1. Lấy thông tin chẩn đoán và ngày tháng
        diagnosis = record.get("ai_diagnosis", "").lower()
        created_at = record.get("created_at", "") # Ví dụ: "2025-11-24 10:30:00"
        
        # 2. Phân loại bệnh (Dựa vào chuỗi kết quả AI trả về)
        is_diabetes = "đái tháo đường" in diagnosis 
        is_hypertension = "tăng huyết áp" in diagnosis 
        is_lipid = "lipid máu" in diagnosis 
        
        # Logic đếm: Nếu chuỗi kết quả có chữ "không bị" thì không đếm là bệnh
        if "đường huyết bình thường" not in diagnosis and "tiền tiểu đường" not in diagnosis:
            if is_diabetes:
                diabetes_count += 1
        if "không bị tăng huyết áp" not in diagnosis and "tiền tăng huyết áp" not in diagnosis:
            if is_hypertension:
                hypertension_count += 1

        if "lipid máu bình thường" not in diagnosis and "lipid máu mức độ nhẹ" not in diagnosis:
            if is_lipid:
                lipid_count += 1


        # if "Bệnh nhân bị tiểu đường không phụ thuộc insulin - type 2" in diagnosis:
        #     diabetes_count += 1
        # if "Bệnh nhân bị tăng huyết áp cấp độ 1" in diagnosis and "Bệnh nhân bị tăng huyết áp cấp độ 2" in diagnosis:
        #     hypertension_count += 1
        
        # 3. Xử lý dữ liệu cho biểu đồ (Gom theo ngày)
        # Lấy phần ngày YYYY-MM-DD (bỏ phần giờ)
        date_str = created_at.split(" ")[0] if created_at else "N/A"
        
        if date_str not in chart_data_dict:
            # Tạo mới nếu ngày này chưa có trong danh sách
            chart_data_dict[date_str] = {
                "name": date_str, # Tên trục hoành
                "diabetes": 0,
                "hypertension": 0,
                "lipid": 0,
                "total": 0
            }
        
        # Cộng dồn số liệu vào ngày tương ứng
        chart_data_dict[date_str]["total"] += 1
        # if "không bị" not in diagnosis and "khong bi" not in diagnosis and "tiền tăng huyết áp" not in diagnosis and "tien huyet ap" not in diagnosis:
        #     if is_diabetes:
        #         chart_data_dict[date_str]["diabetes"] += 1
        #     if is_hypertension:
        #         chart_data_dict[date_str]["hypertension"] += 1

        if "đường huyết bình thường" not in diagnosis and "tiền tiểu đường" not in diagnosis:
            if is_diabetes:
                chart_data_dict[date_str]["diabetes"] += 1
        if "không bị tăng huyết áp" not in diagnosis and "tiền tăng huyết áp" not in diagnosis:
            if is_hypertension:
                chart_data_dict[date_str]["hypertension"] += 1
        if "lipid máu bình thường" not in diagnosis and "lipid máu mức độ nhẹ" not in diagnosis:
            if is_lipid:
                chart_data_dict[date_str]["lipid"] += 1
    # 4. Chuyển dictionary thành list và sắp xếp theo ngày tăng dần
    chart_list = sorted(list(chart_data_dict.values()), key=lambda x: x['name'])

    return {
        "status": "success",
        "summary": {
            "total": total_patients,
            "diabetes": diabetes_count,
            "hypertension": hypertension_count,
            "lipid": lipid_count
        },
        "chart_data": chart_list
    }






# --- API LƯU TRỮ ---
# @app.post("/api/glucose/add")
# async def add_glucose(record: GlucoseRecord):
#     if not record.created_at:
#         record.created_at = datetime.now().strftime("%d/%m/%Y %H:%M")
#     await collection_glucose.insert_one(record.dict())
#     return {"status": "success"}
# --- MODEL DỮ LIỆU ---
class GlucoseRecord(BaseModel):
    patient_id: str
    value: int
    measure_type: str
    note: str = ""
    created_at: str = ""

@app.post("/api/glucose/add")
async def add_glucose(record: GlucoseRecord):
    # 1. Tự động lấy giờ nếu thiếu
    if not record.created_at:
        record.created_at = datetime.now().strftime("%d/%m/%Y %H:%M")
    
    # 2. Tìm bệnh nhân theo mã hồ sơ (record.patient_id) 
    # và PUSH (nhét) dữ liệu mới vào mảng "glucose_history"
    result = await collection.update_one(
        {"patient_info.id": record.patient_id}, # Tìm người có mã này
        {"$push": {"glucose_history": record.dict()}} # Thêm vào danh sách
    )

    # Kiểm tra xem có tìm thấy người để lưu không
    if result.matched_count == 0:
         raise HTTPException(status_code=404, detail="Không tìm thấy hồ sơ bệnh nhân này!")

    return {"status": "success", "message": "Đã lưu vào hồ sơ bệnh nhân"}

# @app.get("/api/glucose/history")
# async def get_glucose_history():
#     cursor = collection_glucose.find({}, {"_id": 0}).sort("_id", -1).limit(20)
#     history = await cursor.to_list(length=20)
#     return {"data": history[::-1]}

@app.get("/api/glucose/history")
async def get_glucose_history(patient_id: str):
    # 1. Tìm bệnh nhân và chỉ lấy trường glucose_history thôi cho nhẹ
    patient = await collection.find_one(
        {"patient_info.id": patient_id}, 
        {"glucose_history": 1, "_id": 0}
    )

    # 2. Nếu không tìm thấy bệnh nhân hoặc chưa có lịch sử đo nào
    if not patient or "glucose_history" not in patient:
        return {"data": []}

    # 3. Lấy dữ liệu và đảo ngược lại (Mới nhất lên đầu)
    history = patient["glucose_history"]
    return {"data": history[::-1]}

# --- API CHATBOT TƯ VẤN ---
@app.post("/api/chat/advice")
async def get_diet_advice(req: ChatRequest):
    try:
        # Tạo ngữ cảnh cho AI hiểu tình trạng bệnh nhân
        context = ""
        if req.glucose_value > 0:
            type_text = "lúc đói (trước ăn)" if req.measure_type == "fasting" else "sau ăn 2 giờ"
            context = f"Tôi là bệnh nhân có đường huyết {req.glucose_value} mg/dL đo vào lúc {type_text}. "
        
        prompt = (f"{context}Câu hỏi: '{req.question}'. "
                  f"Hãy trả lời ngắn gọn, thân thiện như bác sĩ gia đình. ")
                #   f"Đưa ra lời khuyên ăn uống hoặc thực đơn cụ thể cho chỉ số đường huyết này.")
        
        response = model.generate_content(prompt)
        return {"reply": response.text}
    except Exception as e:
        print(e)
        return {"reply": "Hệ thống AI đang bận, bạn thử lại sau nhé!"}
    


# --- API DỰ BÁO ĐƯỜNG HUYẾT ---
# class PredictionRequest(BaseModel):
#     measure_type: str # Chỉ dự báo dựa trên cùng loại (VD: Chỉ dùng lịch sử 'lúc đói' để dự báo 'lúc đói')
class PredictionRequest(BaseModel):
    measure_type: str 
    patient_id: str  # <--- Quan trọng: Phải có dòng này
# @app.post("/api/predict/glucose")
# async def predict_glucose(req: PredictionRequest):
#     # 1. Lấy dữ liệu (Giữ nguyên code cũ của bé)
#     cursor = collection_glucose.find({"measure_type": req.measure_type})
#     records = await cursor.to_list(length=100)
    
#     if len(records) < 3:
#         return {
#             "can_predict": False, 
#             "message": "Cần ít nhất 3 lần đo trong lịch sử để dự báo!"
#         }

#     # 2. Xử lý dữ liệu (Giữ nguyên logic chuẩn hóa thời gian của bé)
#     df = pd.DataFrame(records)
#     df['date_obj'] = pd.to_datetime(df['created_at'], dayfirst=True, format='mixed')
#     df = df.sort_values(by='date_obj')

#     # Mốc thời gian bắt đầu
#     start_time = df['date_obj'].iloc[0].timestamp()
    
#     # Tính X (đầu vào) và y (kết quả)
#     df['timestamp'] = df['date_obj'].map(pd.Timestamp.timestamp)
#     df['days_passed'] = (df['timestamp'] - start_time) / (24 * 3600)
    
#     X = df[['days_passed']].values
#     y = df['value'].values 

#     # 3. Huấn luyện mô hình
#     model = LinearRegression()
#     model.fit(X, y)

#     # --- 4. DỰ BÁO 7 NGÀY (PHẦN MỚI SỬA) ---
#     predictions = []
#     current_date = datetime.now()
#     last_real_value = df['value'].iloc[-1] # Lấy giá trị thật cuối cùng để tham chiếu

#     for i in range(1, 8): # Chạy từ ngày mai (1) đến 7 ngày sau (8)
#         future_date = current_date + timedelta(days=i)
#         future_ts = future_date.timestamp()
        
#         # Chuẩn hóa thời gian tương lai theo mốc start_time cũ
#         future_days_passed = (future_ts - start_time) / (24 * 3600)
        
#         # Dự đoán
#         pred_val = model.predict([[future_days_passed]])[0]
#         result = int(pred_val)

#         # --- LOGIC CHẶN SỐ (Logic cũ của bé nhưng áp dụng trong vòng lặp) ---
#         if result < 50:
#             # Nếu giảm quá sâu, giả định nó đi ngang bằng giá trị cuối cùng
#             result = int(last_real_value) 
#         elif result > 600:
#             result = 600
        
#         predictions.append({
#             "date": future_date.strftime("%d/%m"), # Format ngày tháng cho đẹp (VD: 05/12)
#             "value": result
#         })
        
#         # Cập nhật giá trị tham chiếu cho vòng lặp sau (để đường dây mượt hơn nếu cần)
#         # last_real_value = result 

#     return {
#         "can_predict": True,
#         "predictions": predictions, # Trả về cả danh sách 7 ngày
#         "message": f"Đã dự báo xu hướng cho 7 ngày tới."}

@app.post("/api/predict/glucose")
async def predict_glucose(req: PredictionRequest):
    # 1. Lấy lịch sử từ hồ sơ bệnh nhân
    patient = await collection.find_one(
        {"patient_info.id": req.patient_id}, 
        {"glucose_history": 1, "_id": 0}
    )
    
    # Nếu chưa có dữ liệu gì hết
    if not patient or "glucose_history" not in patient:
         return {"can_predict": False, "message": "Chưa có dữ liệu lịch sử để dự báo!"}

    all_records = patient["glucose_history"]

    # 2. Lọc ra các lần đo đúng loại yêu cầu (VD: chỉ lấy 'fasting')
    # Vì trong glucose_history chứa lộn xộn cả đói cả no
    records = [r for r in all_records if r.get("measure_type") == req.measure_type]
    
    # 3. Kiểm tra đủ dữ liệu (ít nhất 3 điểm)
    if len(records) < 3:
        return {
            "can_predict": False, 
            "message": f"Cần ít nhất 3 lần đo '{req.measure_type}' để dự báo!"
        }

    # --- ĐOẠN DƯỚI NÀY GIỮ NGUYÊN CODE CŨ CỦA BÉ ---
    df = pd.DataFrame(records)
    df['date_obj'] = pd.to_datetime(df['created_at'], dayfirst=True, format='mixed')
    df = df.sort_values(by='date_obj')

    start_time = df['date_obj'].iloc[0].timestamp()
    df['timestamp'] = df['date_obj'].map(pd.Timestamp.timestamp)
    df['days_passed'] = (df['timestamp'] - start_time) / (24 * 3600)
    
    X = df[['days_passed']].values
    y = df['value'].values 

    model = LinearRegression()
    model.fit(X, y)

    predictions = []
    current_date = datetime.now()
    last_real_value = df['value'].iloc[-1]

    for i in range(1, 8):
        future_date = current_date + timedelta(days=i)
        future_ts = future_date.timestamp()
        future_days_passed = (future_ts - start_time) / (24 * 3600)
        
        pred_val = model.predict([[future_days_passed]])[0]
        result = int(pred_val)

        if result < 50:
            result = int(last_real_value) 
        elif result > 600:
            result = 600
        
        predictions.append({
            "date": future_date.strftime("%d/%m"),
            "value": result
        })

    return {
        "can_predict": True,
        "predictions": predictions,
        "message": "Đã dự báo xu hướng cho 7 ngày tới."
    }

# ---------------------------------------------------------
# API: TÌM BỆNH NHÂN THEO MÃ HỒ SƠ (Dùng cho Login)
# ---------------------------------------------------------
def patient_helper(patient) -> dict:
    return {
        "id": str(patient["_id"]), # Chuyển ObjectId thành chuỗi
        "patient_info": patient.get("patient_info"),
        "blood_tests": patient.get("blood_tests"),
        "units": patient.get("units"),
    }
@app.get("/api/patients/{patient_id}")
async def get_patient_by_id(patient_id: str):
    # LƯU Ý QUAN TRỌNG:
    # Vì id nằm trong patient_info, nên query phải là "patient_info.id"
    patient = await collection.find_one({"patient_info.id": patient_id})
    
    if patient:
        return patient_helper(patient)
    
    # Nếu không tìm thấy
    raise HTTPException(status_code=404, detail="Không tìm thấy mã hồ sơ này")

# --- API LẤY LỊCH SỬ ĐƯỜNG HUYẾT CHO BÁC SĨ ---
# API này giúp bác sĩ xem biểu đồ đường huyết của bệnh nhân trong trang Chi tiết hồ sơ
@app.get("/api/glucose/history/{patient_id}")
async def get_glucose_history_by_id(patient_id: str):
    # Tìm bệnh nhân theo mã hồ sơ
    patient = await collection.find_one(
        {"patient_info.id": patient_id}, 
        {"glucose_history": 1, "_id": 0}
    )

    # Nếu không tìm thấy hoặc chưa có lịch sử
    if not patient or "glucose_history" not in patient:
        return {"status": "success", "data": []}

    # Lấy dữ liệu và sắp xếp theo ngày tăng dần để vẽ biểu đồ cho đẹp
    history = patient["glucose_history"]
    
    # Sắp xếp theo thời gian (Cũ -> Mới)
    # Lưu ý: Cần đảm bảo created_at lưu đúng format để sort được, hoặc sort ở frontend cũng được
    # Ở đây mình trả về nguyên danh sách, frontend sẽ lo phần hiển thị
    return {"status": "success", "data": history}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
