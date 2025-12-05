import uvicorn
from fastapi import FastAPI
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
import os

from dotenv import load_dotenv

load_dotenv()

# Cấu hình API key của Gemini




# --- CẤU HÌNH MONGODB ---
# Kết nối đến MongoDB (mặc định là cổng 27017)
MONGO_DETAILS = "mongodb+srv://ngothimyha271:ngothimyha271@updatedata.f1pphvr.mongodb.net/?appName=updatedata" 
client = AsyncIOMotorClient(MONGO_DETAILS)

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

# --- MODEL DỮ LIỆU ---
class GlucoseRecord(BaseModel):
    value: int
    measure_type: str
    note: str = ""
    created_at: str = ""

class ChatRequest(BaseModel):
    question: str
    glucose_value: int
    measure_type: str

class PatientInfo(BaseModel):
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
    return {"message": "Backend Running"}

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
            result_d = "Bệnh nhân không bị tiểu đường"
        elif pre_d[0] == 1: 
            result_d = 'Bệnh nhân có nguy cơ tiền tiểu đường'
        elif pre_d[0] == 2:
            result_d = 'Bệnh nhân bị tiểu đường không phụ thuộc insulin (type 2)'

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
            result_h = "Bệnh nhân có nguy cơ tiền tăng huyết áp"
        elif pre_h[0] == 2:
            result_h = "Bệnh nhân bị tăng huyết áp cấp độ 1"
        elif pre_h[0] == 3:
            result_h = 'Bệnh nhân bị tăng huyết áp cấp độ 2'

        combined_result = f"{result_d} và {result_h}"
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
    
    # Dùng dictionary để gom nhóm theo ngày cho biểu đồ
    # Cấu trúc: { "2025-11-24": { "date": "24/11", "diabetes": 1, "hypertension": 0 } }
    chart_data_dict = {} 

    async for record in collection.find():
        total_patients += 1
        
        # 1. Lấy thông tin chẩn đoán và ngày tháng
        diagnosis = record.get("ai_diagnosis", "").lower()
        created_at = record.get("created_at", "") # Ví dụ: "2025-11-24 10:30:00"
        
        # 2. Phân loại bệnh (Dựa vào chuỗi kết quả AI trả về)
        is_diabetes = "tiểu đường" in diagnosis or "tieu duong" in diagnosis
        is_hypertension = "huyết áp" in diagnosis or "huyet ap" in diagnosis
        
        # Logic đếm: Nếu chuỗi kết quả có chữ "không bị" thì không đếm là bệnh
        if "không bị tiểu đường" not in diagnosis and "tiền tiểu đường" not in diagnosis:
            if is_diabetes:
                diabetes_count += 1
        if "không bị tăng huyết áp" not in diagnosis and "tiền tăng huyết áp" not in diagnosis:
            if is_hypertension:
                hypertension_count += 1

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
                "total": 0
            }
        
        # Cộng dồn số liệu vào ngày tương ứng
        chart_data_dict[date_str]["total"] += 1
        # if "không bị" not in diagnosis and "khong bi" not in diagnosis and "tiền tăng huyết áp" not in diagnosis and "tien huyet ap" not in diagnosis:
        #     if is_diabetes:
        #         chart_data_dict[date_str]["diabetes"] += 1
        #     if is_hypertension:
        #         chart_data_dict[date_str]["hypertension"] += 1

        if "không bị tiểu đường" not in diagnosis and "tiền tiểu đường" not in diagnosis:
            if is_diabetes:
                chart_data_dict[date_str]["diabetes"] += 1
        if "không bị tăng huyết áp" not in diagnosis and "tiền tăng huyết áp" not in diagnosis:
            if is_hypertension:
                chart_data_dict[date_str]["hypertension"] += 1


    # 4. Chuyển dictionary thành list và sắp xếp theo ngày tăng dần
    chart_list = sorted(list(chart_data_dict.values()), key=lambda x: x['name'])

    return {
        "status": "success",
        "summary": {
            "total": total_patients,
            "diabetes": diabetes_count,
            "hypertension": hypertension_count
        },
        "chart_data": chart_list
    }


my_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=my_api_key)
model = genai.GenerativeModel('gemini-2.5-flash')



# --- API LƯU TRỮ ---
@app.post("/api/glucose/add")
async def add_glucose(record: GlucoseRecord):
    if not record.created_at:
        record.created_at = datetime.now().strftime("%d/%m/%Y %H:%M")
    await collection_glucose.insert_one(record.dict())
    return {"status": "success"}

@app.get("/api/glucose/history")
async def get_glucose_history():
    cursor = collection_glucose.find({}, {"_id": 0}).sort("_id", -1).limit(20)
    history = await cursor.to_list(length=20)
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
                  f"Hãy trả lời ngắn gọn, thân thiện như bác sĩ gia đình. "
                  f"Đưa ra lời khuyên ăn uống cụ thể cho chỉ số đường huyết này.")
        
        response = model.generate_content(prompt)
        return {"reply": response.text}
    except Exception as e:
        print(e)
        return {"reply": "Hệ thống AI đang bận, bạn thử lại sau nhé!"}
    

# --- API DỰ BÁO ĐƯỜNG HUYẾT ---
class PredictionRequest(BaseModel):
    measure_type: str # Chỉ dự báo dựa trên cùng loại (VD: Chỉ dùng lịch sử 'lúc đói' để dự báo 'lúc đói')

# @app.post("/api/predict/glucose")
# async def predict_glucose(req: PredictionRequest):
#     # 1. Lấy dữ liệu từ MongoDB (Lấy hết lịch sử của loại đo đó)
#     cursor = collection_glucose.find({"measure_type": req.measure_type})
#     records = await cursor.to_list(length=100)
    
#     # Nếu dữ liệu quá ít (dưới 3 lần đo) thì không dự báo được
#     if len(records) < 3:
#         return {
#             "can_predict": False, 
#             "message": "Cần ít nhất 3 lần đo trong lịch sử để dự báo!"
#         }

# # 2. Xử lý dữ liệu
#     df = pd.DataFrame(records)
    
#     # Chuyển đổi ngày tháng
#     df['date_obj'] = pd.to_datetime(df['created_at'], dayfirst=True, format='mixed')
    
#     # ⚠️ QUAN TRỌNG: Sắp xếp dữ liệu theo thời gian (Cũ trước -> Mới sau)
#     df = df.sort_values(by='date_obj')
#     print(df)

#     # Chuyển thành Timestamp
#     df['timestamp'] = df['date_obj'].map(pd.Timestamp.timestamp)
    
#     # --- BƯỚC CHUẨN HÓA DỮ LIỆU ---
#     # Lấy mốc thời gian đầu tiên làm gốc (Ngày thứ 0)
#     start_time = df['timestamp'].iloc[0] 
    
#     # Tính số ngày trôi qua (Ngày 0, 0.5, 1, 2...)
#     df['days_passed'] = (df['timestamp'] - start_time) / (24 * 3600)
    
#     X = df[['days_passed']].values
#     y = df['value'].values 

#     # 3. Huấn luyện mô hình
#     model = LinearRegression()
#     model.fit(X, y)

#     # future_predictions = []
#     # current_time = datetime.now()
#     # for i in range (1, 8):
#     # 4. Dự báo cho ngày mai
#         # next_date = current_time + timedelta(days=i)
#     tomorrow_timestamp = datetime.now().timestamp() + (24 * 60 * 60)
    
#     # Chuẩn hóa ngày mai theo mốc bắt đầu luôn
#     tomorrow_days_passed = (tomorrow_timestamp - start_time) / (24 * 3600)
    
#     predicted_value = model.predict([[tomorrow_days_passed]])
#     result = int(predicted_value[0])

#     # --- 5. LOGIC CHẶN SỐ ÂM (QUAN TRỌNG) ---
#     # Nếu kết quả ra Âm hoặc quá thấp (< 50), chứng tỏ xu hướng giảm quá mạnh
#     # Ta sẽ lấy trung bình của 3 lần đo gần nhất thay thế, hoặc gán bằng 70 (mức tối thiểu an toàn)
#     if result < 50:
#         # Cách xử lý thông minh: Nếu dự báo âm, ta giả định đường huyết sẽ giữ ổn định như lần đo cuối
#         result = int(df['value'].iloc[-1]) 
    
#     # Nếu cao quá mức sống (trên 600) thì chặn lại
#     if result > 600:
#         result = 600
#     print(result)
#         # Làm tròn kết quả
#         # result = int(predicted_value[0])
#         # future_predictions.append({
#         #     "date": next_date.strftime("%d/%m"), # Chỉ lấy Ngày/Tháng (VD: 05/12)
#         #     "value": result
#         # })

#     return {
#         "can_predict": True,
#         "predictions": result,
#         "message": f"Dựa trên xu hướng cũ, dự báo đường huyết ngày mai khoảng {result} mg/dL"
#     }

# @app.post("/api/predict/glucose")
# async def predict_glucose(req: PredictionRequest):
#     # 1. Lấy dữ liệu
#     cursor = collection_glucose.find({"measure_type": req.measure_type})
#     records = await cursor.to_list(length=100)
    
#     if len(records) < 3:
#         return {
#             "can_predict": False, 
#             "message": "Cần ít nhất 3 lần đo trong lịch sử để dự báo!"
#         }

#     # 2. Xử lý dữ liệu với Pandas
#     df = pd.DataFrame(records)
#     # Xử lý format ngày tháng linh hoạt (mixed)
#     df['date_obj'] = pd.to_datetime(df['created_at'], dayfirst=False, format='mixed')
#     df = df.sort_values(by='date_obj')

#     # Chuẩn hóa thời gian (Time scaling)
#     df['timestamp'] = df['date_obj'].map(pd.Timestamp.timestamp)
#     start_time = df['timestamp'].iloc[0] 
#     df['days_passed'] = (df['timestamp'] - start_time) / (24 * 3600)
    
#     X = df[['days_passed']].values
#     y = df['value'].values 

#     # 3. Huấn luyện Linear Regression
#     model = LinearRegression()
#     model.fit(X, y)

#     # 4. Vòng lặp dự báo 7 ngày tới
#     future_predictions = []
#     current_time = datetime.now()
#     last_real_val = df['value'].iloc[-1] # Lấy giá trị thật cuối cùng để tham chiếu

#     for i in range(1, 8): # Chạy từ 1 đến 7
#         next_date = current_time + timedelta(days=i)
#         next_days_passed = (next_date.timestamp() - start_time) / (24 * 3600)
        
#         pred = model.predict([[next_days_passed]])
#         val = int(pred[0])

#         # --- Logic chặn số âm / số ảo ---
#         if val < 50:
#             # Nếu dự báo tụt quá sâu, giữ bằng giá trị cuối cùng hoặc mức tối thiểu 70
#             val = max(int(last_real_val), 70) 
#         if val > 600:
#             val = 600
            
#         # future_predictions.append({
#         #     "date": next_date.strftime("%d/%m"), # Chỉ lấy Ngày/Tháng (VD: 05/12)
#         #     "value": val
#         # })

#     return {
#         "can_predict": True,
#         "predictions": future_predictions,
#         "message": f"Dự báo xu hướng cho 7 ngày tới (từ {future_predictions[0]['date']} đến {future_predictions[-1]['date']})"
#     }

# @app.post("/api/predict/glucose")

# async def predict_glucose(req: PredictionRequest):

#     # 1. Lấy dữ liệu từ MongoDB (Lấy hết lịch sử của loại đo đó)

#     cursor = collection_glucose.find({"measure_type": req.measure_type})

#     records = await cursor.to_list(length=100)

   

#     # Nếu dữ liệu quá ít (dưới 3 lần đo) thì không dự báo được

#     if len(records) < 3:

#         return {

#             "can_predict": False,

#             "message": "Cần ít nhất 3 lần đo trong lịch sử để dự báo!"

#         }



# # 2. Xử lý dữ liệu

#     df = pd.DataFrame(records)

   

#     # Chuyển đổi ngày tháng

#     df['date_obj'] = pd.to_datetime(df['created_at'], dayfirst=True, format='mixed')

   

#     # ⚠️ QUAN TRỌNG: Sắp xếp dữ liệu theo thời gian (Cũ trước -> Mới sau)

#     df = df.sort_values(by='date_obj')



#     # Chuyển thành Timestamp

#     df['timestamp'] = df['date_obj'].map(pd.Timestamp.timestamp)

   

#     # --- BƯỚC CHUẨN HÓA DỮ LIỆU ---

#     # Lấy mốc thời gian đầu tiên làm gốc (Ngày thứ 0)

#     start_time = df['timestamp'].iloc[0]

   

#     # Tính số ngày trôi qua (Ngày 0, 0.5, 1, 2...)

#     df['days_passed'] = (df['timestamp'] - start_time) / (24 * 3600)

   

#     X = df[['days_passed']].values

#     y = df['value'].values



#     # 3. Huấn luyện mô hình

#     model = LinearRegression()

#     model.fit(X, y)



#     # 4. Dự báo cho ngày mai

#     tomorrow_timestamp = datetime.now().timestamp() + (24 * 60 * 60)

   

#     # Chuẩn hóa ngày mai theo mốc bắt đầu luôn

#     tomorrow_days_passed = (tomorrow_timestamp - start_time) / (24 * 3600)

   

#     predicted_value = model.predict([[tomorrow_days_passed]])

#     result = int(predicted_value[0])



#     # --- 5. LOGIC CHẶN SỐ ÂM (QUAN TRỌNG) ---

#     # Nếu kết quả ra Âm hoặc quá thấp (< 50), chứng tỏ xu hướng giảm quá mạnh

#     # Ta sẽ lấy trung bình của 3 lần đo gần nhất thay thế, hoặc gán bằng 70 (mức tối thiểu an toàn)

#     if result < 50:

#         # Cách xử lý thông minh: Nếu dự báo âm, ta giả định đường huyết sẽ giữ ổn định như lần đo cuối

#         result = int(df['value'].iloc[-1])

   

#     # Nếu cao quá mức sống (trên 600) thì chặn lại

#     if result > 600:

#         result = 180

   

#     # Làm tròn kết quả

#     # result = int(predicted_value[0])



#     return {

#         "can_predict": True,

#         "predicted_value": result,

#         "message": f"Dựa trên xu hướng cũ, dự báo đường huyết ngày mai khoảng {result} mg/dL"}

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ... (Các phần import và setup giữ nguyên)

@app.post("/api/predict/glucose")
async def predict_glucose(req: PredictionRequest):
    # 1. Lấy dữ liệu (Giữ nguyên code cũ của bé)
    cursor = collection_glucose.find({"measure_type": req.measure_type})
    records = await cursor.to_list(length=100)
    
    if len(records) < 3:
        return {
            "can_predict": False, 
            "message": "Cần ít nhất 3 lần đo trong lịch sử để dự báo!"
        }

    # 2. Xử lý dữ liệu (Giữ nguyên logic chuẩn hóa thời gian của bé)
    df = pd.DataFrame(records)
    df['date_obj'] = pd.to_datetime(df['created_at'], dayfirst=True, format='mixed')
    df = df.sort_values(by='date_obj')

    # Mốc thời gian bắt đầu
    start_time = df['date_obj'].iloc[0].timestamp()
    
    # Tính X (đầu vào) và y (kết quả)
    df['timestamp'] = df['date_obj'].map(pd.Timestamp.timestamp)
    df['days_passed'] = (df['timestamp'] - start_time) / (24 * 3600)
    
    X = df[['days_passed']].values
    y = df['value'].values 

    # 3. Huấn luyện mô hình
    model = LinearRegression()
    model.fit(X, y)

    # --- 4. DỰ BÁO 7 NGÀY (PHẦN MỚI SỬA) ---
    predictions = []
    current_date = datetime.now()
    last_real_value = df['value'].iloc[-1] # Lấy giá trị thật cuối cùng để tham chiếu

    for i in range(1, 8): # Chạy từ ngày mai (1) đến 7 ngày sau (8)
        future_date = current_date + timedelta(days=i)
        future_ts = future_date.timestamp()
        
        # Chuẩn hóa thời gian tương lai theo mốc start_time cũ
        future_days_passed = (future_ts - start_time) / (24 * 3600)
        
        # Dự đoán
        pred_val = model.predict([[future_days_passed]])[0]
        result = int(pred_val)

        # --- LOGIC CHẶN SỐ (Logic cũ của bé nhưng áp dụng trong vòng lặp) ---
        if result < 50:
            # Nếu giảm quá sâu, giả định nó đi ngang bằng giá trị cuối cùng
            result = int(last_real_value) 
        elif result > 600:
            result = 600
        
        predictions.append({
            "date": future_date.strftime("%d/%m"), # Format ngày tháng cho đẹp (VD: 05/12)
            "value": result
        })
        
        # Cập nhật giá trị tham chiếu cho vòng lặp sau (để đường dây mượt hơn nếu cần)
        # last_real_value = result 

    return {
        "can_predict": True,
        "predictions": predictions, # Trả về cả danh sách 7 ngày
        "message": f"Đã dự báo xu hướng cho 7 ngày tới."
    }
if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
