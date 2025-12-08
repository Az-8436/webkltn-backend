import google.generativeai as genai
from PIL import Image
import io
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Cấu hình API key của Gemini

my_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=my_api_key)

prompt = """
Bạn là một chuyên gia xử lý dữ liệu y tế. Nhiệm vụ của bạn là phân tích hình ảnh phiếu kết quả xét nghiệm được cung cấp và trích xuất các thông tin cụ thể.

Xuất kết quả dưới dạng một đối tượng JSON THUẦN (không markdown).

Với phần "blood_tests", mỗi chỉ số hãy trích xuất thành một object gồm 2 trường:
- "value": Giá trị đo được (dạng số hoặc chuỗi số).
- "unit": Đơn vị đo (ví dụ: mmol/L, mg/dL, %, U/L...). Nếu không thấy đơn vị, hãy để null.

JSON mẫu:
{
  "patient_info": {
    "name": null,
    "gender": null,
    "birth_date": null,
    "age": null,
    "height": null,
    "weight": null,
    "systolicBloodPressure": null,
    "diastolicBloodPressure": null,
    "heartRate": null,
    "bmi": null
  },
  "blood_tests": {
    "cholesterol": { "value": null, "unit": null },
    "hdl": { "value": null, "unit": null },
    "ldl": { "value": null, "unit": null },
    "triglycerid": { "value": null, "unit": null },
    "creatinin": { "value": null, "unit": null },
    "hba1c": { "value": null, "unit": null },
    "ure": { "value": null, "unit": null },
    "vldl": { "value": null, "unit": null }
  }
}
"""

def extract_info_from_image(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))

        # convert image to bytes
        img_buffer = io.BytesIO()
        img.save(img_buffer, format=img.format if img.format else "PNG")
        img_data = img_buffer.getvalue()

        image_part = {
            "mime_type": "image/jpeg" if img.format == "JPEG" else "image/png",
            "data": img_data
        }

        model = genai.GenerativeModel("gemini-2.5-flash") # Nên dùng bản mới nhất nếu có thể
        response = model.generate_content([prompt, image_part])
        text = response.text.strip()

        # loại bỏ markdown nếu có
        text = text.replace("```json", "").replace("```", "").strip()

        data = json.loads(text)

        # xử lý dữ liệu
        info = data.get("patient_info", {})
        tests = data.get("blood_tests", {})

        # # Tính tuổi (age)
        if info.get("age") is None and info.get("birth_date"):
            try:
                year_str = str(info["birth_date"])[-4:] # Lấy 4 số cuối đề phòng định dạng khác
                year = int(year_str)
                info["age"] = 2025 - year
            except:
                info["age"] = None

        # --- 2. CẬP NHẬT LOGIC TÍNH TOÁN VLDL & URE ---
        # Vì cấu trúc giờ là dict {value, unit} nên phải truy cập vào ["value"]

        # Xử lý VLDL (Tính từ Triglycerid nếu có)
        trig = tests.get("triglycerid")
        
        # Nếu Triglycerid có dữ liệu
        if trig and trig.get("value") is not None:
            try:
                # Tính toán
                val_vldl = round(float(trig["value"]) / 2.2, 2)
                # Gán vào VLDL (giữ nguyên unit của Triglycerid hoặc mặc định mmol/L)
                tests["vldl"] = { 
                    "value": val_vldl, 
                    "unit": trig.get("unit", "mmol/L") 
                }
            except:
                tests["vldl"] = { "value": None, "unit": None }
        else:
            tests["vldl"] = { "value": None, "unit": None }

        # Xử lý Ure (Điền mặc định nếu thiếu)
        ure = tests.get("ure")
        if ure is None or ure.get("value") is None:
            # Gán mặc định là object
            tests["ure"] = { "value": 5.0, "unit": "mmol/L" }

        # Xử lý Creatinin (Điền mặc định 5.5 theo yêu cầu của bé)
        creatinin = tests.get("creatinin")
        if creatinin is None or creatinin.get("value") is None:
             tests["creatinin"] = { "value": 5.5, "unit": "umol/L" } # Bé thay đơn vị nếu muốn

        return data

    except Exception as e:
        print("OCR Error:", e)
        return {"error": str(e)} # Trả về lỗi chi tiết để dễ debug
