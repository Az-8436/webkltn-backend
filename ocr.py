import google.generativeai as genai
from PIL import Image
import io
import json

# Cấu hình API key của Gemini
genai.configure(api_key="AIzaSyBVCXIaCG6lErfoCD9ybXI8fIio-QhBacU")

prompt = """
Bạn là một chuyên gia xử lý dữ liệu y tế. Nhiệm vụ của bạn là phân tích hình ảnh phiếu kết quả xét nghiệm được cung cấp và trích xuất các thông tin cụ thể sau.

Xuất kết quả dưới dạng một đối tượng JSON THUẦN KHÔNG CHỨA BẤT KỲ VĂN BẢN NÀO KHÁC (ví dụ: không có ```json, không giải thích).

Các trường cần trích xuất:
1. name
2. gender  
3. birth_date  
4. height
5. weight
6. systolicBloodPressure
7. diastolicBloodPressure
8. heartRate
9. bmi
10. cholesterol  
11. hdl  
12. ldl  
13. triglycerid  
14. creatinin  
15. hba1c  
16. ure  

JSON mẫu:

{
  "patient_info": {
    "name": null,
    "gender": null,
    "age": null,
    "birth_date": null
    "height": null, 
    "weight": null,
    "systolicBloodPressure": null,
    "diastolicBloodPressure": null,
    "heartRate": null,
    "bmi": null
  },
  "blood_tests": {
    "cholesterol": null,
    "hdl": null,
    "ldl": null,
    "triglycerid": null,
    "creatinin": null,
    "hba1c": null,
    "ure": null,
    "vldl": null
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

        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content([prompt, image_part])
        text = response.text.strip()

        # loại bỏ markdown nếu có
        text = text.replace("```json", "").replace("```", "").strip()

        data = json.loads(text)

        # xử lý dữ liệu
        info = data["patient_info"]
        tests = data["blood_tests"]

        # # chuyển gender
        # if info["gender"] in ["Nu", "Nữ", "Female"]:
        #     tests["gender"] = 0
        # elif info["gender"] in ["Nam", "Male"]:
        #     tests["gender"] = 1
        # else:
        #     tests["gender"] = None

        # # age
        if info["age"] is None:
            try:
                year = int(info["birth_date"][-4:])
                info["age"] = 2025 - year
            except:
                info["age"] = None
        # else:
        #     tests["age"] = None

        # # default bmi nếu thiếu
        # tests["bmi"] = 31

        # VLDL
        if tests["triglycerid"] is not None:
            try:
                tests["vldl"] = round(float(tests["triglycerid"]) / 2.2, 2)
            except:
                tests["vldl"] = None
        else:
            tests["vldl"] = None

        # nếu ure thiếu
        if tests["ure"] is None:
            tests["ure"] = 5.0

        return data 

    except Exception as e:
        print("OCR Error:", e)
        return {"error": "OCR failed"}
