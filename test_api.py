import google.generativeai as genai
import os

# 1. Điền API Key của bạn vào đây
os.environ["GOOGLE_API_KEY"] = "AIzaSyDw22_jV4XJ6UZ0DWE2fnGxOT9uGrtRcEM"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("--- Đang kiểm tra kết nối ---")
try:
    # 2. Lấy danh sách mô hình
    print("Các mô hình hỗ trợ 'generateContent':")
    found = False
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
            found = True
    
    if not found:
        print("KHÔNG tìm thấy mô hình nào. Có thể do lỗi API Key hoặc Vùng (Region).")

except Exception as e:
    print(f"LỖI NGHIÊM TRỌNG: {e}")