
import gradio as gr
import requests

API_URL = "http://127.0.0.1:5000/predict"

def call_api(image):
    """Hàm này gửi ảnh sang Flask và nhận kết quả"""
    if image is None:
        return None
    

    import io
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
    
    try:
        response = requests.post(API_URL, files=files)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"Lỗi": f"Server trả về lỗi: {response.text}"}
            
    except Exception as e:
        return {"Lỗi": f"Không kết nối được tới Flask: {str(e)}"}

interface = gr.Interface(
    fn=call_api, 
    inputs=gr.Image(type="pil", label="Tải ảnh lên"),
    outputs=gr.Label(num_top_classes=2, label="Kết quả dự đoán"),
    title=" Ứng dụng AI Phân Loại (Chó - Mèo - Người)",
    description="Nhóm 3 Trainee Program"
)

if __name__ == "__main__":
    interface.launch(server_port=7860)