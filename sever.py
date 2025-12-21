
from flask import Flask, request, jsonify
from tf_keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import io

app = Flask(__name__)
app.json.sort_keys = False

print(" Đang tải Model AI...")
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()
print(" AI phân biệt người, chó và mèo đã sẵn sàng tại http://127.0.0.1:5000")

def ai_processing(image_file):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    

    image = Image.open(image_file).convert("RGB")
    

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    
    result = {}
    for i in range(len(class_names)):
        label = class_names[i].strip()[2:] 
        score = float(prediction[0][i])
        result[label] = score
    
    return result

@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return jsonify({"error": "Không tìm thấy file"}), 400
    
    file = request.files['file']
    
    try:
        prediction_result = ai_processing(file)

        return jsonify(prediction_result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)