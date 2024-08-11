from flask import Flask,session, render_template, Response, redirect, url_for,jsonify,request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

monitoring=False
app = Flask(__name__)

model = load_model('face_detection_new.h5')

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    global monitoring
    global cap
    cap = cv2.VideoCapture(0)
    while True:
        ret, image = cap.read()
        if not ret:
            break
        if monitoring:
            image = image[50:500, 50:500, :]
    
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize the image
            resized = tf.image.resize(rgb, (120, 120))
            
            # Normalize the resized image
            resized = resized / 255.0
            
            # Predict face coordinates
            yhat = model.predict(np.expand_dims(resized, 0))
            
            if yhat is None:
                print("Model prediction returned None")
                continue
            
            if len(yhat) < 2 or yhat[1] is None:
                print("Invalid model prediction output")
                continue
            
            sample_coords = yhat[1][0]
            
            # If a face is detected with confidence > 0.5
            if yhat[0][0] > 0.5: 
                # Draw the main rectangle
                cv2.rectangle(image, 
                            tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
                            tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)), 
                            (0, 0, 0), 2)
                
                # Draw the label rectangle
                cv2.rectangle(image, 
                            tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int), [0, -30])),
                            tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int), [80, 0])), 
                            (0, 0, 255), -1)
                
                # Render the text label
                cv2.putText(image, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int), [0, -5])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        ret, buffer = cv2.imencode('.jpg', image)
        image = buffer.tobytes()
        yield (b'--image\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=image')

@app.route('/start_monitoring')
def start_monitoring():
    global monitoring
    monitoring = True
    return 'Monitoring started'

@app.route('/stop_monitoring')
def stop_monitoring():
    global monitoring
    monitoring=False
    return 'Monitoring stopped'


















@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        image = Image.open(file.stream)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces in the image
        face_coordinates = detect_faces(image)
        
        if face_coordinates:
            # Unpack coordinates
            x1, y1, x2, y2 = face_coordinates
            # Draw rectangles around the detected face
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Convert image back to RGB (for displaying with PIL)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image)

        # Convert the PIL image to a string of bytes and then to a base64 encoded string
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG')
        byte_im = buf.getvalue()
        encoded_img = base64.b64encode(byte_im).decode('utf-8')

        return jsonify({'faces': len(face_coordinates) // 4, 'image': encoded_img})



if __name__ == '__main__':
    app.run(debug=True)
