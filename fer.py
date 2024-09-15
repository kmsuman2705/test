from flask import Flask, render_template, request, jsonify
import face_recognition
import cv2
import numpy as np
import os
from fer import FER

app = Flask(__name__)

# Directory configuration for face recognition
known_faces_dir = os.path.join('static', 'img', 'known')
unknown_faces_dir = os.path.join('static', 'img', 'unknown')

# Load known faces
known_faces = {}
for filename in os.listdir(known_faces_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_faces[filename] = encoding

# Initialize emotion detector
emotion_detector = FER()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame found in request'}), 400

    # Process the uploaded frame
    frame = request.files['frame'].read()
    np_array = np.frombuffer(frame, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Convert image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    # Detect emotions
    emotion_results = emotion_detector.detect_emotions(rgb_image)
    print(f"Emotion results: {emotion_results}")  # Debugging print statement

    results = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding)
        name = "Unknown"
        if True in matches:
            matched_idx = matches.index(True)
            name = list(known_faces.keys())[matched_idx]

        # Initialize emotion to "Unknown"
        emotion = "Unknown"

        # Iterate through detected emotions and their bounding boxes
        for result in emotion_results:
            box = result.get('box')
            emotions = result.get('emotions')

            if isinstance(box, (np.ndarray, list)) and len(box) == 4:
                rect_x, rect_y, rect_width, rect_height = box
                rect_right = rect_x + rect_width
                rect_bottom = rect_y + rect_height

                # Check if the face bounding box intersects with the emotion box
                if (left <= rect_right <= right) and (top <= rect_bottom <= bottom) and \
                   (left <= rect_x <= right) and (top <= rect_y <= bottom):
                    # Find the emotion with the highest score
                    emotion = max(emotions, key=emotions.get)
                    break
            else:
                print(f"Unexpected box format: {box}")

        results.append({
            'name': name,
            'emotion': emotion,
            'left': left,
            'top': top,
            'right': right,
            'bottom': bottom
        })

    print(f"Detected faces and emotions: {results}")  # Debugging print statement

    return jsonify({'faces': results})

if __name__ == '__main__':
    # Use SSL context for secure connections
    app.run(host='0.0.0.0', port=5000, debug=True, ssl_context=('cert.pem', 'key.pem'))
