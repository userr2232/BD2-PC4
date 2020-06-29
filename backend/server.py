from sklearn.decomposition import PCA
from pc4 import knn_rtree
import face_recognition
from flask import Flask, jsonify, request, redirect, render_template

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)

pca = PCA(n_components=32);

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    # Check if a valid image file was uploaded
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        k = request.form['k'];

        file = request.files['file']

        if not file.filename:
            return redirect(request.url)

        if file and allowed_file(file.filename):
            img = face_recognition.load_image_file(file)
            encoding = face_recognition.face_encodings(img);
            if encoding:
                return knn_rtree(encoding, k);
    # If no valid image file was uploaded, show the file upload form:
    return render_template("index.html");

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)