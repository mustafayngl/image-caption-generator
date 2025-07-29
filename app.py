import os
import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from model import generate_captions

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Store temporary results
last_uploaded_image = None
last_captions = []

@app.route("/", methods=["GET", "POST"])
def index():
    global last_uploaded_image, last_captions

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            last_uploaded_image = filename
            last_captions = generate_captions(filepath)
            return redirect(url_for("index"))  # Redirect to GET

    # On GET, show last result if exists
    image_url = f"/static/uploads/{last_uploaded_image}" if last_uploaded_image else None
    return render_template("index.html", captions=last_captions, image_url=image_url)

# API endpoint
@app.route("/api/caption", methods=["POST"])
def api_caption():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    captions = generate_captions(filepath)
    return jsonify({"captions": captions})

if __name__ == "__main__":
    app.run(debug=True)
