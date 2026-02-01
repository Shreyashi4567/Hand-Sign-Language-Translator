import os
import pickle
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
from torchvision.models import squeezenet1_1

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load Orange model
with open("neural network2.pkcls", "rb") as f:
    model = pickle.load(f)

print("✅ Model loaded")
print(f"Model expects: {len(model.domain.attributes)} features")

# Load SqueezeNet (DO NOT remove classifier)
device = torch.device("cpu")
squeezenet = squeezenet1_1(pretrained=True).to(device)
squeezenet.eval()

# Orange-compatible preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_image_embeddings(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            outputs = squeezenet(img_tensor)

        # SqueezeNet outputs 1000 features (same as Orange)
        return outputs.numpy().flatten()

    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probabilities = None
    filename = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename:
            filename = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)

            try:
                embeddings = get_image_embeddings(path)

                if embeddings is None:
                    prediction = "Error processing image"
                else:
                    embeddings = embeddings.reshape(1, -1)

                    # Orange prediction
                    result = model(embeddings)

                    class_values = list(model.domain.class_var.values)
                    pred_idx = int(result[0])
                    prediction = class_values[pred_idx]

                    # Probabilities
                    if hasattr(result, "probabilities") and len(result.probabilities) > 0:
                        probs = result.probabilities[0]
                        probabilities = {
                            sign: round(prob * 100, 2)
                            for sign, prob in zip(class_values, probs)
                        }

                    print("✅ Prediction:", prediction)

            except Exception as e:
                print("❌ Error:", e)
                import traceback
                traceback.print_exc()
                prediction = "Internal error"

    return render_template(
        "index.html",
        prediction=prediction,
        probabilities=probabilities,
        filename=filename
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
