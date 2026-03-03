from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static"

classes = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)

# ----- Same model architecture -----
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = Net()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    image = Image.open(filepath).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        prediction = classes[predicted.item()]
        confidence = round(confidence.item() * 100, 2)

    return render_template(
        "result.html",
        prediction=prediction,
        confidence=confidence,
        image_path=filepath
    )

if __name__ == "__main__":
    app.run()
