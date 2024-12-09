
from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

# Define the model structure (same as training)
class NvidiaModel(torch.nn.Module):
    def __init__(self):
        super(NvidiaModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = torch.nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = torch.nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = torch.nn.Conv2d(48, 64, kernel_size=3)
        self.conv5 = torch.nn.Conv2d(64, 64, kernel_size=3)
        self.fc1 = torch.nn.Linear(64 * 1 * 18, 100)
        self.fc2 = torch.nn.Linear(100, 50)
        self.fc3 = torch.nn.Linear(50, 10)
        self.fc4 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Load the trained model
model = NvidiaModel()
model.load_state_dict(torch.load("nvidia_model.pth"))
model.eval()

# Define Flask app
app = Flask(__name__)

# Define preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the image from the request
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))

        # Preprocess the image
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            prediction = model(image)
            steering_angle = prediction.item()

        return jsonify({"steering_angle": steering_angle})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
