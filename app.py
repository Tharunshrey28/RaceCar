from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import requests
from io import BytesIO

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
model.load_state_dict(torch.load("nvidia_model.pth", map_location=torch.device('cpu')))
model.eval()

# Define Flask app
app = Flask(__name__)

# Define preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((66, 200)),  # Ensure size matches model input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Route for root URL
@app.route("/", methods=["GET"])
def home():
    return "Tharunshrey Gurrampati, EAI6010"
# Route for prediction
@app.route("/predict", methods=["GET", "POST"])
def predict():
    try:
        if request.method == "POST":
            # Handle image upload via POST
            file = request.files['image']
            image = Image.open(io.BytesIO(file.read()))
        elif request.method == "GET":
            # Handle image URL via GET
            image_url = request.args.get('image_url')
            if not image_url:
                return jsonify({"error": "Please provide an image URL using the 'image_url' query parameter."})
            response = requests.get(image_url)
            if response.status_code != 200:
                return jsonify({"error": "Unable to fetch image from the provided URL."})
            image = Image.open(BytesIO(response.content))

        # Ensure image has 3 channels (RGB)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Preprocess the image
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Validate tensor shape
        if image.shape != (1, 3, 66, 200):
            return jsonify({"error": f"Invalid image dimensions {image.shape}. Expected (1, 3, 66, 200)."})

        # Make prediction
        with torch.no_grad():
            prediction = model(image)
            steering_angle = prediction.item()

        return jsonify({"steering_angle": steering_angle})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Use dynamic port for deployment
    app.run(host="0.0.0.0", port=port)

