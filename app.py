import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import os

# === Helper to calculate conv2d output size ===
def findConv2dOutShape(Hin, Win, conv_layer):
    kernel_size = conv_layer.kernel_size
    stride = conv_layer.stride
    padding = conv_layer.padding
    dilation = conv_layer.dilation

    Hout = int((Hin + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    Wout = int((Win + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)

    return Hout, Wout

# === CNN_TUMOR model ===
class CNN_TUMOR(nn.Module):
    def __init__(self, params):
        super(CNN_TUMOR, self).__init__()
        Cin, Hin, Win = params["shape_in"]
        init_f = params["initial_filters"]
        num_fc1 = params["num_fc1"]
        num_classes = params["num_classes"]
        self.dropout_rate = params["dropout_rate"]

        # Define conv layers
        self.conv1 = nn.Conv2d(Cin, init_f, kernel_size=3)
        self.conv2 = nn.Conv2d(init_f, 2 * init_f, kernel_size=3)
        self.conv3 = nn.Conv2d(2 * init_f, 4 * init_f, kernel_size=3)
        self.conv4 = nn.Conv2d(4 * init_f, 8 * init_f, kernel_size=3)

        # Dynamically compute flatten size using a dummy input
        with torch.no_grad():
           dummy_input = torch.zeros(1, Cin, Hin, Win)
           x = F.relu(self.conv1(dummy_input))
           x = F.max_pool2d(x, 2, 2)
           x = F.relu(self.conv2(x))
           x = F.max_pool2d(x, 2, 2)
           x = F.relu(self.conv3(x))
           x = F.max_pool2d(x, 2, 2)
           x = F.relu(self.conv4(x))
           x = F.max_pool2d(x, 2, 2)
           self.num_flatten = x.view(1, -1).shape[1]

        print(f"✅ Dynamically computed flatten size: {self.num_flatten}")

        # Fully connected layers
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, self.num_flatten)
        X = F.relu(self.fc1(X))
        X = F.dropout(X, self.dropout_rate)
        X = self.fc2(X)
        return F.log_softmax(X, dim=1)

# === Preprocessing function ===
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    print(f"🔍 Original image size: {image.size}")  # (W, H)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # ✅ Enforce resize
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = transform(image)
    print(f" Preprocessed image shape: {image.shape}")  # Should be [3, 224, 224]
    return image.unsqueeze(0)  # [1, 3, 224, 224]

# === Class label mapping ===
CLASS_LABELS = ['Brain Tumor', 'Healthy']

# === Load model and weights ===
def load_model(weights_path, device):
    params = {
    "shape_in": (3, 256, 256),       # Must match resized image
    "initial_filters": 8,
    "num_fc1": 100,
    "num_classes": 2,
    "dropout_rate": 0.5
    }
    print("Using params:", params)
    model = CNN_TUMOR(params)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# === Inference ===
def run_inference(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        pred_index = output.argmax(dim=1).item()
        probs = torch.exp(output).cpu().squeeze().tolist()
        label = CLASS_LABELS[pred_index]
    return label, probs

# === CLI ===
def main():
    parser = argparse.ArgumentParser(description="Brain Tumor Detection Inference")
    parser.add_argument('--model-path', type=str, required=True, help='Path to weights.pt')
    parser.add_argument('--input', type=str, required=True, help='Path to image or folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_path, device)

    if os.path.isdir(args.input):
        for fname in os.listdir(args.input):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(args.input, fname)
                image_tensor = preprocess_image(img_path)
                label, probs = run_inference(model, image_tensor, device)
                print(f"{fname} -> Predicted: {label}, Probabilities: {probs}")
    else:
        image_tensor = preprocess_image(args.input)
        label, probs = run_inference(model, image_tensor, device)
        print(f"{os.path.basename(args.input)} -> Predicted: {label}, Probabilities: {probs}")

if __name__ == "__main__":
    main()

