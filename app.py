"""
Web Application Backend for Deep JSCC Image Semantic Communication.

Provides REST API endpoints using Python's built-in http.server:
- GET  /                     -> Serves index.html
- GET  /api/sample           -> Returns a sample CIFAR-10 image as base64
- POST /api/encode           -> Encodes image to semantic latent vector
- POST /api/channel          -> Simulates AWGN wireless channel noise on vector
- POST /api/decode           -> Decodes semantic latent vector into image
"""

import base64
import io
import json
import math
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import config
from model import DeepJSCC


# Load model globally using config
DEVICE = config.get_device()
MODEL = DeepJSCC(
    in_channels=config.IN_CHANNELS,
    channel_c=config.CHANNEL_C,
    power=config.POWER_CONSTRAINT,
).to(DEVICE)
CHECKPOINT_PATH = config.BEST_MODEL_PATH

if os.path.exists(CHECKPOINT_PATH):
    print(f"Loading trained checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    MODEL.load_state_dict(ckpt["model_state_dict"])
    print(f"Model loaded (trained at SNR={ckpt.get('snr_db', config.DEFAULT_SNR_DB)} dB)")
else:
    print("[INFO] Checkpoint not found. Running with initialized model.")

MODEL.eval()

# Image transforms
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize(config.IMG_SIZE),
    transforms.ToTensor(),
])


def tensor_to_base64_png(tensor: torch.Tensor) -> str:
    """Converts a (3, H, W) float tensor [0, 1] to base64 PNG data URL."""
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    arr = (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    img = Image.fromarray(arr)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def base64_to_tensor(b64_str: str) -> torch.Tensor:
    """Converts a base64 image data URL to a (1, 3, 32, 32) float tensor [0, 1]."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    image_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = IMG_TRANSFORM(img).unsqueeze(0)  # (1, 3, 32, 32)
    return tensor.to(DEVICE)


def generate_feature_map_images(z_tensor: torch.Tensor) -> List[str]:
    """Generates base64 heatmap images for each of the 16 feature channels (8x8)."""
    feature_maps = []
    # z_tensor shape: (1, 16, 8, 8)
    z_np = z_tensor[0].detach().cpu().numpy()  # (16, 8, 8)
    
    for c in range(z_np.shape[0]):
        channel_data = z_np[c]
        # Normalize channel values to [0, 255] for display
        min_v, max_v = channel_data.min(), channel_data.max()
        if max_v > min_v:
            norm_data = (channel_data - min_v) / (max_v - min_v)
        else:
            norm_data = np.zeros_like(channel_data)
        
        # Upsample 8x8 to 64x64 with nearest neighbor for clear pixel grid visualization
        norm_img = Image.fromarray((norm_data * 255).astype(np.uint8), mode="L")
        norm_img = norm_img.resize((64, 64), resample=Image.Resampling.NEAREST)
        
        buffer = io.BytesIO()
        norm_img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        feature_maps.append(f"data:image/png;base64,{b64}")
        
    return feature_maps


class SemanticCommHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html_path = os.path.join(os.path.dirname(__file__), "UserInterface", "index.html")
            if not os.path.exists(html_path):
                html_path = "index.html"
            with open(html_path, "rb") as f:
                self.wfile.write(f.read())
            return

        elif self.path == "/api/sample":
            self.handle_sample()
            return

        super().do_GET()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        data = json.loads(body.decode("utf-8")) if body else {}

        if self.path == "/api/encode":
            self.handle_encode(data)
        elif self.path == "/api/channel":
            self.handle_channel(data)
        elif self.path == "/api/decode":
            self.handle_decode(data)
        else:
            self.send_error(404, "Endpoint not found")

    def handle_sample(self):
        """Returns a sample test image from CIFAR-10 data folder if available, or synthetic."""
        cifar_dir = "./data/cifar-10-batches-py"
        sample_tensor = None

        if os.path.exists(cifar_dir):
            try:
                import pickle
                with open(os.path.join(cifar_dir, "test_batch"), "rb") as f:
                    batch = pickle.load(f, encoding="bytes")
                # Random index
                idx = np.random.randint(0, 1000)
                raw = batch[b"data"][idx]
                label_name = batch[b"filenames"][idx].decode("utf-8")
                # Reshape (3, 32, 32)
                raw = raw.reshape(3, 32, 32)
                sample_tensor = torch.tensor(raw, dtype=torch.float32) / 255.0
            except Exception as e:
                print(f"Error loading sample: {e}")

        if sample_tensor is None:
            # Fallback: create synthetic colorful test pattern
            sample_tensor = torch.rand(3, 32, 32)

        data_url = tensor_to_base64_png(sample_tensor)
        self.send_json_response({"image": data_url})

    def handle_encode(self, data: Dict[str, Any]):
        """Encodes an uploaded image into the semantic latent vector."""
        try:
            image_b64 = data.get("image", "")
            if not image_b64:
                self.send_error(400, "Missing 'image' parameter")
                return

            x = base64_to_tensor(image_b64)
            with torch.no_grad():
                z = MODEL.encoder(x)  # shape (1, 16, 8, 8)

            z_list = z.squeeze(0).cpu().numpy().tolist()  # shape [16, 8, 8]
            z_flat = [float(val) for val in z.view(-1).cpu().numpy()]
            
            # Channel symbol count & average power verification
            k = len(z_flat)
            avg_power = float(np.mean(np.array(z_flat) ** 2))
            
            # Generate previews for each of the 16 feature maps
            feature_map_urls = generate_feature_map_images(z)

            res = {
                "shape": list(z.shape),
                "num_symbols": k,
                "avg_power": round(avg_power, 4),
                "min_val": round(float(min(z_flat)), 4),
                "max_val": round(float(max(z_flat)), 4),
                "vector": z_list,
                "vector_flat": z_flat,
                "feature_maps": feature_map_urls,
            }
            self.send_json_response(res)
        except Exception as e:
            self.send_json_response({"error": str(e)}, status=500)

    def handle_channel(self, data: Dict[str, Any]):
        """Applies AWGN channel noise to a semantic vector at given SNR."""
        try:
            vector_data = data.get("vector")
            snr_db = float(data.get("snr_db", 10.0))
            is_noiseless = bool(data.get("noiseless", False))

            z_tensor = torch.tensor(vector_data, dtype=torch.float32, device=DEVICE)
            if z_tensor.dim() == 3:
                z_tensor = z_tensor.unsqueeze(0)  # (1, 16, 8, 8)

            if is_noiseless:
                z_noisy = z_tensor
            else:
                with torch.no_grad():
                    z_noisy = MODEL.channel(z_tensor, snr_db=snr_db)

            z_noisy_list = z_noisy.squeeze(0).cpu().numpy().tolist()
            z_noisy_flat = [float(val) for val in z_noisy.view(-1).cpu().numpy()]

            res = {
                "noisy_vector": z_noisy_list,
                "noisy_vector_flat": z_noisy_flat,
                "snr_db": snr_db,
                "is_noiseless": is_noiseless,
            }
            self.send_json_response(res)
        except Exception as e:
            self.send_json_response({"error": str(e)}, status=500)

    def handle_decode(self, data: Dict[str, Any]):
        """Decodes a semantic latent vector into a reconstructed image."""
        try:
            vector_data = data.get("vector")
            orig_image_b64 = data.get("original_image", None)

            if vector_data is None:
                self.send_error(400, "Missing 'vector' data")
                return

            z_tensor = torch.tensor(vector_data, dtype=torch.float32, device=DEVICE)
            # Ensure shape is (1, c, 8, 8)
            if z_tensor.dim() == 1:
                # If flattened array of 1024 floats
                z_tensor = z_tensor.view(1, 16, 8, 8)
            elif z_tensor.dim() == 3:
                z_tensor = z_tensor.unsqueeze(0)

            with torch.no_grad():
                x_hat = MODEL.decoder(z_tensor)  # (1, 3, 32, 32)

            recon_b64 = tensor_to_base64_png(x_hat.squeeze(0))

            metrics = {}
            if orig_image_b64:
                try:
                    x_orig = base64_to_tensor(orig_image_b64)
                    mse_val = float(nn.functional.mse_loss(x_hat, x_orig).item())
                    psnr_val = 10.0 * math.log10(1.0 / max(mse_val, 1e-10))
                    metrics["mse"] = round(mse_val, 5)
                    metrics["psnr"] = round(psnr_val, 2)
                except Exception as ex:
                    print(f"Metric calculation error: {ex}")

            res = {
                "reconstructed_image": recon_b64,
                "metrics": metrics,
            }
            self.send_json_response(res)
        except Exception as e:
            self.send_json_response({"error": str(e)}, status=500)

    def send_json_response(self, data: Dict[str, Any], status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(host: str = config.WEB_HOST, port: int = config.WEB_PORT):
    server_address = (host, port)
    httpd = HTTPServer(server_address, SemanticCommHandler)
    print("=" * 60)
    print(f"Deep JSCC Semantic Communication Web App running at:")
    print(f"  -> http://localhost:{port}")
    print("=" * 60)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        httpd.server_close()


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else config.WEB_PORT
    run_server(port=port)

