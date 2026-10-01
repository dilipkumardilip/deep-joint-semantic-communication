"""
Web Application Backend for Deep JSCC Image Semantic Communication.

Provides REST API endpoints using Python's built-in http.server:
- GET  /                     -> Serves index.html
- GET  /api/models           -> Returns list of all available trained models
- POST /api/select_model     -> Switches the active model
- GET  /api/sample           -> Returns a sample image (from CIFAR-10 or synthetic)
- POST /api/encode           -> Encodes image to semantic latent vector
- POST /api/channel          -> Simulates AWGN wireless channel noise
- POST /api/decode           -> Decodes semantic latent vector into image
"""

import base64
import io
import json
import math
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import config
from model import DeepJSCC


# ---------------------------------------------------------------------------
# Model Registry — each entry describes a trained checkpoint
# Format:
#   key         : unique identifier used in the API (e.g. "cifar10_c16")
#   label       : friendly display name shown in the UI
#   description : short description for the UI card
#   checkpoint  : path to .pth checkpoint file
#   patch_size  : input image size the model expects
#   channel_c   : latent channel feature count
#   dataset     : training dataset name
#   badge       : short tag shown in UI badge (e.g. "32×32", "128×128 HD")
# ---------------------------------------------------------------------------
MODEL_REGISTRY: List[Dict[str, Any]] = [
    {
        "key":         "cifar10_c16",
        "label":       "CIFAR-10 Baseline",
        "description": "Trained on 32×32 images from CIFAR-10. Fast baseline model following Bourtsoulatze et al.",
        "checkpoint":  "./checkpoints/best_jscc_model.pth",
        "patch_size":  32,
        "channel_c":   config.CHANNEL_C,
        "dataset":     "CIFAR-10",
        "badge":       "32×32",
        "snr_db":      10.0,
    },
    {
        "key":         "div2k_c16",
        "label":       "DIV2K HD Model",
        "description": "Trained on 128×128 HD patches from DIV2K dataset. Handles high-resolution images with richer semantic features.",
        "checkpoint":  "./checkpoints/best_jscc_model_div2k.pth",
        "patch_size":  128,
        "channel_c":   config.CHANNEL_C,
        "dataset":     "DIV2K",
        "badge":       "128×128 HD",
        "snr_db":      10.0,
    },
]

DEVICE = config.get_device()

# ---------------------------------------------------------------------------
# Load all available models into memory at startup
# ---------------------------------------------------------------------------
def _load_model(entry: Dict[str, Any]) -> Optional[DeepJSCC]:
    """Loads a DeepJSCC model from a registry entry checkpoint. Returns None if not found."""
    ckpt_path = entry["checkpoint"]
    model = DeepJSCC(
        in_channels=config.IN_CHANNELS,
        channel_c=entry["channel_c"],
        power=config.POWER_CONSTRAINT,
        snr_db=entry.get("snr_db", config.DEFAULT_SNR_DB),
    ).to(DEVICE)

    if os.path.exists(ckpt_path):
        print(f"  Loading '{entry['label']}' from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model
    else:
        print(f"  [INFO] Checkpoint not found for '{entry['label']}': {ckpt_path} — will use random weights.")
        model.eval()
        return model


print("=" * 60)
print("Loading model registry...")
LOADED_MODELS: Dict[str, DeepJSCC] = {}
for _entry in MODEL_REGISTRY:
    LOADED_MODELS[_entry["key"]] = _load_model(_entry)  # type: ignore[assignment]
print(f"Registry ready: {list(LOADED_MODELS.keys())}")
print("=" * 60)

# Active model key (default to first entry)
_active_key: str = MODEL_REGISTRY[0]["key"]


def get_active_entry() -> Dict[str, Any]:
    return next(e for e in MODEL_REGISTRY if e["key"] == _active_key)


def get_active_model() -> DeepJSCC:
    return LOADED_MODELS[_active_key]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def make_transform(patch_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
    ])


def tensor_to_base64_png(tensor: torch.Tensor) -> str:
    """Converts a (3, H, W) float tensor [0,1] to a base64 PNG data URL."""
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    arr = (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_tensor(b64_str: str, patch_size: int) -> torch.Tensor:
    """Converts a base64 image data URL to a (1, 3, H, W) float tensor in [0,1]."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    image_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = make_transform(patch_size)(img).unsqueeze(0)
    return tensor.to(DEVICE)


def generate_feature_map_images(z_tensor: torch.Tensor) -> List[str]:
    """Generates base64 heatmap previews for each feature channel in z_tensor."""
    feature_maps = []
    z_np = z_tensor[0].detach().cpu().numpy()  # (C, H, W)
    for c in range(z_np.shape[0]):
        channel_data = z_np[c]
        min_v, max_v = channel_data.min(), channel_data.max()
        norm = (channel_data - min_v) / (max_v - min_v + 1e-8)
        norm_img = Image.fromarray((norm * 255).astype(np.uint8), mode="L")
        norm_img = norm_img.resize((64, 64), resample=Image.Resampling.NEAREST)
        buf = io.BytesIO()
        norm_img.save(buf, format="PNG")
        feature_maps.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8"))
    return feature_maps


def model_registry_json() -> List[Dict[str, Any]]:
    """Returns the model registry as a JSON-serializable list, with availability info."""
    out = []
    for entry in MODEL_REGISTRY:
        spatial = entry["patch_size"] // 4
        out.append({
            "key":          entry["key"],
            "label":        entry["label"],
            "description":  entry["description"],
            "dataset":      entry["dataset"],
            "badge":        entry["badge"],
            "patch_size":   entry["patch_size"],
            "channel_c":    entry["channel_c"],
            "snr_db":       entry["snr_db"],
            "symbols_k":    entry["channel_c"] * spatial * spatial,
            "available":    os.path.exists(entry["checkpoint"]),
            "active":       entry["key"] == _active_key,
        })
    return out


# ---------------------------------------------------------------------------
# HTTP Request Handler
# ---------------------------------------------------------------------------

class SemanticCommHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        # Quieter logs — only print non-asset requests
        if not any(self.path.endswith(ext) for ext in [".ico", ".css", ".js", ".png", ".woff2"]):
            super().log_message(format, *args)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.end_headers()

    # MIME type map for static file serving
    _MIME = {
        ".html": "text/html; charset=utf-8",
        ".css":  "text/css; charset=utf-8",
        ".js":   "application/javascript; charset=utf-8",
        ".json": "application/json",
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg":  "image/svg+xml",
        ".ico":  "image/x-icon",
        ".woff2":"font/woff2",
        ".woff": "font/woff",
    }

    def _serve_ui_file(self, rel_path: str) -> bool:
        """Serve a file from the UserInterface/ directory. Returns True if served."""
        ui_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "UserInterface")
        abs_path = os.path.normpath(os.path.join(ui_root, rel_path.lstrip("/")))
        # Security: ensure path stays inside UserInterface/
        if not abs_path.startswith(ui_root):
            return False
        if not os.path.isfile(abs_path):
            return False
        ext  = os.path.splitext(abs_path)[1].lower()
        mime = self._MIME.get(ext, "application/octet-stream")
        with open(abs_path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)
        return True

    def do_GET(self) -> None:
        path = self.path.split("?")[0]  # strip query string

        # --- HTML pages ---
        if path in ("/", "/index.html"):
            self._serve_ui_file("index.html") or self.send_error(404)
            return

        # --- API endpoints ---
        if path == "/api/models":
            self.send_json_response({"models": model_registry_json(), "active": _active_key})
            return
        if path == "/api/sample":
            self.handle_sample()
            return

        # --- Compare iframe view ---
        if path == "/compare":
            self._serve_ui_file("views/compare.html") or self.send_error(404)
            return

        # --- Static assets from UserInterface/ (styles, scripts, views, etc.) ---
        static_prefixes = ("/styles/", "/scripts/", "/views/", "/assets/")
        if any(path.startswith(p) for p in static_prefixes):
            if not self._serve_ui_file(path):
                self.send_error(404, f"Static file not found: {path}")
            return

        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        data: Dict[str, Any] = json.loads(body.decode("utf-8")) if body else {}

        if self.path == "/api/select_model":
            self.handle_select_model(data)
        elif self.path == "/api/encode":
            self.handle_encode(data)
        elif self.path == "/api/channel":
            self.handle_channel(data)
        elif self.path == "/api/decode":
            self.handle_decode(data)
        else:
            self.send_error(404, "Endpoint not found")

    # -----------------------------------------------------------------------
    # Endpoint Handlers
    # -----------------------------------------------------------------------

    def handle_select_model(self, data: Dict[str, Any]) -> None:
        global _active_key
        key = data.get("key", "")
        if key not in LOADED_MODELS:
            self.send_json_response({"error": f"Unknown model key: '{key}'"}, status=400)
            return
        _active_key = key
        entry = get_active_entry()
        print(f"[Model Switch] Active model → '{entry['label']}' (patch={entry['patch_size']}px)")
        self.send_json_response({"success": True, "active": _active_key, "models": model_registry_json()})

    def handle_sample(self) -> None:
        """Returns a sample image: from CIFAR-10 if available, otherwise a synthetic pattern."""
        sample_tensor: Optional[torch.Tensor] = None
        cifar_dir = "./data/cifar-10-batches-py"
        patch_size = get_active_entry()["patch_size"]

        if os.path.exists(cifar_dir):
            try:
                import pickle
                with open(os.path.join(cifar_dir, "test_batch"), "rb") as f:
                    batch = pickle.load(f, encoding="bytes")
                idx = np.random.randint(0, 1000)
                raw = batch[b"data"][idx].reshape(3, 32, 32)
                sample_tensor = torch.tensor(raw, dtype=torch.float32) / 255.0
                if patch_size != 32:
                    img_pil = Image.fromarray(
                        (sample_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    )
                    img_pil = img_pil.resize((patch_size, patch_size), Image.Resampling.BICUBIC)
                    sample_tensor = transforms.ToTensor()(img_pil)
            except Exception as e:
                print(f"Error loading CIFAR sample: {e}")

        if sample_tensor is None:
            sample_tensor = torch.rand(3, patch_size, patch_size)

        self.send_json_response({"image": tensor_to_base64_png(sample_tensor)})

    def handle_encode(self, data: Dict[str, Any]) -> None:
        try:
            image_b64 = data.get("image", "")
            if not image_b64:
                self.send_error(400, "Missing 'image'")
                return

            model = get_active_model()
            entry = get_active_entry()
            patch_size = entry["patch_size"]

            x = base64_to_tensor(image_b64, patch_size)
            with torch.no_grad():
                z = model.encoder(x)  # (1, C, pH/4, pW/4)

            z_list  = z.squeeze(0).cpu().numpy().tolist()
            z_flat  = [float(v) for v in z.view(-1).cpu().numpy()]
            k       = len(z_flat)
            avg_pwr = float(np.mean(np.array(z_flat) ** 2))

            self.send_json_response({
                "shape":        list(z.shape),
                "num_symbols":  k,
                "avg_power":    round(avg_pwr, 4),
                "min_val":      round(float(min(z_flat)), 4),
                "max_val":      round(float(max(z_flat)), 4),
                "vector":       z_list,
                "vector_flat":  z_flat,
                "feature_maps": generate_feature_map_images(z),
                "model_key":    entry["key"],
                "patch_size":   patch_size,
            })
        except Exception as e:
            self.send_json_response({"error": str(e)}, status=500)

    def handle_channel(self, data: Dict[str, Any]) -> None:
        try:
            vector_data = data.get("vector")
            snr_db      = float(data.get("snr_db", 10.0))
            noiseless   = bool(data.get("noiseless", False))

            model   = get_active_model()
            entry   = get_active_entry()
            p       = entry["patch_size"] // 4
            c       = entry["channel_c"]

            z = torch.tensor(vector_data, dtype=torch.float32, device=DEVICE)
            # Accept flat, 3-D (C,H,W), or 4-D (1,C,H,W)
            if z.dim() == 1:
                z = z.view(1, c, p, p)
            elif z.dim() == 3:
                z = z.unsqueeze(0)

            if noiseless:
                z_noisy = z
            else:
                with torch.no_grad():
                    z_noisy = model.channel(z, snr_db=snr_db)

            self.send_json_response({
                "noisy_vector":      z_noisy.squeeze(0).cpu().numpy().tolist(),
                "noisy_vector_flat": [float(v) for v in z_noisy.view(-1).cpu().numpy()],
                "snr_db":            snr_db,
                "is_noiseless":      noiseless,
            })
        except Exception as e:
            self.send_json_response({"error": str(e)}, status=500)

    def handle_decode(self, data: Dict[str, Any]) -> None:
        try:
            vector_data   = data.get("vector")
            orig_b64      = data.get("original_image")

            if vector_data is None:
                self.send_error(400, "Missing 'vector'")
                return

            model   = get_active_model()
            entry   = get_active_entry()
            p       = entry["patch_size"] // 4
            c       = entry["channel_c"]
            patch   = entry["patch_size"]

            z = torch.tensor(vector_data, dtype=torch.float32, device=DEVICE)
            if z.dim() == 1:
                z = z.view(1, c, p, p)
            elif z.dim() == 3:
                z = z.unsqueeze(0)

            with torch.no_grad():
                x_hat = model.decoder(z)  # (1, 3, H, W)

            recon_b64 = tensor_to_base64_png(x_hat.squeeze(0))
            metrics: Dict[str, Any] = {}

            if orig_b64:
                try:
                    x_orig = base64_to_tensor(orig_b64, patch)
                    mse_val  = float(nn.functional.mse_loss(x_hat, x_orig).item())
                    psnr_val = 10.0 * math.log10(1.0 / max(mse_val, 1e-10))
                    metrics["mse"]  = round(mse_val, 5)
                    metrics["psnr"] = round(psnr_val, 2)
                except Exception as ex:
                    print(f"Metric error: {ex}")

            self.send_json_response({
                "reconstructed_image": recon_b64,
                "metrics":             metrics,
                "model_key":           entry["key"],
            })
        except Exception as e:
            self.send_json_response({"error": str(e)}, status=500)

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def send_json_response(self, data: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Server entrypoint
# ---------------------------------------------------------------------------

def run_server(host: str = config.WEB_HOST, port: int = config.WEB_PORT) -> None:
    httpd = HTTPServer((host, port), SemanticCommHandler)
    print(f"\nDeep JSCC Semantic Communication Web App → http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        httpd.server_close()


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else config.WEB_PORT
    run_server(port=port)
