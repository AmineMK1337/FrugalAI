import torch
from torchvision import models
from codecarbon import OfflineEmissionsTracker  # offline: no geolocation lookup, so it can't hang

# --- Settings -------------------------------------------------------------
BATCH_SIZE = 16          # images per batch
N_BATCHES = 20           # repeat so the run is long enough to measure reliably
COUNTRY_ISO = "TUN"      # your country's ISO-3 code (used for grid carbon intensity)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}", flush=True)

# --- 1. Load pretrained ResNet18 (downloads ~45 MB on first run) ----------
print("Loading model...", flush=True)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()
print("Model loaded.", flush=True)

# --- 2. Random tensors simulating medical images (3 x 224 x 224) ----------
batch = torch.randn(BATCH_SIZE, 3, 224, 224, device=device)

# Warm-up OUTSIDE the tracker so one-time setup costs aren't counted
print("Warm-up pass...", flush=True)
with torch.no_grad():
    model(batch)
if device == "cuda":
    torch.cuda.synchronize()

# --- 3. Measure the inference block only ----------------------------------
print("Starting CodeCarbon tracker...", flush=True)
tracker = OfflineEmissionsTracker(
    country_iso_code=COUNTRY_ISO,
    save_to_file=False,
    log_level="info",
)
tracker.start()
print("Tracker started, running inference...", flush=True)

with torch.no_grad():
    for i in range(N_BATCHES):
        model(batch)
        print(f"  batch {i + 1}/{N_BATCHES}", flush=True)
if device == "cuda":
    torch.cuda.synchronize()  # make sure GPU work finishes before stopping

emissions_kg = tracker.stop()                               # total CO2e in kg
energy_kwh = tracker.final_emissions_data.energy_consumed   # total energy in kWh

# --- 4. Report -------------------------------------------------------------
n_images = BATCH_SIZE * N_BATCHES
print("\n===== Results =====")
print(f"Device:                {device}")
print(f"Images processed:      {n_images}")
print(f"Total energy:          {energy_kwh:.6e} kWh")
print(f"Total CO2e:            {emissions_kg:.6e} kg")
print(f"Energy per image:      {energy_kwh / n_images:.6e} kWh")
print(f"CO2e per image:        {emissions_kg * 1000 / n_images:.6e} g")