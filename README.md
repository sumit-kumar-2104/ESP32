# ESP32 Human Sensing (WiFi CSI) — Complete Setup Guide

This folder contains everything you need to turn your **ESP32-WROOM-32** board
into a **human presence / motion sensor** using WiFi — no camera, no wearable.
It uses the open-source **ESPectre** project (WiFi Channel State Information, CSI).

You will flash one small firmware onto the ESP32, and then watch a **live web
dashboard** in your browser that shows a **Movement Score** and a **Motion
Detected** on/off indicator as people move in the room.

> New to this? Read top to bottom once, then follow the steps. It takes ~20–30
> minutes the first time.

---

## ⚡ Easiest way to run (3 scripts)

This folder is **self-contained** — the ESPectre code is bundled inside
(`components/`), so it builds even without internet. On your laptop:

1. **Install the USB driver** for your board (see Section 4) and the
   **ESP32 plugged in**.
2. Double-click / right-click → *Run with PowerShell*, in order:
   - **`1-setup.ps1`** → installs the build tool + creates `secrets.yaml`
   - ✏️ Edit **`firmware\secrets.yaml`** → put your 2.4 GHz WiFi name + password
   - **`2-flash.ps1`** → builds and flashes the ESP32 (pick your COM port)
   - **`3-monitor.ps1`** *(optional)* → watch the live numbers
3. Open **http://espectre.local** in your browser to see the dashboard.

> If PowerShell blocks the scripts, open PowerShell in this folder and run once:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`
> then run the scripts. The manual steps below explain exactly what each script
> does, in case you prefer typing commands yourself.

---

## 1. What this does (the simple version)

When a person moves in a room, their body slightly disturbs the WiFi radio waves
between your router and the ESP32. The ESP32 constantly measures these tiny
changes (called **CSI = Channel State Information**) and turns them into:

- **Movement Score** — a number from 0 to 10 (higher = more motion)
- **Motion Detected** — a simple **ON / OFF** presence flag

This is the "most common format" for a beginner: a **live gauge + on/off flag**
in a web page that updates in real time. (Later, for research, you can also log
the raw numbers to a file and plot them in Python — see Section 9.)

---

## 2. About your 2 ESP32 boards ✋ (read this)

**You only need ONE ESP32 for this to work** — plus the 2.4 GHz WiFi router you
already have.

- The ESP32 connects to your WiFi router like a phone would.
- It then **pings the router 100×/second** to keep WiFi packets flowing, and
  measures the CSI of those packets.
- That single board is both the "listener" and the traffic source.

So the **second ESP32 is a spare** — keep it as a backup, or flash the same
firmware onto it and place it in **another room** to sense two areas at once.
(You do **not** wire the two boards together.)

---

## 3. Hardware checklist

- [x] ESP32-WROOM-32 dev board (you have 2 — use 1 to start)
- [x] USB cable (data cable — some cheap cables are charge-only; if the board is
      not detected, try another cable)
- [x] A 2.4 GHz WiFi network (ESP32 does **not** support 5 GHz)
- [x] Your laptop (Windows / macOS / Linux all work)

**Best placement of the ESP32:** 3–8 meters from the router, about table height
(1–1.5 m), away from large metal objects. Too close (<2 m) or too far (>10 m)
reduces accuracy.

---

## 4. Install the USB driver (Windows) — do this first

Your board talks to the laptop through a small USB-to-serial chip. Windows often
needs its driver, otherwise no COM port appears.

1. Look at the small chip near the USB port on the board:
   - **CP2102** (square, "Silicon Labs") → install the
     [CP210x driver](https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers)
   - **CH340 / CH341** → install the
     [CH340 driver](https://www.wch-ic.com/downloads/CH341SER_EXE.html)
2. Plug in the ESP32 with the USB cable.
3. Confirm it appears: open **Device Manager → Ports (COM & LPT)**. You should
   see something like `Silicon Labs CP210x (COM5)` or `USB-SERIAL CH340 (COM5)`.
   Note the **COM number** — you'll pick it during flashing.

> macOS/Linux usually detect the board automatically. On Linux you may need:
> `sudo usermod -a -G dialout $USER` then log out/in.

---

## 5. Install the build tool (ESPHome)

ESPHome is the free tool that compiles the firmware and flashes your board.

### 5.1 Install Python 3.12

- Download **Python 3.12** from <https://www.python.org/downloads/>
  (⚠️ avoid Python 3.14 — it currently has issues with ESPHome).
- During install on Windows, **check "Add Python to PATH"**.

Verify in a terminal (Windows PowerShell):

```powershell
python --version   # should print Python 3.12.x
```

### 5.2 Create a virtual environment and install ESPHome

Open a terminal **inside this `esp32` folder** and run:

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install esphome
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install esphome
```

> If PowerShell blocks the activate script, run once:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

Verify:
```powershell
esphome version
```

---

## 6. Add your WiFi credentials

1. In the `firmware` folder, **copy** `secrets.yaml.example` and rename the copy
   to `secrets.yaml`.
2. Open `secrets.yaml` and set your **2.4 GHz** WiFi name and password:

```yaml
wifi_ssid: "MyHomeWiFi"
wifi_password: "supersecret123"
```

`secrets.yaml` is git-ignored so your password won't be committed.

---

## 7. Flash the firmware onto the ESP32

Make sure your virtual environment is active (you see `(venv)` in the prompt).
Then go into the firmware folder and run:

```powershell
cd firmware
esphome run espectre-esp32.yaml
```

What happens:

1. ESPHome uses the **bundled** ESPectre component in `components/` and
   **compiles** the firmware (first build takes a few minutes — this is normal).
2. It asks **how to upload** — choose the **USB / serial** option and pick your
   **COM port** (e.g. `COM5`).
3. After flashing, it automatically opens the **serial log**. You'll see the
   board boot, connect to WiFi, and start printing CSI / movement numbers.

> **Tip:** Some boards need you to hold the **BOOT** button while flashing
> starts, then release. If upload fails with a timeout, hold BOOT, re-run, and
> release when you see "Connecting...".

Look in the logs for a line showing the device IP address, e.g.:
```
[wifi] IP Address: 192.168.1.42
```

---

## 8. See the results 🎉 (the visualization)

Open a browser on your laptop (same WiFi network) and go to:

- **http://espectre.local**  ← try this first
- or **http://192.168.1.42** ← the IP from the serial log, if `.local` fails

You'll see the live dashboard with:

| Item                | What it shows                                        |
|---------------------|------------------------------------------------------|
| **Movement Score**  | live number 0–10 (rises when someone moves)          |
| **Motion Detected** | ON when motion is present, OFF when the room is still |
| **Threshold**       | slider to adjust sensitivity live                    |
| **Calibrate**       | button to re-learn the "empty room" baseline         |

**How to test it works:**

1. After boot, **leave the room empty / stay still ~15 seconds** so it
   calibrates a quiet baseline.
2. Then **walk around** near the sensor — the Movement Score should jump and
   **Motion Detected → ON**.
3. Stand still again — after a few seconds it drops back to **OFF**.

If it's too sensitive or not sensitive enough, drag the **Threshold** slider
(lower = more sensitive) or press **Calibrate** while the room is empty.

The **serial terminal** (still open from the flash step) also streams the raw
movement numbers — handy for research logging.

---

## 9. (Optional) Raw CSI data for research / ML

For your MetaAI-paper work you'll likely want the **raw CSI stream** to process
yourself. Two easy routes:

- **Serial logging:** the DEBUG log level prints per-packet values you can
  capture. Change `level: INFO` to `level: DEBUG` in `espectre-esp32.yaml`,
  re-flash, and pipe the serial output to a file.
- **The ESPectre repo** already in this workspace (`../espectre/`) contains a
  `micro-espectre/` folder with **Python notebooks, ML data-collection tools,
  and algorithm docs** (`ALGORITHMS.md`, `ML_DATA_COLLECTION.md`). That's the
  best starting point for capturing CSI into CSV and training models.

I can wire up a dedicated CSV-capture + live Python plot when you're ready —
just ask.

---

## 10. Troubleshooting

| Problem | Fix |
|--------|-----|
| No COM port in Device Manager | Install the CP2102 or CH340 driver (Section 4); try another USB **data** cable. |
| `esphome` not recognized | Activate the venv again: `venv\Scripts\Activate.ps1`. |
| Upload fails / timeout | Hold the **BOOT** button, re-run `esphome run`, release when "Connecting..." appears. |
| Board reboots in a loop | Usually wrong WiFi password, or 5 GHz network. ESP32 needs **2.4 GHz**. Fix `secrets.yaml`. |
| No serial logs after flashing | Already handled — config uses `hardware_uart: UART0`. Make sure the log terminal is open (`esphome logs espectre-esp32.yaml`). |
| `espectre.local` won't open | Use the numeric IP from the serial log instead. |
| Movement Score stays flat | Press **Calibrate** while the room is empty; move the board 3–8 m from the router; lower the **Threshold**. |
| Too many false triggers | Raise the **Threshold**, keep the board away from fans/curtains/AC vents. |

Re-flashing after edits: just run `esphome run espectre-esp32.yaml` again. After
the first USB flash you can also update **over WiFi** (OTA) — ESPHome will offer
the network option.

---

## 11. Files in this folder

```
esp32/
├─ README.md                      ← this guide
├─ 1-setup.ps1                    ← installs ESPHome + makes secrets.yaml
├─ 2-flash.ps1                    ← builds & flashes the ESP32
├─ 3-monitor.ps1                  ← live serial logs (optional)
├─ components/                    ← bundled ESPectre source (offline build)
└─ firmware/
   ├─ espectre-esp32.yaml         ← the firmware config you flash
   ├─ secrets.yaml.example        ← copy → secrets.yaml, add WiFi
   └─ .gitignore                  ← keeps secrets.yaml private
```

> **To transfer:** just zip the whole `esp32` folder and copy it to your other
> laptop. Everything needed to build is inside (you only need Python + internet
> the first time you run `1-setup.ps1`, which downloads ESPHome).

---

## 12. Useful references

- ESPectre project (in this workspace): `../espectre/README.md`
- ESPectre setup & tuning: `../espectre/SETUP.md`, `../espectre/TUNING.md`
- CSI algorithm explanation: `../espectre/micro-espectre/ALGORITHMS.md`
- ESPHome docs: <https://esphome.io/>
- ESPectre on GitHub: <https://github.com/francescopace/espectre>

---

### Quick command recap

```powershell
# From the esp32 folder:
python -m venv venv
venv\Scripts\Activate.ps1
pip install esphome
#  copy firmware\secrets.yaml.example -> firmware\secrets.yaml and edit WiFi
cd firmware
esphome run espectre-esp32.yaml
#  then open http://espectre.local
```

(Or just run `1-setup.ps1`, edit `firmware\secrets.yaml`, then `2-flash.ps1`.)
