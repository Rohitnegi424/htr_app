# InkVoice HTR backend

This backend runs the local `htr_ctc_words_multilang_best_v4.keras` model and exposes:

- `GET /health`
- `POST /recognize`

## Required file

Create `backend/charset.json` as a JSON array of the exact characters used during training, in the exact index order used by the model output.

Example:

```json
["a", "b", "c", " "]
```

The model has 78 output units, so `charset.json` should usually contain 77 characters and the CTC blank token is assumed to be the final index.

## Start

```powershell
python backend/server.py
```

## App connection

By default the Expo app uses:

- `http://127.0.0.1:8765` on web/iOS
- `http://10.0.2.2:8765` on Android emulator

For a real phone on the same Wi-Fi, set `expo.extra.htrApiBaseUrl` in `app.json` to your computer LAN URL such as `http://192.168.1.20:8765`.
