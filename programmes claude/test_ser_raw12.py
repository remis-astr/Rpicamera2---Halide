#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test SER RAW12 Bayer — IMX585
==============================

Vérifie qu'on peut enregistrer un fichier SER en Bayer brut (non débayérisé)
directement depuis Picamera2 en SRGGB12, via post_callback (même méthode que
la capture vidéo SER RGB existante).

Résultat attendu :
  - Fichier .ser lisible par FireCapture / PIPP / AutoStakkert
  - color_id = 8 (BAYER_RGGB)
  - pixel_depth = 12
  - Chaque frame = width × height × 2 octets (uint16 little-endian, valeurs CSI-2 ×16)

Usage :
    python3 test_ser_raw12.py [résolution] [frames] [fps]
    python3 test_ser_raw12.py 1920x1080 100 25
    python3 test_ser_raw12.py 800x600 200 50

Par défaut : 1920×1080, 50 frames, 25 fps
"""

import sys
import os
import time
import struct
import threading
import numpy as np

# ─── Paramètres depuis la ligne de commande ───────────────────────────────────
def parse_args():
    width, height = 1920, 1080
    num_frames = 50
    fps = 25

    for arg in sys.argv[1:]:
        if 'x' in arg.lower():
            parts = arg.lower().split('x')
            try:
                width, height = int(parts[0]), int(parts[1])
            except ValueError:
                pass
        else:
            try:
                v = int(arg)
                if v > 100:
                    num_frames = v
                else:
                    fps = v
            except ValueError:
                pass

    return width, height, num_frames, fps

# ─── SER header ──────────────────────────────────────────────────────────────
def create_ser_header(width, height, pixel_depth=12, color_id=8):
    """
    Crée un en-tête SER v3.
    color_id : 0=MONO, 8=BAYER_RGGB, 9=BAYER_GRBG, 10=BAYER_GBRG, 11=BAYER_BGGR
               100=RGB, 101=BGR
    pixel_depth : bits par pixel par canal (12 pour RAW12)
    Données stockées en uint16 little-endian (2 octets/pixel même pour 12-bit).
    """
    header = bytearray(178)
    header[0:14] = b'LUCAM-RECORDER'
    struct.pack_into('<I', header, 14, 0)           # LuID
    struct.pack_into('<I', header, 18, color_id)    # ColorID
    struct.pack_into('<I', header, 22, 0)           # LittleEndian (0 = little)
    struct.pack_into('<I', header, 26, width)       # ImageWidth
    struct.pack_into('<I', header, 30, height)      # ImageHeight
    struct.pack_into('<I', header, 34, pixel_depth) # PixelDepth
    struct.pack_into('<I', header, 38, 0)           # FrameCount (mis à jour après)
    header[42:42+11] = b'RPiCamRAW12'
    header[82:82+19] = b'Raspberry Pi IMX585'
    ts = int(time.time() * 1_000_000)
    struct.pack_into('<Q', header, 162, ts)
    struct.pack_into('<Q', header, 170, ts)
    return bytes(header)

def update_ser_frame_count(path, n):
    with open(path, 'r+b') as f:
        f.seek(38)
        f.write(struct.pack('<I', n))

# ─── Vérification du fichier SER produit ─────────────────────────────────────
def verify_ser_file(path, expected_width, expected_height, expected_frames):
    """Lit et vérifie l'en-tête + quelques frames du fichier SER."""
    print("\n── Vérification du fichier SER ──────────────────────────────")
    ok = True
    with open(path, 'rb') as f:
        hdr = f.read(178)

    sig        = hdr[0:14]
    color_id   = struct.unpack_from('<I', hdr, 18)[0]
    endian     = struct.unpack_from('<I', hdr, 22)[0]
    width      = struct.unpack_from('<I', hdr, 26)[0]
    height     = struct.unpack_from('<I', hdr, 30)[0]
    depth      = struct.unpack_from('<I', hdr, 34)[0]
    frame_count= struct.unpack_from('<I', hdr, 38)[0]

    color_names = {0:'MONO', 8:'BAYER_RGGB', 9:'BAYER_GRBG', 10:'BAYER_GBRG',
                   11:'BAYER_BGGR', 100:'RGB', 101:'BGR'}

    print(f"  Signature    : {sig}")
    print(f"  ColorID      : {color_id} ({color_names.get(color_id, '?')})")
    print(f"  Endian       : {'little' if endian == 0 else 'big'}")
    print(f"  Dimensions   : {width} × {height}")
    print(f"  PixelDepth   : {depth} bits")
    print(f"  FrameCount   : {frame_count}")

    bytes_per_frame = width * height * 2  # uint16

    file_size = os.path.getsize(path)
    data_size = file_size - 178
    trailing_bytes = data_size - frame_count * bytes_per_frame
    print(f"  Taille fichier  : {file_size / 1024 / 1024:.2f} MB")
    print(f"  Octets/frame attendu : {bytes_per_frame}")
    print(f"  Trailer timestamps   : {trailing_bytes} octets "
          f"({trailing_bytes // 8} × 8 = {trailing_bytes // 8} timestamps)")

    if sig != b'LUCAM-RECORDER':
        print("  [ERREUR] Signature SER invalide !")
        ok = False
    if color_id != 8:
        print(f"  [ERREUR] color_id={color_id}, attendu 8 (BAYER_RGGB)")
        ok = False
    if depth != 12:
        print(f"  [ERREUR] pixel_depth={depth}, attendu 12")
        ok = False
    if width != expected_width or height != expected_height:
        print(f"  [ERREUR] Dimensions {width}×{height} ≠ attendu {expected_width}×{expected_height}")
        ok = False
    if frame_count != expected_frames:
        print(f"  [AVERTISSEMENT] frame_count={frame_count}, attendu {expected_frames}")

    # Lire la première frame et vérifier les valeurs
    if frame_count > 0:
        with open(path, 'rb') as f:
            f.seek(178)
            raw_bytes = f.read(bytes_per_frame)

        frame = np.frombuffer(raw_bytes, dtype='<u2').reshape(height, width)
        min_v, max_v = int(frame.min()), int(frame.max())
        mean_v = float(frame.mean())
        print(f"\n  Première frame :")
        print(f"    shape     : {frame.shape}, dtype : {frame.dtype}")
        print(f"    min/max   : {min_v} / {max_v}  (ADU CSI-2 ×16, range 0–65520)")
        print(f"    moyenne   : {mean_v:.1f}")
        print(f"    En ADU 12-bit (÷16) : {min_v//16}–{max_v//16}, moy {mean_v/16:.1f}")

        if max_v > 65520:
            print("  [ERREUR] Valeurs hors plage 12-bit CSI-2 ×16 (max 65520)")
            ok = False
        elif max_v == 0:
            print("  [AVERTISSEMENT] Toutes les valeurs sont 0 — frame noire !")
        else:
            print("  [OK] Valeurs dans la plage attendue")

    if ok:
        print("\n  RÉSULTAT : SUCCÈS — fichier SER RAW12 Bayer valide")
    else:
        print("\n  RÉSULTAT : ÉCHEC — voir erreurs ci-dessus")

    return ok

# ─── Capture + écriture SER via post_callback ────────────────────────────────
def capture_raw12_ser(width, height, num_frames, fps, output_path):
    """
    Ouvre Picamera2 en mode video + stream raw SRGGB12.
    Capture num_frames frames via post_callback.
    Écrit directement dans un SER Bayer (color_id=8, pixel_depth=12).
    Retourne (frames_écrits, fps_réel, ok).
    """
    from picamera2 import Picamera2

    print(f"\n── Configuration Picamera2 ──────────────────────────────────")
    print(f"  Résolution  : {width}×{height}")
    print(f"  Format      : SRGGB12 (RAW12 Bayer)")
    print(f"  Frames      : {num_frames} @ {fps} fps (~{num_frames/fps:.1f}s)")
    print(f"  Sortie      : {output_path}")

    picam2 = Picamera2()

    # Configuration video avec stream raw activé
    config = picam2.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"},
        raw={"size": (width, height), "format": "SRGGB12"},
    )

    frame_duration_us = int(1_000_000 / fps)
    config["controls"]["FrameDurationLimits"] = (frame_duration_us, frame_duration_us)

    picam2.configure(config)

    # Vérifier la config réelle
    actual = picam2.camera_configuration()
    raw_cfg = actual.get('raw', {})
    actual_w = raw_cfg.get('size', (width, height))[0]
    actual_h = raw_cfg.get('size', (width, height))[1]
    actual_fmt = raw_cfg.get('format', 'SRGGB12')
    print(f"\n  Config réelle raw : {actual_w}×{actual_h} {actual_fmt}")

    # État partagé du callback
    state = {
        'frame_count': 0,
        'error': None,
        'lock': threading.Lock(),
        'file': None,
        'timestamps': [],
        'actual_width': None,
        'actual_height': None,
        'start_time': None,
    }

    def frame_callback(request):
        if state['error'] is not None:
            return
        if state['frame_count'] >= num_frames:
            return
        try:
            raw_arr = request.make_array("raw")  # uint8 (height × stride_bytes)

            # Dépaqueter en uint16 CSI-2 ×16
            u16 = raw_arr.view(np.uint16)
            u16 = u16.reshape(raw_arr.shape[0], -1)
            # Supprimer le padding de stride : colonnes réelles = width (en pixels = u16 words)
            u16 = u16[:, :width]

            with state['lock']:
                if state['frame_count'] == 0:
                    state['actual_width']  = u16.shape[1]
                    state['actual_height'] = u16.shape[0]
                    state['start_time'] = time.time()

                if state['file'] is not None:
                    state['file'].write(u16.astype('<u2').tobytes())
                    ts_us = int(time.time() * 1_000_000)
                    state['timestamps'].append(ts_us)
                    state['frame_count'] += 1

                    if state['frame_count'] % 25 == 0:
                        print(f"  ... {state['frame_count']}/{num_frames} frames", flush=True)

        except Exception as e:
            state['error'] = e
            print(f"\n  [ERREUR callback] {e}")

    # Ouvrir le fichier SER et écrire l'en-tête (dimensions réelles après configure)
    header = create_ser_header(actual_w, actual_h, pixel_depth=12, color_id=8)
    state['file'] = open(output_path, 'wb')
    state['file'].write(header)

    picam2.start()
    # Désactiver denoise pour ne pas ralentir inutilement
    try:
        from picamera2.controls import NoiseReductionModeEnum
        picam2.set_controls({"NoiseReductionMode": NoiseReductionModeEnum.Off})
    except Exception:
        pass

    picam2.post_callback = frame_callback

    t0 = time.time()
    print(f"\n  Capture en cours...")
    timeout = num_frames / fps + 5  # +5s de marge
    while state['frame_count'] < num_frames and state['error'] is None:
        if time.time() - t0 > timeout:
            print(f"  [AVERTISSEMENT] Timeout ({timeout}s) — {state['frame_count']} frames capturées")
            break
        time.sleep(0.05)

    elapsed = time.time() - t0
    picam2.post_callback = None
    picam2.stop()

    frames_written = state['frame_count']
    fps_actual = frames_written / elapsed if elapsed > 0 else 0

    # Écrire le trailer timestamps (SER v3)
    for ts in state['timestamps']:
        state['file'].write(struct.pack('<Q', ts))
    state['file'].close()

    # Mettre à jour FrameCount dans l'en-tête
    update_ser_frame_count(output_path, frames_written)

    picam2.close()

    print(f"\n  Capture terminée :")
    print(f"    Frames écrites    : {frames_written}/{num_frames}")
    print(f"    FPS réel          : {fps_actual:.1f}")
    print(f"    Durée             : {elapsed:.2f}s")
    if state['actual_width']:
        print(f"    Dimensions frame  : {state['actual_width']}×{state['actual_height']}")
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    debit = size_mb / elapsed if elapsed > 0 else 0
    print(f"    Taille fichier    : {size_mb:.1f} MB")
    print(f"    Débit disque      : {debit:.1f} MB/s")

    ok = state['error'] is None and frames_written > 0
    return frames_written, fps_actual, ok, state.get('actual_width', actual_w), state.get('actual_height', actual_h)

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    width, height, num_frames, fps = parse_args()

    output_dir = "/home/admin/videos"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"test_raw12_bayer_{width}x{height}.ser")

    print("=" * 65)
    print("TEST SER RAW12 BAYER — IMX585")
    print("=" * 65)

    # Estimation taille
    bytes_per_frame = width * height * 2
    total_mb = bytes_per_frame * num_frames / 1024 / 1024
    print(f"\nEstimation :")
    print(f"  {width}×{height} × {num_frames} frames × 2 octets = {total_mb:.1f} MB")
    print(f"  Débit théorique @ {fps} fps : {total_mb * fps / num_frames:.1f} MB/s")

    # Capture
    frames, fps_real, capture_ok, real_w, real_h = capture_raw12_ser(
        width, height, num_frames, fps, output_path
    )

    if not capture_ok or frames == 0:
        print("\n[ÉCHEC] Capture échouée — fichier non vérifié")
        sys.exit(1)

    # Vérification
    verify_ok = verify_ser_file(output_path, real_w, real_h, frames)

    print(f"\n{'='*65}")
    print(f"Fichier : {output_path}")
    if verify_ok:
        print("BILAN : OK — SER RAW12 Bayer prêt à être intégré dans RPiCamera2.py")
    else:
        print("BILAN : PROBLÈME — voir les erreurs ci-dessus")
    print(f"{'='*65}")

    sys.exit(0 if verify_ok else 1)

if __name__ == "__main__":
    main()
