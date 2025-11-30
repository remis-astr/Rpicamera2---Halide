#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Gordon999
# SPDX-License-Identifier: MIT

"""Copyright (c) 2025
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

import time
import pygame
from pygame.locals import *
import os, sys
import datetime
import subprocess
import signal
import cv2
import glob
from datetime import timedelta
import numpy as np
import math
from gpiozero import Button
from gpiozero import LED
import struct
from collections import deque
import threading
import matplotlib
matplotlib.use('Agg')  # Backend sans affichage pour éviter conflits avec pygame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from io import BytesIO

#!/usr/bin/env python3
import sys
import os

# Configuration IMX585 - À mettre TOUT EN HAUT avant tous les imports
sys.path.insert(0, '/usr/local/lib/aarch64-linux-gnu/python3.11/site-packages')
os.environ['LD_LIBRARY_PATH'] = '/usr/local/lib/aarch64-linux-gnu'
os.environ['LIBCAMERA_RPI_CONFIG_FILE'] = ''

# Imports Picamera2 (après configuration IMX585)
from picamera2 import Picamera2
from libcamera import controls, Transform

# Import Live Stack module (dans le même répertoire)
from rpicamera_livestack import create_livestack_session

# ============================================================================
# Helper pour tuer le subprocess rpicam-vid (mode non-Picamera2)
# ============================================================================
def kill_preview_process():
    """Tue le processus rpicam-vid si actif (mode non-Picamera2)"""
    global p
    if not use_picamera2 and p is not None:
        try:
            poll = p.poll()
            if poll is None:
                os.killpg(p.pid, signal.SIGTERM)
        except:
            pass

# ============================================================================
# Post-traitement vidéo pour correction des timestamps (Pi 5)
# ============================================================================
def fix_video_timestamps(input_file, fps_value, quality_preset="ultrafast"):
    """
    Corrige les timestamps des vidéos MP4/H264 sur Pi 5 via ffmpeg.

    Problème: rpicam-vid sur Pi 5 génère des fichiers avec timestamps incorrects
    Solution: Réencodage avec ffmpeg pour recalculer les timestamps

    Args:
        input_file: Chemin du fichier vidéo brut (avec timestamps incorrects)
        fps_value: Framerate de la vidéo (utilisé pour recalculer les timestamps)
        quality_preset: Preset ffmpeg (ultrafast, veryfast, medium, slow)

    Returns:
        True si la correction a réussi, False sinon
    """
    import os
    import subprocess

    # Créer le nom du fichier temporaire
    temp_file = input_file.replace(".mp4", "_temp.mp4").replace(".h264", "_temp.h264")

    # Renommer le fichier original en temporaire
    try:
        os.rename(input_file, temp_file)
    except Exception as e:
        print(f"Erreur lors du renommage: {e}")
        return False

    # Construire la commande ffmpeg
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", temp_file,
        "-vf", f"setpts=N/{fps_value}/TB",
        "-r", str(fps_value),
        "-c:v", "libx264",
        "-preset", quality_preset,
        input_file,
        "-y",
        "-loglevel", "error"
    ]

    try:
        # Exécuter ffmpeg
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            # Succès - supprimer le fichier temporaire
            try:
                os.remove(temp_file)
            except:
                pass
            return True
        else:
            # Échec - restaurer le fichier original
            print(f"Erreur ffmpeg: {result.stderr}")
            try:
                os.rename(temp_file, input_file)
            except:
                pass
            return False

    except subprocess.TimeoutExpired:
        print("Timeout ffmpeg - fichier trop long")
        try:
            os.rename(temp_file, input_file)
        except:
            pass
        return False
    except Exception as e:
        print(f"Erreur lors de l'exécution de ffmpeg: {e}")
        try:
            os.rename(temp_file, input_file)
        except:
            pass
        return False

# ============================================================================
# Helpers pour gérer Picamera2 temporairement pendant rpicam-vid
# ============================================================================
def pause_picamera2():
    """
    Ferme complètement Picamera2 pour libérer la caméra.
    IMPORTANT: Doit être suivi de resume_picamera2() pour recréer Picamera2.
    """
    global picam2, use_picamera2

    print(f"[DEBUG] pause_picamera2() called - use_picamera2={use_picamera2}, picam2={picam2 is not None}")

    if use_picamera2 and picam2 is not None:
        try:
            # Fermer complètement Picamera2 pour libérer le pipeline
            print("[DEBUG] Closing Picamera2 completely...")
            picam2.stop()
            picam2.close()
            print("[DEBUG] Picamera2 closed, waiting for camera to be released...")
            import time
            time.sleep(1.0)  # Attendre que la caméra soit vraiment libérée
            print("[DEBUG] Picamera2 closed successfully - camera should now be free")
            return True
        except Exception as e:
            print(f"[DEBUG] Erreur lors de la fermeture de Picamera2: {e}")
            import traceback
            traceback.print_exc()
            return False
    print("[DEBUG] Picamera2 not paused (not in use or None)")
    return False

def resume_picamera2():
    """
    Recrée et redémarre Picamera2 après une pause temporaire (après close()).
    """
    global picam2, use_picamera2

    print(f"[DEBUG] resume_picamera2() called - use_picamera2={use_picamera2}")

    if use_picamera2:
        try:
            print("[DEBUG] Recreating Picamera2 via preview() function...")
            preview()  # Appelle la fonction qui crée et configure Picamera2
            print("[DEBUG] Picamera2 recreated and started successfully")
            return True
        except Exception as e:
            print(f"[DEBUG] Erreur lors de la reprise de Picamera2: {e}")
            import traceback
            traceback.print_exc()
            return False
    print("[DEBUG] Picamera2 not resumed (not in use)")
    return False

# ============================================================================
# Extracteur MJPEG intégré - Gère le flux continu en arrière-plan
# ============================================================================
class MJPEGExtractor:
    """
    Extracteur de frames MJPEG en thread - Version simplifiée
    Lit un flux MJPEG continu et extrait chaque JPEG dans des fichiers séparés
    Gère automatiquement la rotation des fichiers stream.mjpeg
    Le nettoyage des anciennes frames est fait par la boucle principale
    """
    def __init__(self, input_file, output_pattern, max_files=10):
        self.input_file = input_file
        self.output_pattern = output_pattern
        self.max_files = max_files
        self.running = False
        self.thread = None
        self.frame_counter = 0
        
    def start(self):
        """Démarre l'extraction en arrière-plan"""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._extract_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Arrête l'extraction"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
    def _extract_loop(self):
        """Boucle d'extraction principale"""
        # Marqueurs JPEG
        JPEG_START = b'\xff\xd8'
        JPEG_END = b'\xff\xd9'
        
        buffer = b''
        last_inode = None
        f = None
        consecutive_empty_reads = 0
        
        while self.running:
            try:
                # Vérifier si le fichier existe et a été recréé (nouveau inode)
                if os.path.exists(self.input_file):
                    current_inode = os.stat(self.input_file).st_ino
                    
                    # Rouvrir le fichier si nouveau ou première ouverture
                    if last_inode is None or current_inode != last_inode:
                        if f:
                            f.close()
                        f = open(self.input_file, 'rb')
                        last_inode = current_inode
                        consecutive_empty_reads = 0
                        buffer = b''  # Réinitialiser le buffer
                else:
                    # Fichier n'existe pas encore, attendre
                    time.sleep(0.1)
                    continue
                    
                # Lire par blocs
                chunk = f.read(131072)
                if not chunk:
                    consecutive_empty_reads += 1
                    # Si trop de lectures vides, revérifier si fichier a changé
                    if consecutive_empty_reads > 100:
                        consecutive_empty_reads = 0
                        last_inode = None  # Forcer réouverture au prochain tour
                    time.sleep(0.001)
                    continue
                
                consecutive_empty_reads = 0
                buffer += chunk
                
                # Chercher et extraire tous les JPEG complets dans le buffer
                while self.running:
                    start = buffer.find(JPEG_START)
                    if start == -1:
                        # Garder les derniers octets au cas où le marqueur est coupé
                        buffer = buffer[-2:] if len(buffer) > 2 else buffer
                        break
                    
                    end = buffer.find(JPEG_END, start + 2)
                    if end == -1:
                        # JPEG incomplet, attendre plus de données
                        break
                    
                    # Extraire le JPEG complet
                    jpeg_data = buffer[start:end+2]

                    # Numérotation séquentielle (évite les boucles d'images)
                    output_file = self.output_pattern % self.frame_counter
                    temp_file = output_file + ".tmp"
                    try:
                        # Double buffering: écrire dans un fichier temporaire
                        with open(temp_file, 'wb') as out:
                            out.write(jpeg_data)
                        # Rename atomique pour garantir qu'on ne lit jamais une image partielle
                        os.rename(temp_file, output_file)

                        # Le nettoyage des anciennes images est maintenant fait par la boucle principale
                        # Cela évite les race conditions et garantit qu'on ne supprime jamais
                        # l'image en cours d'affichage
                    except:
                        pass

                    self.frame_counter += 1
                    
                    # Enlever ce JPEG du buffer
                    buffer = buffer[end+2:]
                    
            except Exception as e:
                # En cas d'erreur, attendre un peu et réessayer
                time.sleep(0.1)
                continue
        
        # Nettoyage à l'arrêt
        if f:
            f.close()



def create_ser_header(width, height, pixel_depth=8, color_id=100):
    """
    Crée l'en-tête d'un fichier SER
    color_id: 0=MONO, 8=BAYER_RGGB, 9=BAYER_GRBG, 10=BAYER_GBRG, 11=BAYER_BGGR, 
              100=RGB, 101=BGR
    """
    header = bytearray(178)
    
    # Signature "LUCAM-RECORDER"
    header[0:14] = b'LUCAM-RECORDER'
    
    # LuID (4 bytes) - peut être 0
    struct.pack_into('<I', header, 14, 0)
    
    # ColorID (4 bytes)
    struct.pack_into('<I', header, 18, color_id)
    
    # LittleEndian (4 bytes) - 0 pour little endian
    struct.pack_into('<I', header, 22, 0)
    
    # ImageWidth (4 bytes)
    struct.pack_into('<I', header, 26, width)
    
    # ImageHeight (4 bytes)
    struct.pack_into('<I', header, 30, height)
    
    # PixelDepth (4 bytes) - bits per pixel per channel
    struct.pack_into('<I', header, 34, pixel_depth)
    
    # FrameCount (4 bytes) - sera mis à jour à la fin
    struct.pack_into('<I', header, 38, 0)
    
    # Observer (40 bytes)
    observer = b'RPiCamGUI'
    header[42:42+len(observer)] = observer
    
    # Instrument (40 bytes)
    instrument = b'Raspberry Pi Camera'
    header[82:82+len(instrument)] = instrument
    
    # Telescope (40 bytes)
    telescope = b''
    header[122:162] = telescope.ljust(40, b'\x00')
    
    # DateTime (8 bytes) - timestamp en microseconds depuis epoch
    timestamp = int(time.time() * 1000000)
    struct.pack_into('<Q', header, 162, timestamp)
    
    # DateTime_UTC (8 bytes)
    struct.pack_into('<Q', header, 170, timestamp)
    
    return bytes(header)

def update_ser_frame_count(filename, frame_count):
    """Met à jour le nombre de frames dans l'en-tête SER"""
    with open(filename, 'r+b') as f:
        f.seek(38)
        f.write(struct.pack('<I', frame_count))

def convert_raw_to_ser(raw_input, ser_output, width, height, fps=None, bit_depth=8, progress_callback=None):
    """
    Convertit un fichier .raw (RGB) en fichier SER

    Args:
        raw_input: Chemin du fichier .raw d'entrée
        ser_output: Chemin du fichier .ser de sortie
        width: Largeur des frames
        height: Hauteur des frames
        fps: Framerate de la vidéo (optionnel, défaut=25). Utilisé pour calculer les timestamps
        bit_depth: Profondeur en bits (8 ou 16, défaut=8)
        progress_callback: Fonction appelée avec (frame_num, total_frames) pour afficher la progression

    Returns:
        (success, frame_count, message)
    """

    if not os.path.exists(raw_input):
        return False, 0, f"Fichier d'entrée introuvable: {raw_input}"

    # Utiliser fps par défaut si non spécifié
    if fps is None:
        fps = 25

    file_size = os.path.getsize(raw_input)

    # Calculer bytes par frame selon la profondeur
    if bit_depth == 16:
        bytes_per_frame = width * height * 6  # RGB48 (16-bit par canal)
    else:
        bytes_per_frame = width * height * 3  # RGB24 (8-bit par canal)

    # Calculer le nombre de frames complètes
    total_frames = file_size // bytes_per_frame
    remaining_bytes = file_size % bytes_per_frame

    if total_frames == 0:
        return False, 0, "Aucune frame complète dans le fichier"

    print(f"📊 Informations:")
    print(f"   Fichier entrée: {raw_input}")
    print(f"   Taille: {file_size:,} bytes ({file_size/(1024*1024):.2f} MB)")
    print(f"   Résolution: {width}x{height}")
    print(f"   Profondeur: {bit_depth}-bit")
    print(f"   Frames complètes: {total_frames}")
    print(f"   Framerate: {fps} fps")
    if remaining_bytes > 0:
        print(f"   ⚠  Bytes ignorés en fin de fichier: {remaining_bytes}")
    print()

    try:
        frame_count = 0
        frame_timestamps = []

        # Calculer le timestamp de départ (maintenant)
        start_time = datetime.datetime.utcnow()
        start_timestamp_us = int(start_time.timestamp() * 1000000)

        # Calculer l'intervalle entre frames en microsecondes
        frame_interval_us = int(1000000 / fps)

        with open(ser_output, 'wb') as ser_file:
            # Écrire l'en-tête SER avec la bonne profondeur
            header = create_ser_header(width, height, bit_depth, 100)  # RGB, bit_depth
            ser_file.write(header)

            # Ouvrir le fichier RAW et convertir frame par frame
            with open(raw_input, 'rb') as raw_file:
                for i in range(total_frames):
                    # Lire une frame RGB
                    frame_data = raw_file.read(bytes_per_frame)

                    if len(frame_data) != bytes_per_frame:
                        print(f"⚠  Frame {i+1}: données incomplètes, arrêt")
                        break

                    # Calculer le timestamp de cette frame
                    frame_timestamp = start_timestamp_us + (i * frame_interval_us)
                    frame_timestamps.append(frame_timestamp)

                    # Écrire directement dans le fichier SER
                    # (les données RGB sont déjà au bon format)
                    ser_file.write(frame_data)
                    frame_count += 1

                    # Afficher la progression
                    if progress_callback:
                        progress_callback(i + 1, total_frames)
                    elif (i + 1) % 50 == 0 or i == 0:
                        print(f"   Conversion: {i+1}/{total_frames} frames ({(i+1)/total_frames*100:.1f}%)")

            # Écrire le trailer avec les timestamps (SER v3)
            # Chaque timestamp est un entier 64-bit little-endian (microsecondes depuis epoch)
            for timestamp in frame_timestamps:
                ser_file.write(struct.pack('<Q', timestamp))

        # Mettre à jour le nombre de frames dans l'en-tête
        update_ser_frame_count(ser_output, frame_count)

        output_size = os.path.getsize(ser_output)

        print()
        print(f"✓ Conversion terminée!")
        print(f"   Fichier sortie: {ser_output}")
        print(f"   Frames converties: {frame_count}")
        print(f"   Taille: {output_size:,} bytes ({output_size/(1024*1024):.2f} MB)")
        print(f"   Taille par frame: {output_size/frame_count/1024:.2f} KB")

        return True, frame_count, "Conversion réussie"

    except Exception as e:
        return False, 0, f"Erreur: {str(e)}"

def yuv420_to_rgb(yuv_data, width, height):
    """
    Convertit une frame YUV420 en RGB

    YUV420 format:
    - Y plane: width * height bytes
    - U plane: (width/2) * (height/2) bytes
    - V plane: (width/2) * (height/2) bytes

    Note: Les dimensions doivent être paires pour YUV420
    """
    # S'assurer que les dimensions sont paires pour YUV420
    # Si elles sont impaires, on les ajuste (rare mais possible)
    actual_width = width
    actual_height = height

    # Calculer les dimensions UV (sous-échantillonnées par 2)
    uv_width = (width + 1) // 2  # Arrondi supérieur si impair
    uv_height = (height + 1) // 2

    y_size = width * height
    uv_size = uv_width * uv_height

    try:
        # Extraire les plans Y, U, V
        y_plane = np.frombuffer(yuv_data[:y_size], dtype=np.uint8).reshape((height, width))

        # Vérifier qu'on a assez de données pour U et V
        if len(yuv_data) < y_size + 2 * uv_size:
            # Fallback: créer des plans UV neutres
            print(f"   ⚠️  Données UV incomplètes, utilisation de valeurs neutres")
            u_plane = np.full((uv_height, uv_width), 128, dtype=np.uint8)
            v_plane = np.full((uv_height, uv_width), 128, dtype=np.uint8)
        else:
            u_plane = np.frombuffer(yuv_data[y_size:y_size + uv_size], dtype=np.uint8).reshape((uv_height, uv_width))
            v_plane = np.frombuffer(yuv_data[y_size + uv_size:y_size + 2 * uv_size], dtype=np.uint8).reshape((uv_height, uv_width))

        # Upscale U et V pour correspondre à Y (utiliser la taille exacte de Y)
        u_upscale = cv2.resize(u_plane, (width, height), interpolation=cv2.INTER_LINEAR)
        v_upscale = cv2.resize(v_plane, (width, height), interpolation=cv2.INTER_LINEAR)

        # Recombiner en YUV
        yuv_image = np.stack([y_plane, u_upscale, v_upscale], axis=-1).astype(np.uint8)

        # Convertir YUV → RGB
        rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)

        return rgb_image

    except Exception as e:
        print(f"   ❌ Erreur conversion YUV->RGB: {e}")
        print(f"      Dimensions: {width}x{height}, UV: {uv_width}x{uv_height}")
        print(f"      Taille données: {len(yuv_data)}, Y: {y_size}, UV: {uv_size}")
        # Retourner une image noire en cas d'erreur
        return np.zeros((height, width, 3), dtype=np.uint8)

def convert_yuv420_to_ser(yuv_file, ser_file, width, height, fps=25):
    """
    Convertit un fichier YUV420 en SER avec timestamps

    Args:
        yuv_file: Fichier .yuv d'entrée
        ser_file: Fichier .ser de sortie
        width: Largeur
        height: Hauteur
        fps: Framerate pour les timestamps

    Returns:
        (success, frame_count, message)
    """

    if not os.path.exists(yuv_file):
        return False, 0, f"Fichier YUV introuvable: {yuv_file}"

    # Calculer la taille d'une frame YUV420 en tenant compte des dimensions exactes
    # Y plane: width * height
    # U plane: ((width+1)//2) * ((height+1)//2)
    # V plane: ((width+1)//2) * ((height+1)//2)
    y_size = width * height
    uv_width = (width + 1) // 2
    uv_height = (height + 1) // 2
    uv_size = uv_width * uv_height
    frame_size = y_size + 2 * uv_size

    file_size = os.path.getsize(yuv_file)
    frame_count = file_size // frame_size

    if frame_count == 0:
        return False, 0, "Aucune frame YUV420 trouvée"

    print(f"📹 Conversion YUV420 → SER")
    print(f"   Fichier: {yuv_file}")
    print(f"   Résolution: {width}x{height}")
    print(f"   Taille frame: {frame_size} bytes (Y:{y_size}, UV:{uv_size}x2)")
    print(f"   Frames: {frame_count}")
    print(f"   FPS: {fps}")
    print()

    # Créer un fichier RGB temporaire
    temp_rgb = "/run/shm/temp_rgb_from_yuv.raw"

    try:
        frames_converted = 0
        with open(yuv_file, 'rb') as yuvf, open(temp_rgb, 'wb') as rgbf:
            for i in range(frame_count):
                # Lire frame YUV420
                yuv_data = yuvf.read(frame_size)
                if len(yuv_data) < frame_size:
                    print(f"   ⚠️  Frame {i+1} incomplète ({len(yuv_data)}/{frame_size} bytes)")
                    break

                # Convertir YUV420 → RGB
                rgb = yuv420_to_rgb(yuv_data, width, height)

                if rgb is None:
                    print(f"   ❌ Erreur conversion frame {i+1}")
                    continue

                # Écrire RGB
                rgbf.write(rgb.tobytes())
                frames_converted += 1

                if (i + 1) % 10 == 0:
                    print(f"   Frame {i+1}/{frame_count}...")

        if frames_converted < frame_count:
            print(f"   ⚠️  {frames_converted}/{frame_count} frames converties")

        print(f"\n🎬 Création du SER avec timestamps @ {fps} fps...")

        # Utiliser convert_raw_to_ser qui fonctionne !
        success, frames, msg = convert_raw_to_ser(
            temp_rgb, ser_file, width, height, fps=fps, bit_depth=8
        )

        # Nettoyer
        try:
            os.remove(temp_rgb)
        except:
            pass

        return success, frames, msg

    except Exception as e:
        return False, 0, f"Erreur: {str(e)}"

def calculate_snr(image):
    """
    Calcule le rapport signal/bruit (SNR) d'une image
    Retourne le SNR en ratio linéaire (signal/bruit)
    """
    try:
        # Convertir en niveaux de gris si l'image est en couleur
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculer le signal (moyenne)
        signal = np.mean(gray)
        
        # Calculer le bruit (écart-type)
        noise = np.std(gray)
        
        # Éviter la division par zéro
        if noise == 0:
            return 999.9
        
        # Calculer le SNR en ratio linéaire
        snr = signal / noise
        
        return snr
    except:
        return 0.0

def calculate_hfr(image_surface, center_x, center_y, area_size):
    """
    Calcule le HFR (Half Flux Radius) - rayon contenant 50% du flux
    Plus robuste aux aigrettes que le FWHM
    Retourne le HFR en pixels
    """
    try:
        # Convertir la surface pygame en array
        image_array = pygame.surfarray.array3d(image_surface)
        
        # Extraire la région d'intérêt
        y1 = max(0, center_y - area_size)
        y2 = min(image_array.shape[1], center_y + area_size)
        x1 = max(0, center_x - area_size)
        x2 = min(image_array.shape[0], center_x + area_size)
        
        # Transposer pour obtenir (height, width, channels)
        roi = image_array[x1:x2, y1:y2, :]
        roi = np.transpose(roi, (1, 0, 2))
        
        # Convertir en niveaux de gris
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = roi
        
        # Trouver le centroïde (centre de masse)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
        
        # Vérifier qu'il y a un contraste suffisant
        if max_val <= min_val or (max_val - min_val) / max_val < 0.1:
            return None
        
        # Soustraire le fond (minimum local)
        gray_sub = gray.astype(float) - min_val

        # CORRECTION : Utiliser un seuil pour ignorer le bruit
        # Ne considérer que les pixels avec au moins 30% de l'intensité maximale
        # Seuil plus élevé pour éviter les variations erratiques (2 à 60)
        threshold = (max_val - min_val) * 0.30

        # Créer un masque pour les pixels significatifs
        significant_mask = gray_sub >= threshold

        # Si pas assez de pixels significatifs, retourner None
        # Minimum de 10 pixels requis pour un calcul fiable
        if np.sum(significant_mask) < 10:
            return None

        # Calculer le flux total (uniquement les pixels significatifs)
        total_flux = np.sum(gray_sub[significant_mask])

        if total_flux <= 0:
            return None

        # Calculer le centroïde (uniquement sur les pixels significatifs)
        height, width = gray_sub.shape
        y_indices, x_indices = np.mgrid[0:height, 0:width]

        cx = np.sum(x_indices[significant_mask] * gray_sub[significant_mask]) / total_flux
        cy = np.sum(y_indices[significant_mask] * gray_sub[significant_mask]) / total_flux

        # Calculer la distance de chaque pixel significatif au centroïde
        distances = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)

        # Extraire seulement les pixels significatifs
        sig_distances = distances[significant_mask]
        sig_flux = gray_sub[significant_mask]

        # Trier les pixels par distance croissante
        sorted_indices = np.argsort(sig_distances)
        sorted_distances = sig_distances[sorted_indices]
        sorted_flux = sig_flux[sorted_indices]

        # Calculer le flux cumulé
        cumulative_flux = np.cumsum(sorted_flux)
        half_flux = total_flux / 2.0

        # Trouver le rayon contenant 50% du flux
        idx = np.searchsorted(cumulative_flux, half_flux)
        
        if idx >= len(sorted_distances):
            return None
        
        hfr = sorted_distances[idx]
        
        return float(hfr)
        
    except Exception as e:
        return None

version = 1.07

# streaming parameters
stream_type = 2             # 0 = TCP, 1 = UDP, 2 = RTSP
stream_port = 5000          # set video streaming port number
udp_ip_addr = "10.42.0.52"  # IP address of the client for UDP streaming

# Set displayed preview image size (must be less than screen size to allow for the menu!!)
# Optomised for Pi 7" v1 screen

preview_width  = 880
preview_height = 580
fullscreen     = 0   # set to 1 for FULLSCREEN
frame          = 0   # set to 0 for NO frame 
FUP            = 21  # Pi v3 camera Focus UP GPIO button
FDN            = 16  # Pi v3 camera Focus DN GPIO button
sw_ir          = 26  # Waveshare IR Filter switch GPIO
STR            = 12  # external GPIO trigger for capture

# set default values (see limits below)
camera      = 0    # choose camera to use, usually 0 unless using a Pi5 or multiswitcher
mode        = 1    # set camera mode ['manual','normal','sport'] 
speed       = 16   # position in shutters list (16 = 1/125th)
gain        = 0    # set gain , 0 = AUTO
brightness  = 0    # set camera brightness
contrast    = 70   # set camera contrast 
ev          = 0    # eV correction 
blue        = 12   # blue balance 
red         = 15   # red balance 
extn        = 0    # still file type  (0 = jpg), see extns below
vlen        = 10   # video length in seconds
fps         = 15   # video fps - Réduit pour IMX585
vformat     = 10   # set video format (10 = 1920x1080), see vwidths & vheights below
codec       = 0    # set video codec  (0 = h264), see codecs below
tinterval   = 5.0   # time between timelapse shots in seconds
tshots      = 10   # number of timelapse shots
saturation  = 10   # picture colour saturation
meter       = 2    # metering mode (2 = average), see meters below
awb         = 1    # auto white balance mode, off, auto etc (1 = auto), see awbs below
sharpness   = 15   # set sharpness level
denoise     = 1    # set denoise level, see denoises below
quality     = 93   # set quality level
profile     = 0    # set h264 profile, see h264profiles below
level       = 0    # set h264 level
histogram   = 5    # OFF = 0, 1 = red, 2 = green, 3 = blue, 4 = luminance, 5 = ALL
histarea    = 50   # set histogram area size
v3_f_mode   = 0    # v3 focus mode,  see v3_f_modes below
v3_f_range  = 0    # v3 focus range, see v3_f_ranges below
v3_f_speed  = 0    # v3 focus speed, see v3_f_speeds below
IRF         = 0    # Waveshare imx290-83 IR filter, 1 = ON
str_cap     = 0    # 0 = STILL, see strs below
v3_hdr      = 0    # HDR (v3 camera or Pi5 ONLY), see v3_hdrs below
timet       = 2000 # -t setting when capturing STILLS
vflip       = 0    # set to 1 to vertically flip images
hflip       = 0    # set tp 1 tp horizontally flip images
# NOTE if you change any of the above defaults you need to delete the con_file and restart.

# default directories and files
pic         = "Pictures"
vid         = "Videos"
con_file    = "PiLCConfig104.txt"

# setup directories
Home_Files  = []
Home_Files.append(os.getlogin())
pic_dir     = "/media/admin/THKAILAR/Pictures/"
vid_dir     = "/home/" + Home_Files[0]+ "/" + vid + "/"
config_file = "/home/" + Home_Files[0]+ "/" + con_file

# inital parameters
prev_fps    = 5   # Réduit pour IMX585 
focus_fps   = 5  # Réduit pour IMX585
focus       = 700
foc_man     = 0
focus_mode  = 0
v3_focus    = 480
v3_hdr      = 0
vpreview    = 1
scientific  = 0
scientif    = 0
zx          = int(preview_width/2)
zy          = int(preview_height/2)
fxz         = 1
zoom        = 0
igw         = 2592
igh         = 1944
zwidth      = igw 
zheight     = igh
buttonFUP   = Button(FUP)
buttonFDN   = Button(FDN)
buttonSTR   = Button(STR)
led_sw_ir   = LED(sw_ir)
str_btn     = 0
lo_res      = 1
show_cmds   = 1  # Debug: afficher les commandes
v3_af       = 1
v5_af       = 1
menu        = 0
alt_dis     = 0
rotate      = 0
still       = 0
video       = 0
timelapse   = 0
stream      = 0
stretch_mode = 0  # Mode stretch astro pour le preview
stretch_p_low = 1    # Percentile bas pour stretch (0% à 0.1%, stocké x10 pour slider)
stretch_p_high = 995 # Percentile haut pour stretch (99.5% à 100%, stocké x10 pour slider)
stretch_factor = 100 # Facteur de stretch (1 à 30, stocké x10 pour slider)
stretch_preset = 0   # Préréglage stretch: 0=OFF, 1=AUTO, 2=Nébuleuse, 3=Galaxie
fwhm_history = deque(maxlen=240)
fwhm_times = deque(maxlen=240)
fwhm_start_time = 0
fwhm_fig = None
fwhm_ax = None
hfr_history = deque(maxlen=240)
hfr_times = deque(maxlen=240)
hfr_start_time = 0
hfr_fig = None
hfr_ax = None
focus_history = deque(maxlen=240)
focus_times = deque(maxlen=240)
focus_start_time = 0
focus_fig = None
focus_ax = None
mjpeg_extractor = None  # Instance de l'extracteur MJPEG
p = None  # Subprocess rpicam-vid (None en mode Picamera2)

# Picamera2 variables
picam2 = None  # Instance Picamera2
use_picamera2 = True  # Flag pour activer Picamera2 (False = utiliser rpicam-vid)

# Live Stack variables
livestack = None  # Instance RPiCameraLiveStack
livestack_active = False  # Mode Live Stack actif

# Live Stack parameters
ls_preview_refresh = 5  # Rafraîchir preview toutes les N images (1-10)
ls_alignment_mode = 2  # 0=OFF/none, 1=translation, 2=rotation, 3=affine
ls_alignment_modes = ["OFF", "translation", "rotation", "affine"]
ls_enable_qc = 1  # Contrôle qualité activé (0=OFF, 1=ON)
ls_max_fwhm = 170  # FWHM max x10 (0=OFF, 100-250 = 10.0-25.0)
ls_min_sharpness = 70  # Netteté min x1000 (0=OFF, 30-150 = 0.030-0.150)
ls_max_drift = 2500  # Dérive max en pixels (0=OFF, 500-5000)
ls_min_stars = 10  # Nombre min d'étoiles (0=OFF, 1-20)

# set button sizes
bw = int(preview_width/5.66)
bh = int(preview_height/10)
ft = int(preview_width/46)
fv = int(preview_width/46)

if tinterval > 0:
    tduration  = tshots * tinterval
else:
    tduration = 5

dis_height = preview_height
dis_width  = preview_width
    
# data
cameras      = [  '', 'Pi v1', 'Pi v2', 'Pi v3', 'Pi HQ','Ard 16MP','Hawkeye', 'Pi GS','Owlsight',"imx290",'imx585','imx293','imx294','imx283','imx500','ov9281']
camids       = [  '','ov5647','imx219','imx708','imx477',  'imx519', 'arduca','imx296',  'ov64a4','imx290','imx585','imx293','imx294','imx283','imx500','ov9281']
x_sens       = [   0,    2592,    3280,    4608,    4056,      4656,     9152,    1456,      9248,    1920,    3856,    3856,    4168,    5472,    4056,    1280]
y_sens       = [   0,    1944,    2464,    2592,    3040,      3496,     6944,    1088,      6944,    1080,    2180,    2180,    2824,    3648,    3040,     800]
max_gains    = [  64,     255,      40,      64,      88,        64,       64,      64,        64,      64,    3000,      64,      64,      64,      64,      64]  # IMX585: max 3000 avec courbe non-linéaire
max_shutters = [ 100,       1,      11,     112,     650,       200,      435,      15,       435,     100,     163,     100,     100,     100,     100,     100]
max_vfs      = [  10,      15,      16,      21,      20,        15,       22,       7,        22,      10,      18,      18,      18,      23,      20,       3]
modes        = ['manual','normal','sport']
extns        = ['jpg','png','bmp','rgb','yuv420','raw']
extns2       = ['jpg','png','bmp','data','data','dng']
vwidths      = [640,720,800,1280,1280,1296,1332,1456,1536,1640,1920,1928,2028,2028,2304,2592,3280,3840,3856,4032,4056,4608,4656,5472,8000,9152,9248]
vheights     = [480,540,600, 720, 960, 972, 990,1088, 864,1232,1080,1090,1080,1520,1296,1944,2464,2160,2180,3024,3040,2592,3496,3648,6000,6944,6944]
v_max_fps    = [200,120, 40,  40,  40,  30,  60,  30,  30,  30,  30,  30,  50,  40,  25,  20,  20,  20,  20,  20,  10,  20,  20,  20,  20,  20,  20]
v3_max_fps   = [200,120,125, 120, 120, 120, 120, 120, 120, 100, 100,  50, 100,  56,  56,  20,  20,  20,  20,  20,  15,  20,  20,  20  ,20,  20,  20]
v9_max_fps   = [240, 200, 150, 120, 100, 100, 80, 60, 60, 60, 60]
v15_max_fps  = [240,200,200, 130]
zwidths      = [640,800,1280,2592,3280,4056,4656,9152]
zheights     = [480,600, 960,1944,2464,3040,3496,6944]
zfs          = [1, 0.5, 0.333333, 0.25, 0.2, 0.166666]  # Zoom: 1x, 2x, 3x, 4x, 5x, 6x
# Labels de résolution pour le slider de zoom
zoom_res_labels = {
    0: "",
    1: "1920x1080",  # 2x zoom
    2: "1280x720",   # 3x zoom
    3: "DISABLED",   # 3x désactivé
    4: "640x480",    # 5x zoom
    5: "640x368"     # 6x zoom
}

# FPS optimaux pour chaque niveau de zoom (ROI permet des FPS plus élevés)
# Format: {zoom_level: (fps_standard, fps_v3, fps_v9, fps_imx585)}
zoom_optimal_fps = {
    1: (60, 100, 120, 30),   # 1920x1080 - 2x zoom
    2: (120, 120, 150, 60),  # 1280x720  - 3x zoom
    4: (200, 200, 240, 120), # 640x480   - 5x zoom
    5: (200, 200, 240, 120)  # 640x368   - 6x zoom
}

def sync_video_resolution_with_zoom():
    """
    Synchronise vwidth, vheight, vformat ET fps avec la résolution du zoom actuel.
    Appelée automatiquement quand le zoom change pour éviter les incohérences.
    Optimise aussi les FPS pour tirer parti des résolutions ROI plus petites.
    """
    global zoom, vwidth, vheight, vformat, vwidths, vheights, fps, video_limits, Pi_Cam

    # Résolutions correspondant aux niveaux de zoom
    zoom_resolutions = {
        1: (1920, 1080),  # 2x zoom
        2: (1280, 720),   # 3x zoom
        4: (640, 480),    # 5x zoom
        5: (640, 368)     # 6x zoom
    }

    # Si zoom actif, synchroniser avec la résolution du zoom
    if zoom > 0 and zoom in zoom_resolutions:
        target_width, target_height = zoom_resolutions[zoom]

        # Chercher l'index correspondant dans vwidths/vheights
        vformat_found = False
        for i in range(len(vwidths)):
            if vwidths[i] == target_width and vheights[i] == target_height:
                vformat = i
                vwidth = target_width
                vheight = target_height
                vformat_found = True
                break

        # Si pas trouvé dans la liste, mettre à jour directement vwidth/vheight
        if not vformat_found:
            vwidth = target_width
            vheight = target_height
            # vformat reste inchangé

        # Optimiser les FPS pour le zoom (ROI permet des FPS plus élevés)
        if zoom in zoom_optimal_fps:
            fps_standard, fps_v3, fps_v9, fps_imx585 = zoom_optimal_fps[zoom]

            # Choisir les FPS selon le type de caméra
            if Pi_Cam == 3:
                optimal_fps = fps_v3
            elif Pi_Cam == 9:
                optimal_fps = fps_v9
            elif Pi_Cam == 10:  # IMX585
                optimal_fps = fps_imx585
            elif Pi_Cam == 15:
                optimal_fps = fps_v9  # OV9281 utilise les mêmes que IMX290
            else:
                optimal_fps = fps_standard

            # Sauvegarder les FPS d'origine si première activation du zoom
            if not hasattr(sync_video_resolution_with_zoom, 'fps_backup'):
                sync_video_resolution_with_zoom.fps_backup = fps
                sync_video_resolution_with_zoom.vfps_backup = video_limits[5]

            # Mettre à jour fps et la limite max pour profiter du ROI
            # IMPORTANT: Ces variables sont déclarées global en haut de la fonction
            globals()['fps'] = optimal_fps  # Utiliser les FPS optimaux
            video_limits[5] = optimal_fps  # Nouvelle limite max
    else:
        # Zoom désactivé : restaurer les FPS d'origine si sauvegardés
        if hasattr(sync_video_resolution_with_zoom, 'fps_backup'):
            globals()['fps'] = sync_video_resolution_with_zoom.fps_backup
            video_limits[5] = sync_video_resolution_with_zoom.vfps_backup
            # Nettoyer les sauvegardes
            delattr(sync_video_resolution_with_zoom, 'fps_backup')
            delattr(sync_video_resolution_with_zoom, 'vfps_backup')

shutters     = [-4000,-2000,-1600,-1250,-1000,-800,-640,-500,-400,-320,-288,-250,-240,-200,-160,-144,-125,-120,-100,-96,-80,-60,-50,-48,-40,-30,-25,
                -20,-15,-13,-10,-8,-6,-5,-4,-3,0.4,0.5,0.6,0.8,1,1.1,1.2,2,3,4,5,6,7,8,9,10,11,15,20,25,30,40,50,60,75,100,112,120,150,200,220,230,
                239,435,500,600,650,660,670]
codecs       = ['h264','mjpeg','yuv420','raw','ser']
codecs2      = ['h264','mjpeg','data','raw','ser']
h264profiles = ['baseline 4','baseline 4.1','baseline 4.2','main 4','main 4.1','main 4.2','high 4','high 4.1','high 4.2']
meters       = ['centre','spot','average']
awbs         = ['off','auto','incandescent','tungsten','fluorescent','indoor','daylight','cloudy']
denoises     = ['off','cdn_off','cdn_fast','cdn_hq']
v3_f_modes   = ['auto','manual','continuous']
v3_f_ranges  = ['normal','macro','full']
v3_f_speeds  = ['normal','fast']
histograms   = ["OFF","Red","Green","Blue","Lum","ALL"]
strs         = ["Still","Video","Stream","Timelapse"]
stretch_presets = ['OFF','AUTO','Nebuleuse','Galaxie']
v3_hdrs      = ["off","single-exp","auto","sensor"]

#check linux version.
if os.path.exists ("/run/shm/lv.txt"): 
    os.remove("/run/shm/lv.txt")
os.system("cat /etc/os-release >> /run/shm/lv.txt")
with open("/run/shm/lv.txt", "r") as file:
    line = file.readline()
    while line:
       line = file.readline()
       if line[0:16] == "VERSION_CODENAME":
           lver = line
lvers = lver.split("=")
lver = lvers[1][0:6]
print(lver)

#check Pi model.
Pi = -1
if os.path.exists ('/run/shm/md.txt'): 
    os.remove("/run/shm/md.txt")
os.system("cat /proc/cpuinfo >> /run/shm/md.txt")
with open("/run/shm/md.txt", "r") as file:
    line = file.readline()
    while line:
       line = file.readline()
       if line[0:5] == "Model":
           model = line
mod = model.split(" ")
if mod[3] == "Compute":
    Pi = int(mod[5][0:1])
elif mod[3] == "Zero":
    Pi = 0
else:
    Pi = int(mod[3])
print("Pi:",Pi)
# Note : MP4 sur Pi 5 a des métadonnées de framerate incorrectes (bug rpicam-vid)
# mais reste utilisable car les frames sont encodées correctement
if Pi == 5:
    codecs.append('mp4')
    codecs2.append('mp4')

still_limits = ['mode',0,len(modes)-1,'speed',0,len(shutters)-1,'gain',0,30,'brightness',-100,100,'contrast',0,200,'ev',-10,10,'blue',1,80,'sharpness',0,30,
                'denoise',0,len(denoises)-1,'quality',0,100,'red',1,80,'extn',0,len(extns)-1,'saturation',0,20,'meter',0,len(meters)-1,'awb',0,len(awbs)-1,
                'histogram',0,len(histograms)-1,'v3_f_speed',0,len(v3_f_speeds)-1]
video_limits = ['vlen',0,3600,'fps',1,40,'v5_focus',10,2500,'vformat',0,7,'0',0,0,'zoom',0,5,'Focus',0,1,'tduration',1,86400,'tinterval',0.01,10,'tshots',1,999,
                'flicker',0,3,'codec',0,len(codecs)-1,'profile',0,len(h264profiles)-1,'v3_focus',10,2000,'histarea',10,300,'v3_f_range',0,len(v3_f_ranges)-1,
                'str_cap',0,len(strs)-1,'v6_focus',10,1020,'stretch_p_low',0,1,'stretch_p_high',995,1000,'stretch_factor',10,300,'stretch_preset',0,3]

livestack_limits = ['ls_preview_refresh',1,10,'ls_alignment_mode',0,len(ls_alignment_modes)-1,'ls_enable_qc',0,1,'ls_max_fwhm',0,250,'ls_min_sharpness',0,150,'ls_max_drift',0,5000,'ls_min_stars',0,20]

# check config_file exists, if not then write default values
titles = ['mode','speed','gain','brightness','contrast','frame','red','blue','ev','vlen','fps','vformat','codec','tinterval','tshots','extn','zx','zy','zoom','saturation',
          'meter','awb','sharpness','denoise','quality','profile','level','histogram','histarea','v3_f_speed','v3_f_range','rotate','IRF','str_cap','v3_hdr','timet','vflip','hflip',
          'stretch_p_low','stretch_p_high','stretch_factor','stretch_preset','ls_preview_refresh','ls_alignment_mode','ls_enable_qc','ls_max_fwhm','ls_min_sharpness','ls_max_drift','ls_min_stars']
points = [mode,speed,gain,brightness,contrast,frame,red,blue,ev,vlen,fps,vformat,codec,tinterval,tshots,extn,zx,zy,zoom,saturation,
          meter,awb,sharpness,denoise,quality,profile,level,histogram,histarea,v3_f_speed,v3_f_range,rotate,IRF,str_cap,v3_hdr,timet,vflip,hflip,
          stretch_p_low,stretch_p_high,stretch_factor,stretch_preset,ls_preview_refresh,ls_alignment_mode,ls_enable_qc,ls_max_fwhm,ls_min_sharpness,ls_max_drift,ls_min_stars]
if not os.path.exists(config_file):
    with open(config_file, 'w') as f:
        for item in range(0,len(titles)):
            f.write( titles[item] + " : " + str(points[item]) + "\n")

# read config_file
config = []
with open(config_file, "r") as file:
   line = file.readline()
   while line:
       line = line.strip()
       item = line.split(" : ")
       config.append(item[1])
       line = file.readline()
# Convertir d'abord en float, puis en int sauf pour tinterval (index 13)
config = list(map(float,config))
for i in range(len(config)):
    if i != 13:  # Garder tinterval comme float
        config[i] = int(config[i])

mode        = config[0]
speed       = config[1]
gain        = config[2]
brightness  = config[3]
contrast    = config[4]
red         = config[6]
blue        = config[7]
ev          = config[8]
vlen        = config[9]
fps         = config[10]
vformat     = config[11]
codec       = config[12]
tinterval   = config[13]
tshots      = config[14]
extn        = config[15]
zx          = config[16]
zy          = config[17]
zoom        = 0
saturation  = config[19]
meter       = config[20]
awb         = config[21]
sharpness   = config[22]
denoise     = config[23]
quality     = config[24]
profile     = config[25]
level       = config[26]
histogram   = config[27]
histarea    = config[28]
v3_f_speed  = config[29]
v3_f_range  = config[30]
rotate      = config[31]
IRF         = config[32]
str_cap     = config[33]
v3_hdr      = config[34]
timet       = config[35]
vflip       = config[36]
hflip       = config[37]

# Ajouter les nouveaux paramètres stretch si le fichier de config est ancien
if len(config) <= 38:
    config.append(1)     # stretch_p_low par défaut (0.1%)
if len(config) <= 39:
    config.append(995)   # stretch_p_high par défaut
if len(config) <= 40:
    config.append(100)   # stretch_factor par défaut
if len(config) <= 41:
    config.append(0)     # stretch_preset par défaut (OFF)

stretch_p_low    = config[38]
stretch_p_high   = config[39]
stretch_factor   = config[40]
stretch_preset   = config[41]

# Ajouter les paramètres livestack si le fichier de config est ancien
if len(config) <= 42:
    config.append(5)     # ls_preview_refresh par défaut
if len(config) <= 43:
    config.append(2)     # ls_alignment_mode par défaut (rotation)
if len(config) <= 44:
    config.append(1)     # ls_enable_qc par défaut (activé)
if len(config) <= 45:
    config.append(170)   # ls_max_fwhm par défaut (17.0)
if len(config) <= 46:
    config.append(70)    # ls_min_sharpness par défaut (0.070)
if len(config) <= 47:
    config.append(2500)  # ls_max_drift par défaut
if len(config) <= 48:
    config.append(10)    # ls_min_stars par défaut

ls_preview_refresh = config[42]
ls_alignment_mode  = config[43]
ls_enable_qc       = config[44]
ls_max_fwhm        = config[45]
ls_min_sharpness   = config[46]
ls_max_drift       = config[47]
ls_min_stars       = config[48]

if codec > len(codecs)-1:
    codec = 0

def setmaxvformat():
    # set max video format
    global codec,Pi_Cam,configtxt,max_vformat,max_vfs
    if codec > 0 and (Pi_Cam == 5 or Pi_Cam == 6 or Pi_Cam == 8) and ("dtoverlay=vc4-kms-v3d,cma-512" in configtxt): # Arducam IMX519 16MP or 64MP
        max_vformat = max_vfs[6]
    elif codec > 0 and (Pi_Cam == 5 or Pi_Cam == 6 or Pi_Cam == 8): # Arducam IMX519 16MP or 64MP
        max_vformat = max_vfs[5]
    elif codec > 0:
        max_vformat = max_vfs[Pi_Cam]
    elif Pi_Cam == 7 or Pi_Cam == 15:  # Pi GS or ov9281
        max_vformat = max_vfs[Pi_Cam]
    else:
        max_vformat = max_vfs[0]
    if Pi_Cam == 4 and codec == 0:
        max_vformat = 12
    
def slider_to_gain_nonlinear(slider_value, max_gain):
    """
    Convertit position slider linéaire (0-max_gain) en gain non-linéaire (1-max_gain)
    70% de la plage → gain 1-1000 (précis)
    30% de la plage → gain 1000-max_gain (moins précis)
    """
    if max_gain <= 1000:
        # Pour caméras avec gain max <= 1000, rester linéaire
        return max(1, slider_value)

    # Normaliser la position (0.0 à 1.0)
    slider_pos = slider_value / max_gain

    if slider_pos <= 0.7:
        # 70% du slider → gain 1 à 1000 (précis)
        return int(1 + (slider_pos / 0.7) * 999)
    else:
        # 30% du slider → gain 1000 à max_gain (moins précis)
        return int(1000 + ((slider_pos - 0.7) / 0.3) * (max_gain - 1000))

def gain_to_slider_nonlinear(gain_value, max_gain):
    """
    Inverse : convertit gain (1-max_gain) en position slider linéaire (0-max_gain)
    Pour ajustements +/- des boutons
    """
    if max_gain <= 1000:
        return gain_value

    if gain_value <= 1000:
        # gain 1-1000 → 0-70% du slider
        return int(((gain_value - 1) / 999) * 0.7 * max_gain)
    else:
        # gain 1000-max_gain → 70-100% du slider
        return int((0.7 + ((gain_value - 1000) / (max_gain - 1000)) * 0.3) * max_gain)

def Camera_Version():
    # Check for Pi Camera version
    global lver,v3_af,camera,vwidths2,vheights2,configtxt,mode,mag,max_gain,max_shutter,Pi_Cam,max_camera,same_cams,x_sens,y_sens,igw,igh
    global cam0,cam1,cam2,cam3,max_gains,max_shutters,scientif,max_vformat,vformat,vwidth,vheight,vfps,sspeed,tduration,video_limits,lo_res
    global speed,shutter,max_vf_7,max_vf_6,max_vf_5,max_vf_4,max_vf_3,max_vf_2,max_vf_1,max_vf_4a,max_vf_0,max_vf_8,max_vf_9,IRF,foc_sub3
    global foc_sub5,v3_hdr,windowSurfaceObj,cam1
    # DETERMINE NUMBER OF CAMERAS (FOR ARDUCAM MULITPLEXER or Pi5)
    if os.path.exists('rpicams.txt'):
        os.rename('rpicams.txt', 'oldrpicams.txt')
    if lver != "bookwo" and lver != "trixie":
        os.system("libcamera-vid --list-cameras >> rpicams.txt")
    else:
        os.system("rpicam-vid --list-cameras >> rpicams.txt")
    time.sleep(0.5)
    # read rpicams.txt file
    camstxt = []
    with open("rpicams.txt", "r") as file:
        line = file.readline()
        while line:
            camstxt.append(line.strip())
            line = file.readline()
    max_camera = 0
    same_cams  = 0
    lo_res = 1
    cam0 = "0"
    cam1 = "1"
    cam2 = "2"
    cam3 = "3"
    Pi_Cam = -1
    vwidths2  = []
    vheights2 = []
    vfps2 = []
    for x in range(0,len(camstxt)):
        # Determine camera models
        if camstxt[x][0:4] == "0 : ":
            cam0 = camstxt[x][4:10]
        if cam0 != "0" and cam1 == "1" and camera == 0:
            # determine native formats
            forms = camstxt[x].split(" ")
            for q in range(0,len(forms)):
                if "x" in forms[q] and "/" not in forms[q] and "m" not in forms[q] and "[" not in forms[q]:
                    qwidth,qheight = forms[q].split("x")
                    vwidths2.append(int(qwidth))
                    vheights2.append(int(qheight))
                if forms[q][0:1] == "[" and "x" not in forms[q]:
                    vfps2.append(int(float(forms[q][1:4])))
        if camstxt[x][0:4] == "1 : ":
            cam1 = camstxt[x][4:10]
        if cam0 != "0" and cam1 != "1" and camera == 1:
              # determine native formats
              forms = camstxt[x].split(" ")
              for q in range(0,len(forms)):
               if "x" in forms[q] and "/" not in forms[q] and "m" not in forms[q] and "[" not in forms[q]:
                  qwidth,qheight = forms[q].split("x")
                  vwidths2.append(int(qwidth))
                  vheights2.append(int(qheight))
               if forms[q][0:1] == "[" and "x" not in forms[q]:
                    vfps2.append(int(float(forms[q][1:4])))
        if camstxt[x][0:4] == "2 : ":
            cam2 = camstxt[x][4:10]
        if camstxt[x][0:4] == "3 : ":
            cam3 = camstxt[x][4:10]
        # Determine MAXIMUM number of cameras available 
        if camstxt[x][0:4]   == "3 : " and max_camera < 3:
            max_camera = 3
        elif camstxt[x][0:4] == "2 : " and max_camera < 2:
            max_camera = 2
        elif camstxt[x][0:4] == "1 : " and max_camera < 1:
            max_camera = 1
        pic = 0
        Pi_Cam = -1
        for x in range(0,len(camids)):
            if camera == 0:
                if cam0 == camids[x]:
                    Pi_Cam = x
                    pic = 1
            elif camera == 1:
                if cam1 == camids[x]:
                    Pi_Cam = x
                    pic = 1
            elif camera == 2:
                if cam2 == camids[x]:
                    Pi_Cam = x
                    pic = 1
            elif camera == 3:
                if cam3 == camids[x]:
                    Pi_Cam = x
                    pic = 1
            if pic == 1:
                max_shutter = max_shutters[Pi_Cam]
                max_gain = max_gains[Pi_Cam]
                mag = int(max_gain/4)
                still_limits[8] = max_gain
    if Pi_Cam != -1:
        print("Camera:",cameras[Pi_Cam])
    elif cam0 != "0" and camera == 0:
        Pi_Cam      = 0
        cameras[0]  = cam0
        camids[0]   = cam0[0:6]
        print("Camera:",cameras[Pi_Cam])
        max_shutter = max_shutters[Pi_Cam]
        max_gain    = max_gains[Pi_Cam]
        mag         = int(max_gain/4)
        still_limits[8] = max_gain
        x_sens[0] = vwidths2[len(vwidths2)-1]
        y_sens[0] = vheights2[len(vheights2)-1]
    elif cam1 != "1" and camera == 1:
        Pi_Cam      = 0
        cameras[0]  = cam1
        camids[0]   = cam1[0:6]
        print("Camera:",cameras[Pi_Cam])
        max_shutter = max_shutters[Pi_Cam]
        max_gain    = max_gains[Pi_Cam]
        mag         = int(max_gain/4)
        still_limits[8] = max_gain
        x_sens[0] = vwidths2[len(vwidths2)-1]
        y_sens[0] = vheights2[len(vheights2)-1]
    else:
        print("No Camera Found")
        pygame.display.quit()
        sys.exit()

    igw = x_sens[Pi_Cam]
    igh = y_sens[Pi_Cam]

    if igw/igh > 1.5:
        pygame.draw.rect(windowSurfaceObj,(0,0,0),Rect(0,int(preview_height * .75),preview_width,int(preview_height *.25) ))

            
    if max_camera == 1 and cam0 == cam1:
        same_cams = 1
    configtxt = []
    if Pi_Cam == 9:
        if IRF == 0:
            led_sw_ir.off()
        else:
            led_sw_ir.on()
    if Pi_Cam != 3 and v3_hdr > 0:
        v3_hdr = 1
    if Pi_Cam == 3 or Pi_Cam == 5 or Pi_Cam == 6 or Pi_Cam == 8:
        # read /boot/config.txt file
        if lver != "bookwo" and lver != "trixie":
          with open("/boot/config.txt", "r") as file:
            line = file.readline()
            while line:
                configtxt.append(line.strip())
                line = file.readline()
        else:
          with open("/boot/firmware/config.txt", "r") as file:
            line = file.readline()
            while line:
                configtxt.append(line.strip())
                line = file.readline()
        # determine /dev/v4l-subdevX for Pi v3 and Arducam 16/64MP cameras
        foc_sub3 = -1
        foc_sub5 = -1
        if Pi_Cam == 3 or Pi_Cam == 5 or Pi_Cam == 6 or Pi_Cam == 8: # AF cameras
          for x in range(0,10):
            if os.path.exists("ctrls1.txt"):
                os.remove("ctrls1.txt")
            os.system("v4l2-ctl -d /dev/v4l-subdev" + str(x) + " --list-ctrls >> ctrls1.txt")
            time.sleep(0.25)
            ctrlstxt = []
            with open("ctrls1.txt", "r") as file:
                line = file.readline()
                while line:
                    ctrlstxt.append(line.strip())
                    line = file.readline()
            for a in range(0,len(ctrlstxt)):
                if ctrlstxt[a][0:45] == "exposure 0x00980911 (int)    : min=9 max=7079" and foc_sub5 == -1 and Pi_Cam == 6: # arducam 64mp hawkeye
                    foc_sub5 = x + 1
                elif ctrlstxt[a][0:51] == "focus_absolute 0x009a090a (int)    : min=0 max=4095" and foc_sub5 == -1 and Pi_Cam == 5: # arducam 16mp
                    foc_sub5 = x
                elif ctrlstxt[a][0:45] == "exposure 0x00980911 (int)    : min=1 max=2602" and Pi_Cam == 3: # pi v3
                    foc_sub3 = x + 1
                elif ctrlstxt[a][0:37] == "exposure 0x00980911 (int)    : min=16" and Pi_Cam == 8: # arducam owlsight 64mp
                    foc_sub3 = x + 1
    if cam0 != "0" and cam1 != "1":
        pygame.display.set_caption('RPiCamera:  v' + str(version) + "  Pi: " + str(Pi) + "  Camera: "  + cameras[Pi_Cam] + " : " + str(camera))
    else:
        pygame.display.set_caption('RPiCamera:  v' + str(version) + "  Pi: " + str(Pi) + "  Camera: "  + cameras[Pi_Cam] )

    if (Pi_Cam == 5 or Pi_Cam == 6 or Pi_Cam == 8) and ("dtoverlay=vc4-kms-v3d,cma-512" in configtxt):
        lo_res = 0
    # set max video format
    setmaxvformat()
    if vformat > max_vformat:
        vformat = max_vformat
    if Pi_Cam == 4:  # Pi HQ
        if codec == 0:
            max_vformat = 12
        if ((Pi != 5 and os.path.exists('/usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json')) or (Pi == 5 and os.path.exists('/usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json'))):
            scientif = 1
        else:
            scientif = 0
    vwidth    = vwidths[vformat]
    vheight   = vheights[vformat]
    # set max fps
    if Pi_Cam == 3:
        vfps = v3_max_fps[vformat]
    elif Pi_Cam == 9:
        vfps = v9_max_fps[vformat]
    elif Pi_Cam == 15:
        vfps = v15_max_fps[vformat]
    else:
        vfps = v_max_fps[vformat]
    video_limits[5] = vfps
    if tinterval > 0:
        tduration = tinterval * tshots
    else:
        tduration = 5
    shutter = shutters[speed]
    if shutter < 0:
        shutter = abs(1/shutter)
    sspeed = int(shutter * 1000000)
    if (shutter * 1000000) - int(shutter * 1000000) > 0.5:
        sspeed +=1
    # determine max speed for camera
    max_speed = 0
    while max_shutter > shutters[max_speed]:
        max_speed +=1
    if speed > max_speed:
        speed = max_speed
        shutter = shutters[speed]
        if shutter < 0:
            shutter = abs(1/shutter)
        sspeed = int(shutter * 1000000)
        if mode == 0:
            if shutters[speed] < 0:
                text(0,2,3,1,1,"1/" + str(abs(shutters[speed])),fv,10)
            else:
                text(0,2,3,1,1,str(shutters[speed]),fv,10)
        else:
            if shutters[speed] < 0:
                text(0,2,0,1,1,"1/" + str(abs(shutters[speed])),fv,10)
            else:
                text(0,2,0,1,1,str(shutters[speed]),fv,10)

pygame.init()

if frame == 1:
    if fullscreen == 1:
        windowSurfaceObj = pygame.display.set_mode((preview_width + bw,dis_height),pygame.FULLSCREEN, 24)
    elif fullscreen == 0:
        windowSurfaceObj = pygame.display.set_mode((preview_width + bw,dis_height),0,24)

else:
    windowSurfaceObj = pygame.display.set_mode((preview_width + bw,dis_height), pygame.NOFRAME,24)

Camera_Version()

global greyColor, redColor, greenColor, blueColor, dgryColor, lgrnColor, blackColor, whiteColor, purpleColor, yellowColor,lpurColor,lyelColor
# Palette de couleurs sobres et professionnelles
bredColor =   pygame.Color(180,  60,  60)  # Rouge atténué
lgrnColor =   pygame.Color(140, 160, 145)  # Vert grisâtre doux
lpurColor =   pygame.Color(150, 140, 160)  # Mauve discret
lyelColor =   pygame.Color(160, 155, 135)  # Beige/taupe
blackColor =  pygame.Color( 20,  20,  25)  # Noir légèrement bleuté
whiteColor =  pygame.Color(230, 230, 235)  # Blanc cassé
greyColor =   pygame.Color(110, 115, 120)  # Gris moyen neutre
dgryColor =   pygame.Color( 45,  48,  52)  # Gris foncé moderne
greenColor =  pygame.Color( 80, 160,  90)  # Vert modéré
purpleColor = pygame.Color(140, 100, 150)  # Violet sobre
yellowColor = pygame.Color(200, 180,  80)  # Jaune moutarde
blueColor =   pygame.Color(200, 100,  40)  # Orange foncé
redColor =    pygame.Color(160,  70,  70)  # Rouge brique

def button(col,row,bkgnd_Color,border_Color):
    global preview_width,bw,bh,alt_dis,preview_height,menu
    colors = [greyColor, dgryColor,yellowColor,purpleColor,greenColor,whiteColor,lgrnColor,lpurColor,lyelColor,blueColor]
    Color = colors[bkgnd_Color]
    bx = preview_width + (col * bw)
    by = row * bh
    # but = pygame.image.load("button.jpg")  # Image des boutons désactivée
    pygame.draw.rect(windowSurfaceObj,Color,Rect(bx+1,by,bw-2,bh))
    pygame.draw.line(windowSurfaceObj,whiteColor,(bx,by),(bx,by+bh-1),2)
    pygame.draw.line(windowSurfaceObj,whiteColor,(bx,by),(bx+bw-1,by),1)
    pygame.draw.line(windowSurfaceObj,dgryColor,(bx,by+bh-1),(bx+bw-1,by+bh-1),1)
    pygame.draw.line(windowSurfaceObj,dgryColor,(bx+bw-2,by),(bx+bw-2,by+bh),2)
    # Images des boutons (Still, Video, Timelapse) désactivées pour un look plus épuré
    # if menu == 0 and row < 3:
    #     windowSurfaceObj.blit(but, (preview_width + 2,by + 2))
    pygame.display.update()

def text(col,row,fColor,top,upd,msg,fsize,bkgnd_Color):
    global bh,preview_width,fv,tduration,menu
    colors =  [dgryColor, greenColor, yellowColor, redColor, purpleColor, blueColor, whiteColor, greyColor, blackColor, purpleColor,lgrnColor,lpurColor,lyelColor]
    Color  =  colors[fColor]
    bColor =  colors[bkgnd_Color]
    bx = preview_width + (col * bw)
    by = row * bh
    if menu == 0 and row < 3:
        by +=10
    # Polices modernes en ordre de préférence
    modern_fonts = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',           # Police sans-serif moderne et lisible
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',  # Alternative clean
        '/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf',  # Version condensée
        '/usr/share/fonts/truetype/freefont/FreeSerif.ttf'           # Fallback original
    ]
    
    fontObj = None
    for font_path in modern_fonts:
        if os.path.exists(font_path):
            fontObj = pygame.font.Font(font_path, int(fsize))
            break
    
    if fontObj is None:
        fontObj = pygame.font.Font(None, int(fsize))  # Police système par défaut
    msgSurfaceObj = fontObj.render(msg, False, Color)
    msgRectobj = msgSurfaceObj.get_rect()
    if top == 0:
        if menu != 0:
             pygame.draw.rect(windowSurfaceObj,bColor,Rect(bx+2,by+int(bh/3),bw-4,int(bh/3)))
        msgRectobj.topleft = (bx + 5, by + int(bh/3)-int(preview_width/640))
    elif msg == "Config":
        if menu != 0:
            pygame.draw.rect(windowSurfaceObj,bColor,Rect(bx+2,by+int(bh/1.5),int(bw/2)-1,int(bh/3)-1))
        msgRectobj.topleft = (bx+5,  by + int(bh/1.5)-1)
    elif top == 1:
        if menu != 0 :
            pygame.draw.rect(windowSurfaceObj,bColor,Rect(bx+20,by+int(bh/1.5)-1,int(bw-21)-1,int(bh/3)))
        elif timelapse == 1:
            pygame.draw.rect(windowSurfaceObj,bColor,Rect(bx+20,by+int(bh/1.5)-1,int(bw-101),int(bh/5)))
        elif video == 1 or stream == 1:
            pygame.draw.rect(windowSurfaceObj,bColor,Rect(bx+20,by+int(bh/1.5)-1,int(bw-61)-1,int(bh/5)))
        msgRectobj.topleft = (bx + 20, by + int(bh/1.5)-int(preview_width/640)-1) 
    elif top == 2:
        if bkgnd_Color == 1:
            pygame.draw.rect(windowSurfaceObj,(0,0,0),Rect(0,row * fsize,preview_width,fv*2)) 
        msgRectobj.topleft = (0,row * fsize)
    windowSurfaceObj.blit(msgSurfaceObj, msgRectobj)
    if upd == 1 and top == 2:
        pygame.display.update(0,0,preview_width,fv*2)
    if upd == 1:
        pygame.display.update(bx, by, bw, bh)

def draw_bar(col,row,color,msg,value):
    global bw,bh,preview_width,still_limits,max_speed,v3_mag
    for f in range(0,len(still_limits)-1,3):
        if still_limits[f] == msg:
            pmin = still_limits[f+1]
            pmax = still_limits[f+2]
    if msg == "speed":
        pmax = max_speed

    # Pour le gain non-linéaire (IMX585), convertir gain → position slider
    display_value = value
    display_mag = mag
    if msg == "gain" and pmax > 1000 and value > 0:
        display_value = gain_to_slider_nonlinear(value, pmax)
        display_mag = gain_to_slider_nonlinear(mag, pmax)

    pygame.draw.rect(windowSurfaceObj,color,Rect(preview_width + col*bw,(row * bh) + 1,bw-2,int(bh/3)))
    if pmin > -1:
        j = display_value / (pmax - pmin)  * bw
        jag = display_mag / (pmax - pmin) * bw
    else:
        j = int(bw/2) + (display_value / (pmax - pmin)  * bw)
    j = min(j,bw-5)
    pygame.draw.rect(windowSurfaceObj,(80,140,90),Rect(int(preview_width + int(col*bw) + 2),int(row * bh)+1,int(j+1),int(bh/3)))
    if msg == "gain" and value > mag:
        pygame.draw.rect(windowSurfaceObj,(180,160,70),Rect(int(preview_width + int(col*bw) + 2 + jag),int(row * bh),int(j+1 - jag),int(bh/3)))
    pygame.draw.rect(windowSurfaceObj,(120,90,130),Rect(int(preview_width + int(col*bw) + j ),int(row * bh)+1,3,int(bh/3)))
    pygame.display.update()

def draw_Vbar(col,row,color,msg,value):
    global bw,bh,preview_width,video_limits,livestack_limits
    pmin = None
    pmax = None
    # Chercher d'abord dans video_limits
    for f in range(0,len(video_limits)-1,3):
        if video_limits[f] == msg:
            pmin = video_limits[f+1]
            pmax = video_limits[f+2]
            break
    # Si pas trouvé, chercher dans livestack_limits
    if pmin is None:
        for f in range(0,len(livestack_limits)-1,3):
            if livestack_limits[f] == msg:
                pmin = livestack_limits[f+1]
                pmax = livestack_limits[f+2]
                break
    if msg == "vformat":
        pmax = max_vformat
    if alt_dis == 0:
        pygame.draw.rect(windowSurfaceObj,color,Rect(preview_width + col*bw,(row * bh) +1,bw-2,int(bh/3)))
    else:
        if row < 8:
            if alt_dis == 1:
                 pygame.draw.rect(windowSurfaceObj,color,Rect(row*bw,preview_height + (bh*2),bw-1,int(bh/3)))
            else:
                 pygame.draw.rect(windowSurfaceObj,color,Rect(row*bw,int((preview_height *.75) + (bh*2)),bw-1,int(bh/3)))
        else:
            if alt_dis == 1:
                pygame.draw.rect(windowSurfaceObj,color,Rect((row-8)*bw,preview_height + (bh*3),bw-1,int(bh/3)))
            else:
                pygame.draw.rect(windowSurfaceObj,color,Rect((row-8)*bw,int((preview_height *.75) + (bh*3)),bw-1,int(bh/3)))
    if pmin > -1: 
        j = value / (pmax - pmin)  * bw
    else:
        j = int(bw/2) + (value / (pmax - pmin)  * bw)
    j = min(j,bw-5)
    pygame.draw.rect(windowSurfaceObj,(120,110,130),Rect(int(preview_width + (col*bw) + 2),int(row * bh)+1,int(j+1),int(bh/3)))
    pygame.draw.rect(windowSurfaceObj,(120,90,130),Rect(int(preview_width + (col*bw) + j ),int(row * bh)+1,3,int(bh/3)))
                
    pygame.display.update()
def calculate_fwhm(image_surface, center_x, center_y, area_size):
    """
    Calcule le FWHM pour mesurer la netteté
    Version corrigée qui ne verrouille pas la surface
    """
    try:
        # IMPORTANT : Utiliser array3d au lieu de pixels3d (ne verrouille pas)
        image_array = pygame.surfarray.array3d(image_surface)
        
        # Extraire la région d'intérêt
        y1 = max(0, center_y - area_size)
        y2 = min(image_array.shape[1], center_y + area_size)  # Note: shape[1] pour y
        x1 = max(0, center_x - area_size)
        x2 = min(image_array.shape[0], center_x + area_size)  # Note: shape[0] pour x
        
        # pygame.surfarray retourne (width, height, channels), donc on transpose
        roi = image_array[x1:x2, y1:y2, :]
        roi = np.transpose(roi, (1, 0, 2))  # Convertir en (height, width, channels)
        
        # Convertir en niveaux de gris
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = roi
        
        # Trouver le point le plus brillant dans la ROI
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
        
        # Profils horizontal et vertical passant par le point le plus brillant
        profile_h = gray[max_loc[1], :]
        profile_v = gray[:, max_loc[0]]
        
        # Calculer FWHM pour chaque profil
        fwhm_h = calculate_fwhm_1d(profile_h)
        fwhm_v = calculate_fwhm_1d(profile_v)
        
        # Retourner la moyenne
        if fwhm_h is not None and fwhm_v is not None:
            return (fwhm_h + fwhm_v) / 2.0
        elif fwhm_h is not None:
            return fwhm_h
        elif fwhm_v is not None:
            return fwhm_v
        else:
            return None
            
    except Exception as e:
        print(f"Erreur FWHM: {e}")
        return None

def calculate_fwhm_1d(profile):
    """Calcule le FWHM d'un profil 1D avec détection de pic améliorée"""
    try:
        max_val = np.max(profile)
        min_val = np.min(profile)
        
        # Vérifier qu'il y a un contraste suffisant (au moins 10% de différence)
        if max_val <= min_val or (max_val - min_val) / max_val < 0.1:
            return None
            
        half_max = min_val + (max_val - min_val) / 2.0
        above_half = profile >= half_max
        
        if not np.any(above_half):
            return None
        
        indices = np.where(above_half)[0]
        
        if len(indices) < 2:
            return None
        
        # Calculer FWHM (largeur à mi-hauteur)
        fwhm = indices[-1] - indices[0] + 1  # +1 pour inclure les deux bords
            
        return float(fwhm)
    except:
        return None

def get_fwhm_color(fwhm):
    """Retourne une couleur RGB selon la qualité du FWHM"""
    if fwhm is None:
        return (128, 128, 128)
    elif fwhm < 3:
        return (0, 255, 0)
    elif fwhm < 6:
        return (255, 255, 0)
    elif fwhm < 10:
        return (255, 165, 0)
    else:
        return (255, 0, 0)

def get_fwhm_quality_text(fwhm):
    """Retourne un texte de qualité selon le FWHM"""
    if fwhm is None:
        return "N/A"
    elif fwhm < 3:
        return "Excellente"
    elif fwhm < 6:
        return "Bonne"
    elif fwhm < 10:
        return "Moyenne"
    else:
        return "Mauvaise"

def init_fwhm_graph():
    """Initialise le graphique matplotlib pour le FWHM"""
    global fwhm_fig, fwhm_ax

    if fwhm_fig is None:
        fwhm_fig, fwhm_ax = plt.subplots(figsize=(6, 3), dpi=100)
        fwhm_fig.patch.set_facecolor('#1a1a1a')
        fwhm_ax.set_facecolor('#0a0a0a')

    return fwhm_fig, fwhm_ax

def init_hfr_graph():
    """Initialise le graphique matplotlib pour le HFR"""
    global hfr_fig, hfr_ax

    if hfr_fig is None:
        # Taille et style élégants pour meilleure lisibilité
        hfr_fig, hfr_ax = plt.subplots(figsize=(6, 3), dpi=100)
        hfr_fig.patch.set_facecolor('#1a1a1a')
        hfr_ax.set_facecolor('#0a0a0a')

    return hfr_fig, hfr_ax

def update_fwhm_graph(fwhm_val):
    """Met à jour le graphique FWHM et retourne une surface pygame"""
    global fwhm_history, fwhm_times, fwhm_start_time, fwhm_fig, fwhm_ax
    
    if fwhm_val is None:
        return None
    
    if fwhm_start_time == 0:
        fwhm_start_time = time.time()
    
    current_time = time.time() - fwhm_start_time
    fwhm_history.append(fwhm_val)
    fwhm_times.append(current_time)
    
    if len(fwhm_history) < 2:
        return None
    
    fig, ax = init_fwhm_graph()
    ax.clear()
    
    # Zones de qualité
    ax.axhspan(0, 3, alpha=0.2, color='green', linewidth=0)
    ax.axhspan(3, 6, alpha=0.2, color='yellow', linewidth=0)
    ax.axhspan(6, 10, alpha=0.2, color='orange', linewidth=0)
    ax.axhspan(10, max(list(fwhm_history) + [15]), alpha=0.2, color='red', linewidth=0)
    
    # Courbe FWHM avec couleurs
    times_list = list(fwhm_times)
    fwhm_list = list(fwhm_history)
    
    for i in range(len(fwhm_list) - 1):
        color = get_fwhm_color(fwhm_list[i])
        color_norm = tuple(c/255.0 for c in color)
        ax.plot(times_list[i:i+2], fwhm_list[i:i+2], 
               color=color_norm, linewidth=2, marker='o', markersize=3)
    
    ax.set_xlabel('Temps (s)', color='white', fontsize=11, fontweight='bold')
    ax.set_ylabel('FWHM (px)', color='white', fontsize=11, fontweight='bold')
    ax.set_title('Évolution FWHM', color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='white', labelsize=10)
    ax.grid(True, alpha=0.3, color='gray', linestyle='--', linewidth=0.8)
    
    max_fwhm = max(fwhm_list)
    ax.set_ylim(0, max(max_fwhm * 1.2, 15))

    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_linewidth(1.5)

    plt.tight_layout()

    try:
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        w, h = canvas.get_width_height()
        graph_surface = pygame.image.frombuffer(buf, (w, h), 'RGBA')
        return graph_surface
    except:
        return None

def update_hfr_graph(hfr_val):
    """Met à jour le graphique HFR et retourne une surface pygame"""
    global hfr_history, hfr_times, hfr_start_time, hfr_fig, hfr_ax
    
    if hfr_val is None:
        return None
    
    if hfr_start_time == 0:
        hfr_start_time = time.time()
    
    current_time = time.time() - hfr_start_time
    hfr_history.append(hfr_val)
    hfr_times.append(current_time)
    
    if len(hfr_history) < 2:
        return None
    
    fig, ax = init_hfr_graph()
    ax.clear()
    
    # Zones de qualité HFR (valeurs différentes de FWHM)
    ax.axhspan(0, 2, alpha=0.2, color='green', linewidth=0)
    ax.axhspan(2, 3.5, alpha=0.2, color='yellow', linewidth=0)
    ax.axhspan(3.5, 5, alpha=0.2, color='orange', linewidth=0)
    ax.axhspan(5, max(list(hfr_history) + [8]), alpha=0.2, color='red', linewidth=0)
    
    # Courbe HFR avec couleurs
    times_list = list(hfr_times)
    hfr_list = list(hfr_history)
    
    for i in range(len(hfr_list) - 1):
        # Déterminer la couleur selon HFR
        if hfr_list[i] < 2:
            color = (0, 255, 0)  # vert
        elif hfr_list[i] < 3.5:
            color = (255, 255, 0)  # jaune
        elif hfr_list[i] < 5:
            color = (255, 165, 0)  # orange
        else:
            color = (255, 0, 0)  # rouge
        
        color_norm = tuple(c/255.0 for c in color)
        ax.plot(times_list[i:i+2], hfr_list[i:i+2],
               color=color_norm, linewidth=2.5, marker='o', markersize=4)
    
    ax.set_xlabel('Temps (s)', color='white', fontsize=11, fontweight='bold')
    ax.set_ylabel('HFR (px)', color='white', fontsize=11, fontweight='bold')
    ax.set_title('Évolution HFR', color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='white', labelsize=10)
    ax.grid(True, alpha=0.3, color='gray', linestyle='--', linewidth=0.8)

    max_hfr = max(hfr_list)
    ax.set_ylim(0, max(max_hfr * 1.2, 8))

    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    try:
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        w, h = canvas.get_width_height()
        graph_surface = pygame.image.frombuffer(buf, (w, h), 'RGBA')
        return graph_surface
    except:
        return None

def reset_fwhm_history():
    """Réinitialise l'historique FWHM"""
    global fwhm_history, fwhm_times, fwhm_start_time
    fwhm_history.clear()
    fwhm_times.clear()
    fwhm_start_time = 0

def reset_hfr_history():
    """Réinitialise l'historique HFR"""
    global hfr_history, hfr_times, hfr_start_time
    hfr_history.clear()
    hfr_times.clear()
    hfr_start_time = 0

def init_focus_graph():
    """Initialise le graphique Focus (Laplacian variance)"""
    global focus_fig, focus_ax
    if focus_fig is None:
        focus_fig, focus_ax = plt.subplots(figsize=(6, 3), dpi=100)
        focus_fig.patch.set_facecolor('#1a1a1a')
        focus_ax.set_facecolor('#0a0a0a')
    return focus_fig, focus_ax

def update_focus_graph(focus_val):
    """Met à jour le graphique Focus et retourne une surface pygame"""
    global focus_history, focus_times, focus_start_time, focus_fig, focus_ax

    if focus_val is None or focus_val == 0:
        return None

    if focus_start_time == 0:
        focus_start_time = time.time()

    current_time = time.time() - focus_start_time
    focus_history.append(focus_val)
    focus_times.append(current_time)

    if len(focus_history) < 2:
        return None

    fig, ax = init_focus_graph()
    ax.clear()

    # Zones de qualité Focus avec gradients subtils
    ax.axhspan(0, 50, alpha=0.15, color='red', linewidth=0)
    ax.axhspan(50, 200, alpha=0.15, color='orange', linewidth=0)
    ax.axhspan(200, 500, alpha=0.15, color='yellow', linewidth=0)
    max_focus = max(list(focus_history) + [800])
    ax.axhspan(500, max_focus * 1.2, alpha=0.15, color='green', linewidth=0)

    # Courbe Focus avec dégradé de couleurs
    times_list = list(focus_times)
    focus_list = list(focus_history)

    for i in range(len(focus_list) - 1):
        # Déterminer la couleur selon la qualité du focus
        if focus_list[i] > 500:
            color = (0, 255, 0)  # vert
        elif focus_list[i] > 200:
            color = (255, 255, 0)  # jaune
        elif focus_list[i] > 50:
            color = (255, 165, 0)  # orange
        else:
            color = (255, 0, 0)  # rouge

        color_norm = tuple(c/255.0 for c in color)
        ax.plot(times_list[i:i+2], focus_list[i:i+2],
               color=color_norm, linewidth=3, marker='o', markersize=5,
               markeredgecolor='white', markeredgewidth=0.5)

    ax.set_xlabel('Temps (s)', color='white', fontsize=11, fontweight='bold')
    ax.set_ylabel('Focus', color='white', fontsize=11, fontweight='bold')
    ax.set_title('Évolution Focus (Laplacian)', color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='white', labelsize=10)
    ax.grid(True, alpha=0.3, color='gray', linestyle='--', linewidth=0.8)

    ax.set_ylim(0, max(max_focus * 1.2, 800))

    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_linewidth(1.5)

    plt.tight_layout()

    try:
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        w, h = canvas.get_width_height()
        graph_surface = pygame.image.frombuffer(buf, (w, h), 'RGBA')
        return graph_surface
    except:
        return None

def update_combined_hfr_fwhm_graph(hfr_val, fwhm_val):
    """Met à jour un graphique combiné HFR+FWHM et retourne une surface pygame"""
    global hfr_history, hfr_times, hfr_start_time
    global fwhm_history, fwhm_times, fwhm_start_time

    # Synchroniser les temps de départ
    if hfr_start_time == 0 and fwhm_start_time == 0:
        hfr_start_time = fwhm_start_time = time.time()
    elif hfr_start_time == 0:
        hfr_start_time = fwhm_start_time
    elif fwhm_start_time == 0:
        fwhm_start_time = hfr_start_time

    current_time = time.time() - hfr_start_time

    # Ajouter les valeurs aux historiques
    if hfr_val is not None:
        hfr_history.append(hfr_val)
        hfr_times.append(current_time)

    if fwhm_val is not None:
        fwhm_history.append(fwhm_val)
        fwhm_times.append(current_time)

    if len(hfr_history) < 2 and len(fwhm_history) < 2:
        return None

    # Créer le graphique avec double axe Y
    fig, ax1 = plt.subplots(figsize=(6, 3), dpi=100)
    fig.patch.set_facecolor('#1a1a1a')
    ax1.set_facecolor('#0a0a0a')

    # Axe pour HFR (gauche)
    if len(hfr_history) >= 2:
        hfr_times_list = list(hfr_times)
        hfr_list = list(hfr_history)

        # Tracer HFR avec couleur dynamique
        for i in range(len(hfr_list) - 1):
            if hfr_list[i] < 2:
                color = '#00ff00'  # vert
            elif hfr_list[i] < 3.5:
                color = '#ffff00'  # jaune
            else:
                color = '#ff6600'  # orange

            ax1.plot(hfr_times_list[i:i+2], hfr_list[i:i+2],
                    color=color, linewidth=2.5, marker='o', markersize=4)

        ax1.set_ylabel('HFR (px)', color='#00ff88', fontsize=11, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#00ff88', colors='#00ff88', labelsize=10)
        max_hfr = max(hfr_list)
        ax1.set_ylim(0, max(max_hfr * 1.2, 8))

    # Axe pour FWHM (droite)
    ax2 = ax1.twinx()
    if len(fwhm_history) >= 2:
        fwhm_times_list = list(fwhm_times)
        fwhm_list = list(fwhm_history)

        # Tracer FWHM avec couleur dynamique
        for i in range(len(fwhm_list) - 1):
            if fwhm_list[i] < 5:
                color = '#ff00ff'  # magenta
            elif fwhm_list[i] < 10:
                color = '#ff88ff'  # rose
            else:
                color = '#ff0088'  # rouge-rose

            ax2.plot(fwhm_times_list[i:i+2], fwhm_list[i:i+2],
                    color=color, linewidth=2.5, marker='s', markersize=4, linestyle='--')

        ax2.set_ylabel('FWHM (px)', color='#ff00ff', fontsize=11, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#ff00ff', colors='#ff00ff', labelsize=10)
        max_fwhm = max(fwhm_list)
        ax2.set_ylim(0, max(max_fwhm * 1.2, 15))

    ax1.set_xlabel('Temps (s)', color='white', fontsize=11, fontweight='bold')
    ax1.set_title('HFR + FWHM', color='white', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', colors='white', labelsize=10)
    ax1.grid(True, alpha=0.3, color='gray', linestyle=':', linewidth=0.8)

    for spine in ax1.spines.values():
        spine.set_color('white')
        spine.set_linewidth(1.5)
    for spine in ax2.spines.values():
        spine.set_color('white')
        spine.set_linewidth(1.5)

    plt.tight_layout()

    try:
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        w, h = canvas.get_width_height()
        graph_surface = pygame.image.frombuffer(buf, (w, h), 'RGBA')
        return graph_surface
    except:
        return None

def reset_focus_history():
    """Réinitialise l'historique Focus"""
    global focus_history, focus_times, focus_start_time
    focus_history.clear()
    focus_times.clear()
    focus_start_time = 0

def astro_stretch(array):
    """
    Applique un étirement astro (autostretch) sur une image numpy array
    Utilise une technique de normalisation par percentiles et fonction asinh
    pour accentuer les détails faibles tout en préservant les hautes lumières

    Args:
        array: numpy array de l'image (H, W, 3) en RGB

    Returns:
        numpy array étiré de même dimension
    """
    global stretch_p_low, stretch_p_high, stretch_factor

    # Convertir en float pour les calculs
    img_float = array.astype(np.float32)

    # Calculer les percentiles pour éviter l'effet des pixels chauds
    # Utilise les valeurs configurables (divisées par 10 car stockées x10)
    p_low = np.percentile(img_float, stretch_p_low / 10.0)
    p_high = np.percentile(img_float, stretch_p_high / 10.0)

    # Éviter la division par zéro
    if p_high - p_low < 1:
        return array

    # Normaliser entre 0 et 1
    img_normalized = (img_float - p_low) / (p_high - p_low)
    img_normalized = np.clip(img_normalized, 0, 1)

    # Appliquer une transformation asinh pour accentuer les détails faibles
    # asinh est plus doux que sqrt et préserve mieux les détails
    # Utilise le facteur configurable (divisé par 10 car stocké x10)
    factor = stretch_factor / 10.0
    img_stretched = np.arcsinh(img_normalized * factor) / np.arcsinh(factor)

    # Reconvertir en uint8
    img_stretched = (img_stretched * 255).astype(np.uint8)

    return img_stretched

def preview():
    global use_ard,lver,Pi,scientif,scientific,fxx,fxy,fxz,v3_focus,v3_hdr,v3_f_mode,v3_f_modes,prev_fps,focus_fps,focus_mode,restart,datastr
    global count,p, brightness,contrast,modes,mode,red,blue,gain,sspeed,ev,preview_width,preview_height,zoom,igw,igh,zx,zy,awbs,awb,saturations
    global saturation,meters,meter,flickers,flicker,sharpnesss,sharpness,rotate,v3_hdrs,mjpeg_extractor
    global picam2, use_picamera2, Pi_Cam, camera, v3_af, v5_af, vflip, hflip, denoise, denoises, quality

    # Variables statiques pour mémoriser la configuration précédente
    if not hasattr(preview, 'prev_config'):
        preview.prev_config = {}

    # ===== MODE PICAMERA2 =====
    if use_picamera2:
        # Déterminer la taille de capture pour détecter si changement
        # Si zoom actif, utiliser la résolution correspondante au niveau de zoom
        if zoom > 0:
            # Résolutions correspondant aux niveaux de zoom
            zoom_capture_sizes = {
                1: (1920, 1080),  # 2x zoom
                2: (1280, 720),   # 3x zoom
                4: (640, 480),    # 5x zoom
                5: (640, 368)     # 6x zoom
            }
            capture_size = zoom_capture_sizes.get(zoom, (vwidth, vheight))
        elif (Pi_Cam == 5 or Pi_Cam == 6 or Pi_Cam == 8) and focus_mode == 1:
            capture_size = (3280, 2464)
        elif (Pi_Cam == 5 or Pi_Cam == 6 or Pi_Cam == 8):
            capture_size = (1920, 1440)
        elif Pi_Cam == 3:
            capture_size = (2304, 1296)
        elif Pi_Cam == 10:
            capture_size = (1928, 1090)
        else:
            capture_size = (preview_width, preview_height)

        # Détecter si recréation nécessaire (changements majeurs de config)
        need_recreation = (
            picam2 is None or
            preview.prev_config.get('camera') != camera or
            preview.prev_config.get('capture_size') != capture_size or
            preview.prev_config.get('vflip') != vflip or
            preview.prev_config.get('hflip') != hflip or
            preview.prev_config.get('mode_type') != (0 if mode == 0 or sspeed > 80000 else 1)
        )

        # Calculer speed2 et autres paramètres (avant le if/else)
        speed2 = sspeed
        max_exposure_seconds = max_shutters[Pi_Cam]
        max_frame_duration = int(max_exposure_seconds * 1_000_000)
        min_frame_duration = 11415 if Pi_Cam == 10 else 100

        # ========== CHEMIN RAPIDE : Juste changer les contrôles ==========
        if not need_recreation and picam2 is not None:
            if show_cmds == 1:
                print(f"  [FAST PATH] Just updating controls (no recreation)...")

            # Préparer les contrôles à changer
            fast_controls = {}

            # Exposition et gain (mode manuel)
            if mode == 0:
                fast_controls["FrameDurationLimits"] = (min_frame_duration, max(max_frame_duration, speed2))
                fast_controls["ExposureTime"] = speed2
                fast_controls["AnalogueGain"] = float(gain)

            # Brightness & Contrast
            fast_controls["Brightness"] = brightness / 100
            fast_controls["Contrast"] = contrast / 100

            # AWB
            if awb == 0:
                fast_controls["ColourGains"] = (red/10, blue/10)
            else:
                awb_modes = {
                    1: controls.AwbModeEnum.Auto, 2: controls.AwbModeEnum.Incandescent,
                    3: controls.AwbModeEnum.Tungsten, 4: controls.AwbModeEnum.Fluorescent,
                    5: controls.AwbModeEnum.Indoor, 6: controls.AwbModeEnum.Daylight, 7: controls.AwbModeEnum.Cloudy,
                }
                if awb in awb_modes:
                    fast_controls["AwbMode"] = awb_modes[awb]

            # Saturation & Sharpness
            fast_controls["Saturation"] = saturation / 10
            fast_controls["Sharpness"] = sharpness / 10

            # Appliquer tous les contrôles en une seule fois (rapide!)
            try:
                picam2.set_controls(fast_controls)

                if show_cmds == 1:
                    print(f"  ✓ Controls updated instantly - ExposureTime={speed2}µs, Gain={gain}")

                # Pas besoin de mémoriser la config car elle n'a pas changé
                restart = 0
                return  # Early return - évite toute la recréation !
            except RuntimeError as e:
                # Si les contrôles ne sont pas disponibles (ex: après rpicam-still),
                # forcer une recréation complète
                if show_cmds == 1:
                    print(f"  ⚠ Fast path failed ({e}), forcing full recreation...")
                need_recreation = True  # Forcer la recréation
                # Continue vers le chemin complet ci-dessous

        # ========== CHEMIN COMPLET : Recréation nécessaire ==========
        if show_cmds == 1:
            print(f"  [FULL RECREATION] Config changed - recreating Picamera2...")

            # Arrêter l'ancienne instance
            if picam2 is not None:
                try:
                    picam2.stop()
                    picam2.close()
                    time.sleep(0.5)
                except:
                    pass

            # Créer nouvelle instance
            picam2 = Picamera2(camera)

        # Déterminer la taille de capture selon caméra (pour les deux chemins)
        # Si zoom actif, utiliser la résolution correspondante au niveau de zoom
        if zoom > 0:
            zoom_capture_sizes = {
                1: (1920, 1080),  # 2x zoom
                2: (1280, 720),   # 3x zoom
                4: (640, 480),    # 5x zoom
                5: (640, 368)     # 6x zoom
            }
            capture_size = zoom_capture_sizes.get(zoom, (vwidth, vheight))
        elif (Pi_Cam == 5 or Pi_Cam == 6 or Pi_Cam == 8) and focus_mode == 1:
            capture_size = (3280, 2464)
        elif (Pi_Cam == 5 or Pi_Cam == 6 or Pi_Cam == 8):
            capture_size = (1920, 1440)
        elif Pi_Cam == 3:  # Pi v3
            capture_size = (2304, 1296)
        elif Pi_Cam == 10: # imx585
            capture_size = (1928, 1090)
        else:
            capture_size = (preview_width, preview_height)

        # Configuration de base
        # IMPORTANT: Forcer le mode 12-bit au lieu de 16-bit pour que les contrôles fonctionnent
        # Le mode 16-bit de l'IMX585 ne supporte pas les contrôles automatiques
        from picamera2 import Preview

        # Calculer speed2 (comme dans le code rpicam-vid)
        speed2 = sspeed
        # Limite retirée pour permettre les longues expositions
        # speed2 = min(speed2, 2000000)

        # FrameDurationLimits : Calculer basé sur les capacités du capteur
        # Pour IMX585: min=11415µs, max=163070574µs (~163s)
        max_exposure_seconds = max_shutters[Pi_Cam]
        max_frame_duration = int(max_exposure_seconds * 1_000_000)  # Convertir en µs
        # Utiliser le minimum du capteur IMX585, pas une valeur arbitraire
        min_frame_duration = 11415 if Pi_Cam == 10 else 100  # 11.415ms pour IMX585

        # Créer la configuration adaptée au mode
        # En mode manuel ou avec exposition > 80ms, utiliser still_configuration
        # (preview_configuration limite l'exposition à ~83ms pour l'IMX585)
        if mode == 0 or sspeed > 80000:
            config = picam2.create_still_configuration(
                main={"size": capture_size, "format": "RGB888"},
                raw={"size": capture_size, "format": "SRGGB12"}  # Forcer 12-bit
            )
        else:
            config = picam2.create_preview_configuration(
                main={"size": capture_size, "format": "RGB888"},
                raw={"size": capture_size, "format": "SRGGB12"}  # Forcer 12-bit
            )

        # Préparer TOUS les contrôles pour application après start()
        controls_dict = {}

        # Brightness & Contrast
        controls_dict["Brightness"] = brightness / 100
        controls_dict["Contrast"] = contrast / 100

        # Exposition et mode
        if mode == 0:
            # Mode manuel
            # IMPORTANT: Appliquer FrameDurationLimits dynamiquement pour éviter que la caméra
            # garde un FrameDuration élevé après une exposition longue
            controls_dict["FrameDurationLimits"] = (min_frame_duration, max(max_frame_duration, speed2))
            controls_dict["ExposureTime"] = speed2
            controls_dict["AnalogueGain"] = float(gain)
        else:
            # Mode auto - ne rien définir, laisser l'AE gérer
            pass

        # Framerate
        # NE PAS définir FrameRate en mode manuel pour permettre les longues expositions
        if mode != 0:
            if zoom > 0:
                fps = focus_fps
            else:
                fps = prev_fps
            controls_dict["FrameRate"] = fps

        # AWB (Balance des blancs)
        # Ne PAS utiliser AwbEnable qui n'est pas supporté sur toutes les caméras
        if awb == 0:
            # AWB manuel - définir ColourGains désactive automatiquement AWB
            controls_dict["ColourGains"] = (red/10, blue/10)
        else:
            # AWB auto - utiliser AwbMode
            awb_modes = {
                1: controls.AwbModeEnum.Auto,
                2: controls.AwbModeEnum.Incandescent,
                3: controls.AwbModeEnum.Tungsten,
                4: controls.AwbModeEnum.Fluorescent,
                5: controls.AwbModeEnum.Indoor,
                6: controls.AwbModeEnum.Daylight,
                7: controls.AwbModeEnum.Cloudy,
            }
            if awb in awb_modes:
                controls_dict["AwbMode"] = awb_modes[awb]

        # Saturation & Sharpness
        controls_dict["Saturation"] = saturation / 10
        controls_dict["Sharpness"] = sharpness / 10

        # Focus (autofocus ou manuel) - sera appliqué séparément après start()
        focus_controls = {}
        if ((Pi_Cam == 3 and v3_af == 1) and v3_f_mode > 0) or (((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6)) or Pi_Cam == 8:
            if v3_f_mode == 1:  # Manuel
                focus_controls["AfMode"] = controls.AfModeEnum.Manual
                if Pi_Cam == 3:
                    focus_controls["LensPosition"] = v3_focus / 100
                elif Pi_Cam == 8 or ((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6):
                    focus_controls["LensPosition"] = focus / 100
            else:  # Auto
                focus_controls["AfMode"] = controls.AfModeEnum.Auto

        # Transform (flip)
        if vflip == 1 or hflip == 1:
            config["transform"] = Transform(vflip=(vflip == 1), hflip=(hflip == 1))

        picam2.configure(config)

        # CRITIQUE: En mode manuel, appliquer TOUS les contrôles AVANT start()
        # Cela évite que libcamera garde un état résiduel d'une configuration précédente
        if mode == 0:
            initial_frame_duration = (min_frame_duration, max(max_frame_duration, speed2))
            pre_start_controls = {
                "FrameDurationLimits": initial_frame_duration,
                "ExposureTime": speed2,
                "AnalogueGain": float(gain)
            }
            picam2.set_controls(pre_start_controls)
            if show_cmds == 1:
                print(f"  Pre-start controls: FDL={initial_frame_duration[0]}-{initial_frame_duration[1]}µs, Exp={speed2}µs, Gain={gain}")
        else:
            initial_frame_duration = (min_frame_duration, max(max_frame_duration, speed2))
            picam2.set_controls({"FrameDurationLimits": initial_frame_duration})
            if show_cmds == 1:
                print(f"  Pre-start FrameDurationLimits: {initial_frame_duration[0]}µs to {initial_frame_duration[1]}µs")

        picam2.start()

        # Attendre que la caméra démarre (réduit pour accélérer)
        time.sleep(0.1)

        # APPLIQUER TOUS LES CONTRÔLES APRÈS LE START
        # Pour l'IMX585, c'est nécessaire pour que les contrôles soient correctement appliqués
        try:
            if show_cmds == 1:
                print(f"Applying controls: ExposureTime={speed2}µs, Gain={controls_dict.get('AnalogueGain')}, Mode={mode}")
                fd_limits = controls_dict.get('FrameDurationLimits', (min_frame_duration, max_frame_duration))
                print(f"  FrameDurationLimits: {fd_limits[0]}µs to {fd_limits[1]}µs ({fd_limits[1]/1000000:.1f}s max)")

            # En mode manuel, appliquer ExposureTime et AnalogueGain ensemble
            # Cela désactive automatiquement l'auto-exposition
            if mode == 0:
                exposure_controls = {
                    "FrameDurationLimits": controls_dict["FrameDurationLimits"],
                    "ExposureTime": controls_dict["ExposureTime"],
                    "AnalogueGain": controls_dict["AnalogueGain"]
                }

                picam2.set_controls(exposure_controls)

                # Suppression des étapes de stabilisation/réapplication pour expositions longues
                # (trop lent, pas nécessaire)
                time.sleep(0.1)

                # Appliquer les autres contrôles
                other_controls = {k: v for k, v in controls_dict.items()
                                if k not in ["FrameDurationLimits", "ExposureTime", "AnalogueGain"]}
                if other_controls:
                    picam2.set_controls(other_controls)
                    time.sleep(0.05)
            else:
                # Mode auto : appliquer tous les contrôles ensemble
                picam2.set_controls(controls_dict)
                time.sleep(0.1)

            # Appliquer les contrôles de focus séparément si nécessaire
            if focus_controls:
                picam2.set_controls(focus_controls)
                time.sleep(0.05)

            # ===== APPLIQUER LE ZOOM PREVIEW avec ScalerCrop =====
            if zoom > 0 and zoom <= 5 and zoom != 3:  # Zoom 3 désactivé
                try:
                    # Obtenir la taille native du capteur pour ScalerCrop
                    # ScalerCrop utilise les coordonnées absolues du capteur natif
                    if Pi_Cam == 10:  # IMX585
                        sensor_width = 3856
                        sensor_height = 2180
                    elif Pi_Cam == 3:  # Pi v3
                        sensor_width = 4608
                        sensor_height = 2592
                    elif Pi_Cam == 4:  # Pi HQ
                        sensor_width = 4056
                        sensor_height = 3040
                    else:
                        sensor_width = 3280
                        sensor_height = 2464

                    # Calculer la région crop (même logique que le ROI pour rpicam-vid)
                    # Arrondir à des nombres pairs
                    crop_width = int(sensor_width * zfs[zoom])
                    crop_height = int(sensor_height * zfs[zoom])
                    if crop_width % 2 != 0:
                        crop_width -= 1
                    if crop_height % 2 != 0:
                        crop_height -= 1

                    # Centrer le crop
                    crop_x = (sensor_width - crop_width) // 2
                    crop_y = (sensor_height - crop_height) // 2

                    # Appliquer ScalerCrop
                    scaler_crop = (crop_x, crop_y, crop_width, crop_height)
                    picam2.set_controls({"ScalerCrop": scaler_crop})
                    time.sleep(0.05)

                    if show_cmds == 1:
                        zoom_factor = 1.0 / zfs[zoom]
                        print(f"  Zoom preview: {zoom_factor:.1f}x applied via ScalerCrop")
                        print(f"    ScalerCrop: {scaler_crop} (x,y,w,h)")
                except Exception as e:
                    if show_cmds == 1:
                        print(f"  Warning: ScalerCrop application failed: {e}")

        except Exception as e:
            if show_cmds == 1:
                print(f"Warning: Control application failed: {e}")

        if show_cmds == 1:
            print("✓ Picamera2 started successfully")
            print(f"  Capture size: {capture_size}")
            print(f"  Mode: {mode} (0=manuel, other=auto)")
            print(f"  ExposureTime demandé: {speed2}µs ({speed2/1000}ms)")
            print(f"  Gain demandé: {gain}")

            # Afficher les limites des contrôles disponibles
            camera_controls = picam2.camera_controls
            if "ExposureTime" in camera_controls:
                exp_limits = camera_controls["ExposureTime"]
                print(f"  Camera ExposureTime limits: {exp_limits}")
            if "FrameDurationLimits" in camera_controls:
                fd_limits = camera_controls["FrameDurationLimits"]
                print(f"  Camera FrameDurationLimits: {fd_limits}")

            # Lire les contrôles effectifs après application
            # Délai réduit pour accélérer
            time.sleep(0.1)
            actual_controls = picam2.capture_metadata()
            if actual_controls:
                actual_exp = actual_controls.get('ExposureTime')
                actual_gain = actual_controls.get('AnalogueGain')
                actual_fd = actual_controls.get('FrameDuration')
                print(f"  ExposureTime réel: {actual_exp}µs ({actual_exp/1000:.2f}ms)")
                print(f"  Gain réel: {actual_gain:.2f}")
                if actual_fd:
                    print(f"  FrameDuration réel: {actual_fd}µs ({actual_fd/1000:.2f}ms)")
                # Vérifier si les valeurs sont proches
                if abs(actual_exp - speed2) > speed2 * 0.1:  # Plus de 10% de différence
                    print(f"  ⚠ WARNING: ExposureTime réel différent de la demande!")
                    print(f"    Écart: {speed2 - actual_exp}µs ({100*(speed2-actual_exp)/speed2:.1f}%)")

        # Mémoriser la configuration actuelle pour la prochaine fois
        preview.prev_config = {
            'camera': camera,
            'capture_size': capture_size,
            'vflip': vflip,
            'hflip': hflip,
            'mode_type': 0 if mode == 0 or sspeed > 80000 else 1
        }

        restart = 0
        # Suppression du sleep final pour accélérer
        return

    # ===== MODE RPICAM-VID (code original) =====
    # Nettoyer les anciens fichiers et processus
    files = glob.glob('/run/shm/*.jpg')
    for f in files:
        if os.path.exists(f):
            os.remove(f)
    if os.path.exists('/run/shm/stream.mjpeg'):
        os.remove('/run/shm/stream.mjpeg')
    
    # Arrêter l'ancien extracteur s'il existe
    if mjpeg_extractor is not None:
        mjpeg_extractor.stop()
        mjpeg_extractor = None
    speed2 = sspeed
    # Limite retirée pour permettre les longues expositions
    # speed2 = min(speed2,2000000)
    if lver != "bookwo" and lver != "trixie":
        datastr = "libcamera-vid"
    else:
        datastr = "rpicam-vid"

    # RETOUR À LA MÉTHODE ÉPROUVÉE : --segment 1 (programme original ligne 704)
    # --segment 1 = 1ms, force un nouveau fichier à chaque frame
    datastr += " --camera " + str(camera) + " -n --codec mjpeg -t 0 --segment 1"

    # SORTIE DIRECTE VERS FICHIERS JPEG (méthode originale simple et stable)
    # Si zoom actif, utiliser la résolution correspondante au niveau de zoom
    if zoom > 0:
        zoom_cmd_resolutions = {
            1: " --width 1920 --height 1080 -o /run/shm/test%04d.jpg ",
            2: " --width 1280 --height 720 -o /run/shm/test%04d.jpg ",
            4: " --width 640 --height 480 -o /run/shm/test%04d.jpg ",
            5: " --width 640 --height 368 -o /run/shm/test%04d.jpg "
        }
        datastr += zoom_cmd_resolutions.get(zoom, f" --width {vwidth} --height {vheight} -o /run/shm/test%04d.jpg ")
    elif (Pi_Cam == 5 or Pi_Cam == 6 or Pi_Cam == 8) and focus_mode == 1:
        datastr += " --width 3280 --height 2464 -o /run/shm/test%04d.jpg "
    elif (Pi_Cam == 5 or Pi_Cam == 6 or Pi_Cam == 8):
        datastr += " --width 1920 --height 1440 -o /run/shm/test%04d.jpg "
    elif Pi_Cam == 3:  # Pi v3
        datastr += " --width 2304 --height 1296 -o /run/shm/test%04d.jpg "
    elif Pi_Cam == 7:  # Pi GS
        datastr += " --width 1456 --height 1088 -o /run/shm/test%04d.jpg "
    elif Pi_Cam == 9:  # imx290
        datastr += " --width 1920 --height 1080 -o /run/shm/test%04d.jpg "
    elif Pi_Cam == 10: # imx585
        datastr += " --width 1928 --height 1090 -o /run/shm/test%04d.jpg "
    elif Pi_Cam == 11: # imx293
        datastr += " --width 1920 --height 1080 -o /run/shm/test%04d.jpg "
    elif Pi_Cam == 12: # imx294
        datastr += " --width 2048 --height 1080 -o /run/shm/test%04d.jpg "
    elif Pi_Cam == 13: # imx283
        datastr += " --width 1920 --height 1080 -o /run/shm/test%04d.jpg "
    elif Pi_Cam == 14: # imx500
        datastr += " --width 2028 --height 1520 -o /run/shm/test%04d.jpg "
    elif Pi_Cam == 15: # ov9281
        datastr += " --width 1280 --height  800 -o /run/shm/test%04d.jpg "
    elif Pi_Cam == 1:  # v1 / ov5647
        datastr += " --width 1296 --height 972 -o /run/shm/test%04d.jpg "
    else:
        if preview_width == 640 and preview_height == 480:
            datastr += " --width 720 --height 540 -o /run/shm/test%04d.jpg "
        else:
            # Pour un affichage 880x580, forcer le mode de capteur le plus bas : 1280x720
            # --mode force le capteur à utiliser ce mode natif (au lieu de 3856x2180 + downscale)
            # Format: largeur:hauteur:profondeur_bits (10 ou 12 bits selon capteur)
            datastr += " --mode 1280:720:10 --width 1280 --height 720 -o /run/shm/test%04d.jpg "
    if ev != 0:
        datastr += " --ev " + str(ev)
    datastr += " --brightness " + str(brightness/100) + " --contrast " + str(contrast/100)
    if mode == 0:
        datastr += " --shutter " + str(speed2)
    else:
        datastr += " --exposure " + str(modes[mode])
    # Framerate (méthode identique au programme original ligne 741-748)
    if zoom > 0 and mode != 0:
        datastr += " --framerate " + str(focus_fps)
    elif mode != 0:
        datastr += " --framerate " + str(prev_fps)
    elif mode == 0:
        # Calculer FPS basé sur l'exposition (protection division par zéro)
        if speed2 > 0:
            speed3 = max(min(1000000/speed2, 25), 0.01)
        else:
            speed3 = 25
        datastr += " --framerate " + str(speed3)
    if sspeed > 5000000 and mode == 0:
        datastr += " --gain 1 --awbgains " + str(red/10) + "," + str(blue/10)
    else:
        datastr += " --gain " + str(gain)
        if awb == 0:
            datastr += " --awbgains " + str(red/10) + "," + str(blue/10)
        else:
            datastr += " --awb " + awbs[awb]
    datastr += " --metering "   + meters[meter]
    datastr += " --saturation " + str(saturation/10)
    datastr += " --sharpness "  + str(sharpness/10)
    datastr += " --denoise "    + denoises[denoise]
    datastr += " --quality " + str(quality)
    if vflip == 1:
        datastr += " --vflip"
    if hflip == 1:
        datastr += " --hflip"
    if ((Pi_Cam == 3 and v3_af == 1) and v3_f_mode > 0 and fxx == 0) or (((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6)) or Pi_Cam == 8:
        datastr += " --autofocus-mode " + v3_f_modes[v3_f_mode]
        if v3_f_mode == 1:
            if Pi_Cam == 3:
                datastr += " --lens-position " + str(v3_focus/100)
            if Pi_Cam == 8 or ((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6):
                datastr += " --lens-position " + str(focus/100)
    if ((Pi_Cam == 3 and v3_af == 1) or (((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6)) or Pi_Cam == 8) and zoom == 0 and fxx != 0 and v3_f_mode != 1:
        datastr += " --autofocus-window " + str(fxx) + "," + str(fxy) + "," + str(fxz) + "," + str(fxz)
    if (Pi_Cam == 3 and v3_af == 1) and v3_f_speed != 0:
        datastr += " --autofocus-speed " + v3_f_speeds[v3_f_speed]
    if (Pi_Cam == 3 and v3_af == 1) and v3_f_range != 0:
        datastr += " --autofocus-range " + v3_f_ranges[v3_f_range]
    if Pi_Cam == 3 or Pi == 5:
        datastr += " --hdr " + v3_hdrs[v3_hdr]
    if Pi_Cam == 9 and os.path.exists("/home/" + Home_Files[0] + "/imx290a.json") and Pi == 5:
            datastr += " --tuning-file /home/" + Home_Files[0] + "/imx290a.json"
    if Pi_Cam == 10 and os.path.exists("/home/" + Home_Files[0] + "/imx585_lowlight.json") and Pi == 5:
            datastr += " --tuning-file /home/" + Home_Files[0] + "/imx585_lowlight.json"
    if Pi_Cam == 4 and scientific == 1:
        if os.path.exists('/usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json') and Pi == 4:
            datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json"
        if os.path.exists('/usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json') and Pi == 5:
            datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json"
    # Zoom fixe 1x à 6x
    if zoom > 0 and zoom <= 5 and zoom != 3:  # Zoom 3 désactivé
        zws = int(igw * zfs[zoom])
        zhs = int(igh * zfs[zoom])
        zxo = ((igw-zws)/2)/igw
        zyo = ((igh-zhs)/2)/igh
        datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(zws/igw) + "," + str(zhs/igh)
    # Supprimé: ancien zoom manuel (zoom == 5)
    if False and zoom == 5:
        zxo = ((igw/2)-(preview_width/2))/igw
        if alt_dis == 2:
            zyo = ((igh/2)-((preview_height * .75)/2))/igh
        else:
            zyo = ((igh/2)-(preview_height/2))/igh
        if igw/igh > 1.5:
            datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(int(preview_width)/igw) + "," + str((preview_height * .75)/igh)
        else:
            datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(preview_width/igw) + "," + str(preview_height/igh)
    p = subprocess.Popen(datastr, shell=True, preexec_fn=os.setsid)
    if show_cmds == 1:
        print(datastr)

    # Plus besoin d'extracteur MJPEG : libcamera-vid écrit directement les fichiers JPEG
    # C'est la méthode simple et éprouvée du programme original

    restart = 0
    time.sleep(0.2)

def Menu():
    global vwidths2,vheights2,Pi_Cam,scientif,mode,v3_hdr,scientific,tinterval,zoom,vwidth,vheight,preview_width,preview_height,ft,fv,focus,fxz,v3_hdr,v3_hdrs,bw,bh,ft,fv,cam1,v3_f_mode,v3_af,button_row,xx,xy
    pygame.draw.rect(windowSurfaceObj,(0,0,0),Rect(preview_width,0,bw,preview_height))
    if menu > 0: 
        # set button sizes
        bw = int(preview_width/5.66)
        bh = int(preview_height/10)
        ft = int(preview_width/46)
        fv = int(preview_width/46)
        for d in range(1,9):
            button(0,0,0,4)
            if menu == 1:
                button(0,d,0,4)
            elif menu == 2:
                button(0,d,0,4)
            elif menu == 3 or menu == 4:
                button(0,d,6,4)
            elif menu == 5:
                button(0,d,7,4)
            elif menu == 6:
                button(0,d,8,4)
            elif menu == 7:
                button(0,d,0,4)
            elif menu == 8:
                button(0,d,0,4)
        text(0,0,1,0,1,"MAIN MENU ",ft,7)
      
    if menu == 0:
        # set button sizes
        bw = int(preview_width/5.66)
        bh = int(preview_height/8)
        ft = int(preview_width/46)
        fv = int(preview_width/46)
        # Effacer la zone des boutons pour éviter les sliders résiduels
        pygame.draw.rect(windowSurfaceObj,blackColor,Rect(preview_width,0,bw,preview_height))
        button(0,0,4,4)
        button(0,1,2,4)
        button(0,2,3,4)
        button(0,3,9,4)
        button(0,4,5,4)
        button(0,5,0,4)
        button(0,6,0,4)
        button(0,7,0,4)
        text(0,0,8,0,1,"         STILL ",ft,1)
        text(0,1,8,0,1,"         VIDEO",ft,2)
        text(0,2,8,0,1,"       TIMELAPSE",ft-1,4)
        text(0,3,1,0,1,"     LIVE STACK",ft,1)
        text(0,4,1,0,1,"       STRETCH",ft,1)
        text(0,5,1,0,1,"      CAMERA",ft,7)
        text(0,5,1,1,1,"    Settings",ft,7)
        text(0,6,1,0,1,"       OTHER",ft,7)
        text(0,6,1,1,1,"    Settings",ft,7)
        text(0,7,2,0,1,"     EXIT",fv+10,7)
      
    elif menu == 1:
      text(0,1,1,0,1,"STILL",ft,7)
      text(0,1,1,1,1,"Settings",ft,7)
      text(0,2,1,0,1,"VIDEO",ft,7)
      text(0,2,1,1,1,"Settings",ft,7)
      text(0,3,1,0,1,"TIMELAPSE",ft,7)
      text(0,3,1,1,1,"Settings",ft,7)
      text(0,6,1,0,1,"LIVE STACK",ft,7)
      text(0,6,1,1,1,"Settings",ft,7)
      text(0,7,1,0,1,"STRETCH",ft,7)
      text(0,7,1,1,1,"Settings",ft,7)
      if zoom == 0:
          button(0,4,0,9)
          text(0,4,5,0,1,"Zoom",ft,7)
          text(0,4,3,1,1,"",fv,7)
          # determine if camera native format
          vw = 0
          x = 0
          while x < len(vwidths2) and vw == 0:
              if vwidth == vwidths2[x]:
                  if vheight == vheights2[x]:
                      vw = 1
              x += 1
      elif zoom < 10:
          button(0,4,1,9)
          text(0,4,2,0,1,"ZOOMED",ft,0)
          # Afficher la résolution ROI au lieu du numéro de zoom
          text(0,4,3,1,1,zoom_res_labels.get(zoom, str(zoom)),fv,0)
          draw_Vbar(0,4,greyColor,'zoom',zoom)
      if Pi_Cam == 3 and v3_af == 1:
          if fxz != 1:
              text(0,5,3,1,1,"Spot",fv,7)
          else:
              text(0,5,3,1,1,str(v3_f_modes[v3_f_mode]),fv,7)
          if v3_f_mode == 1 :
              button(0,5,1,9)
              draw_Vbar(0,5,dgryColor,'v3_focus',v3_focus-pmin)
              fd = 1/(v3_focus/100)
              text(0,5,3,0,1,'<<< ' + str(fd)[0:5] + "m" + ' >>>',fv,7)
          elif v3_f_mode == 0 or v3_f_mode == 2:
              button(0,5,0,9)
              text(0,5,5,0,1,"FOCUS",ft,7)
          text(0,7,2,0,1,"Focus Speed",ft,7)
          text(0,7,3,1,1,v3_f_speeds[v3_f_speed],fv,7)
          text(0,7,2,0,1,"Focus Range",ft,7)
          text(0,7,3,1,1,v3_f_ranges[v3_f_range],fv,7)
          
      else:
          button(0,5,0,9)
          text(0,5,5,0,1,"FOCUS",ft,7)
          text(0,5,3,1,1,"    ",fv,7)
          
      draw_Vbar(0,4,greyColor,'zoom',zoom)
      if Pi_Cam == 3:
          draw_bar(0,6,greyColor,'v3_f_speed',v3_f_speed)
          draw_Vbar(0,7,greyColor,'v3_f_range',v3_f_range)
        
                 
    elif menu == 2:
        if cam1 != "1":
            text(0,1,2,0,1,"Switch Camera",ft,7)
            text(0,1,3,1,1,str(camera),fv,7)
        text(0,2,2,0,1,"Ext Trig: " + str(STR),ft,7)
        text(0,2,3,1,1,strs[str_cap],fv,7)
        text(0,3,3,0,1,"Histogram",ft,7)
        text(0,3,3,1,1,histograms[histogram],fv,7)
        text(0,4,2,0,1,"Hist Area",ft,7)
        text(0,4,3,1,1,str(histarea),fv,7)
        text(0,5,5,0,1,"Vert Flip",ft,7)
        text(0,5,3,1,1,str(vflip),fv,7)
        text(0,6,5,0,1,"Horiz Flip",ft,7)
        text(0,6,3,1,1,str(hflip),fv,7)
        text(0,7,5,0,1," STILL -t time ",fv,7)
        text(0,7,3,1,1,str(timet),fv,7)
        text(0,8,2,0,1,"SAVE CONFIG",fv,7)
        draw_Vbar(0,2,greyColor,'str_cap',str_cap)
        draw_bar(0,3,greyColor,'histogram',histogram)
        draw_Vbar(0,4,greyColor,'histarea',histarea)
      
    elif menu == 3:
      text(0,1,5,0,1,"Mode",ft,10)
      text(0,1,3,1,1,modes[mode],fv,10)
      if mode == 0:
          text(0,2,5,0,1,"Shutter S",ft,10)
          if shutters[speed] < 0:
              text(0,2,3,1,1,"1/" + str(abs(shutters[speed])),fv,10)
          else:
              text(0,2,3,1,1,str(shutters[speed]),fv,10)
      else:
          text(0,2,5,0,1,"eV",ft,10)
          text(0,2,3,1,1,str(ev),fv,10)
      text(0,3,5,0,1,"Gain    A/D",ft,10)
      if gain > 0:
          text(0,3,5,0,1,"Gain    A/D",ft,10)
          if gain <= mag:
              text(0,3,3,1,1,str(gain) + " :  " + str(gain) + "/1",fv,10)
          else:
              text(0,3,3,1,1,str(gain) + " :  " + str(int(mag)) + "/" + str(((gain/mag)*10)/10)[0:3],fv,10)
      else:
          text(0,3,5,0,1,"Gain",ft,10)
          text(0,3,3,1,1,"Auto",fv,10)
      text(0,4,5,0,1,"Brightness",ft,10)
      text(0,4,3,1,1,str(brightness/100)[0:4],fv,10)
      text(0,5,5,0,1,"Contrast",ft,10)
      text(0,5,3,1,1,str(contrast/100)[0:4],fv,10)
      text(0,6,5,0,1,"AWB",ft,10)
      text(0,6,3,1,1,awbs[awb],fv,10)
      text(0,7,5,0,1,"Blue",ft,10)
      text(0,7,3,1,1,str(blue/10)[0:3],fv,10)
      text(0,8,5,0,1,"Red",ft,10)
      text(0,8,3,1,1,str(red/10)[0:3],fv,10)
      button(0,9,0,9) 
      text(0,9,1,0,1,"Page 2 ",ft,7)
      draw_bar(0,2,lgrnColor,'mode',mode)
      if mode == 0:
            draw_bar(0,2,lgrnColor,'speed',speed)  # Affiche la barre du shutter en mode manuel
      else:
            draw_bar(0,2,lgrnColor,'ev',ev)
      draw_bar(0,3,lgrnColor,'gain',gain)
      draw_bar(0,4,lgrnColor,'brightness',brightness)
      draw_bar(0,5,lgrnColor,'contrast',contrast)
      draw_bar(0,6,lgrnColor,'awb',awb)
      draw_bar(0,7,lgrnColor,'blue',blue)
      draw_bar(0,8,lgrnColor,'red',red)
                
    elif menu == 4:

        # Ligne 1 - Page 1 (retour)
        button(0,1,0,9)
        text(0,1,1,0,1,"Page 1 ",ft,7)

        # Ligne 2 - Metering
        text(0,2,5,0,1,"Metering",ft,10)
        text(0,2,3,1,1,meters[meter],fv,10)
        draw_bar(0,2,lgrnColor,'meter',meter)

        # Ligne 3 - Quality
        text(0,3,5,0,1,"Quality",ft,10)
        text(0,3,3,1,1,str(quality)[0:3],fv,10)
        draw_bar(0,3,lgrnColor,'quality',quality)

        # Ligne 4 - Saturation
        text(0,4,5,0,1,"Saturation",ft,10)
        text(0,4,3,1,1,str(saturation/10),fv,10)
        draw_bar(0,4,lgrnColor,'saturation',saturation)

        # Ligne 5 - Denoise
        text(0,5,5,0,1,"Denoise",ft,10)
        text(0,5,3,1,1,denoises[denoise],fv,10)
        draw_bar(0,5,lgrnColor,'denoise',denoise)

        # Ligne 6 - Sharpness
        text(0,6,5,0,1,"Sharpness",ft,10)
        text(0,6,3,1,1,str(sharpness/10),fv,10)
        draw_bar(0,6,lgrnColor,'sharpness',sharpness)

        # Ligne 7 - HDR / IR Filter / Scientific (selon caméra)
        if (Pi_Cam == 3 or Pi == 5):
            button(0,7,6,4)
            text(0,7,5,0,1,"HDR",ft,10)
            text(0,7,3,1,1,v3_hdrs[v3_hdr],fv,10)
        elif Pi_Cam == 9:
            button(0,7,6,4)
            text(0,7,5,0,1,"IR Filter",ft,10)
            if IRF == 0:
                text(0,7,3,1,1,"Off",fv,10)
            else:
                text(0,7,3,1,1,"ON ",fv,10)
        elif Pi_Cam == 4 and scientif == 1:
            button(0,7,6,4)
            text(0,7,5,0,1,"Scientific",ft,10)
            if scientific == 0:
                text(0,7,3,1,1,"Off",fv,10)
            else:
                text(0,7,3,1,1,"ON ",fv,10)

        # Ligne 8 - File Format
        text(0,8,5,0,1,"File Format",ft,10)
        text(0,8,3,1,1,extns[extn],fv,10)
        draw_bar(0,8,lgrnColor,'extn',extn)

        # Ligne 9 - SAVE CONFIG
        button(0,9,6,4)
        text(0,9,2,0,1,"SAVE CONFIG",fv,10)
      
    elif menu == 5:
        text(0,1,5,0,1,"V_Length",ft,11)
        td = timedelta(seconds=vlen)
        text(0,1,3,1,1,str(td),fv,11)
        text(0,2,5,0,1,"V_FPS",ft,11)
        text(0,2,3,1,1,str(fps),fv,11)
        text(0,3,5,0,1,"V_Format",ft,11)
        text(0,4,5,0,1,"V_Codec",ft,11)
        text(0,4,3,1,1,codecs[codec],fv,11)
        text(0,5,5,0,1,"h264 Profile",ft,11)
        text(0,5,3,1,1,str(h264profiles[profile]),fv,11)
        text(0,7,5,0,1,"V_Preview",ft,11)
        text(0,7,3,1,1,"ON ",fv,11)
        draw_Vbar(0,3,lpurColor,'vformat',vformat)
        text(0,8,2,0,1,"SAVE CONFIG",fv,11)
        # determine if camera native format
        vw = 0
        x = 0
        while x < len(vwidths2) and vw == 0:
            if vwidth == vwidths2[x]:
                if vheight == vheights2[x]:
                    vw = 1
            x += 1
        if vw == 0:
            text(0,3,3,1,1,str(vwidth) + "x" + str(vheight),fv,11)
        if vw == 1:
            text(0,3,1,1,1,str(vwidth) + "x" + str(vheight),fv,11)
        draw_Vbar(0,1,lpurColor,'vlen',vlen)
        draw_Vbar(0,2,lpurColor,'fps',fps)
        draw_Vbar(0,3,lpurColor,'vformat',vformat)
        draw_Vbar(0,4,lpurColor,'codec',codec)
        draw_Vbar(0,5,lpurColor,'profile',profile)
      
    elif menu == 6:
        td = timedelta(seconds=tduration)
        text(0,1,5,0,1,"Duration",ft,12)
        text(0,1,3,1,1,str(td),fv,12)
        td = timedelta(seconds=tinterval)
        text(0,2,5,0,1,"Interval",ft,12)
        text(0,2,3,1,1,str(td),fv,12)
        text(0,3,5,0,1,"No. of Shots",ft,12)
        if tinterval > 0:
            text(0,3,3,1,1,str(tshots),fv,12)
        else:
            text(0,3,3,1,1," ",fv,12)
        text(0,8,2,0,1,"SAVE CONFIG",fv,12)
        draw_Vbar(0,1,lyelColor,'tduration',tduration)
        draw_Vbar(0,2,lyelColor,'tinterval',tinterval)
        draw_Vbar(0,3,lyelColor,'tshots',tshots)

    elif menu == 7:
        # STRETCH Settings
        # Ligne 1 - Stretch Low %
        text(0,1,5,0,1,"Stretch Low %",ft,7)
        text(0,1,3,1,1,str(stretch_p_low/10)[0:4],fv,7)
        draw_Vbar(0,1,greyColor,'stretch_p_low',stretch_p_low)

        # Ligne 2 - Stretch High %
        text(0,2,5,0,1,"Stretch High %",ft,7)
        text(0,2,3,1,1,str(stretch_p_high/10)[0:5],fv,7)
        draw_Vbar(0,2,greyColor,'stretch_p_high',stretch_p_high)

        # Ligne 3 - Stretch Factor
        text(0,3,5,0,1,"Stretch Factor",ft,7)
        text(0,3,3,1,1,str(stretch_factor/10)[0:4],fv,7)
        draw_Vbar(0,3,greyColor,'stretch_factor',stretch_factor)

        # Ligne 4 - Preset
        text(0,4,5,0,1,"Preset",ft,7)
        text(0,4,3,1,1,stretch_presets[stretch_preset],fv,7)
        draw_Vbar(0,4,greyColor,'stretch_preset',stretch_preset)

        # Ligne 8 - SAVE CONFIG
        text(0,8,2,0,1,"SAVE CONFIG",fv,7)

        # Ligne 9 - Retour
        button(0,9,0,9)
        text(0,9,1,0,1,"CAMERA",ft,7)
        text(0,9,1,1,1,"Settings",ft,7)

    elif menu == 8:
        # LIVE STACK Settings
        # Ligne 1 - Preview Refresh
        text(0,1,5,0,1,"Preview Refresh",ft,7)
        text(0,1,3,1,1,str(ls_preview_refresh),fv,7)
        draw_Vbar(0,1,greyColor,'ls_preview_refresh',ls_preview_refresh)

        # Ligne 2 - Alignment Mode
        text(0,2,5,0,1,"Alignment Mode",ft,7)
        text(0,2,3,1,1,ls_alignment_modes[ls_alignment_mode],fv,7)
        draw_Vbar(0,2,greyColor,'ls_alignment_mode',ls_alignment_mode)

        # Ligne 3 - Max FWHM
        text(0,3,5,0,1,"Max FWHM",ft,7)
        if ls_max_fwhm == 0:
            text(0,3,3,1,1,"OFF",fv,7)
        else:
            text(0,3,3,1,1,str(ls_max_fwhm/10)[0:4],fv,7)
        draw_Vbar(0,3,greyColor,'ls_max_fwhm',ls_max_fwhm)

        # Ligne 4 - Min Sharpness
        text(0,4,5,0,1,"Min Sharpness",ft,7)
        if ls_min_sharpness == 0:
            text(0,4,3,1,1,"OFF",fv,7)
        else:
            text(0,4,3,1,1,str(ls_min_sharpness/1000)[0:5],fv,7)
        draw_Vbar(0,4,greyColor,'ls_min_sharpness',ls_min_sharpness)

        # Ligne 5 - Max Drift
        text(0,5,5,0,1,"Max Drift",ft,7)
        if ls_max_drift == 0:
            text(0,5,3,1,1,"OFF",fv,7)
        else:
            text(0,5,3,1,1,str(ls_max_drift),fv,7)
        draw_Vbar(0,5,greyColor,'ls_max_drift',ls_max_drift)

        # Ligne 6 - Min Stars
        text(0,6,5,0,1,"Min Stars",ft,7)
        if ls_min_stars == 0:
            text(0,6,3,1,1,"OFF",fv,7)
        else:
            text(0,6,3,1,1,str(ls_min_stars),fv,7)
        draw_Vbar(0,6,greyColor,'ls_min_stars',ls_min_stars)

        # Ligne 7 - Enable QC
        text(0,7,5,0,1,"Quality Control",ft,7)
        if ls_enable_qc == 0:
            text(0,7,3,1,1,"OFF",fv,7)
        else:
            text(0,7,3,1,1,"ON",fv,7)
        draw_Vbar(0,7,greyColor,'ls_enable_qc',ls_enable_qc)

        # Ligne 8 - SAVE CONFIG
        text(0,8,2,0,1,"SAVE CONFIG",fv,7)

        # Ligne 9 - Retour
        button(0,9,0,9)
        text(0,9,1,0,1,"CAMERA",ft,7)
        text(0,9,1,1,1,"Settings",ft,7)

text(0,0,6,2,1,"Please Wait, checking camera",int(fv* 1.7),1)
text(0,0,6,2,1,"Found " + str(cameras[Pi_Cam]),int(fv*1.7),1)

Menu()

time.sleep(1)
pygame.display.update()

# determine max speed for camera
max_speed = 0
while max_shutter > shutters[max_speed]:
    max_speed +=1
if speed > max_speed:
    speed = max_speed
    shutter = shutters[speed]
    if shutter < 0:
        shutter = abs(1/shutter)
    sspeed = int(shutter * 1000000)
    # Les lignes suivantes sont commentées car elles dessinent prématurément
    # un slider avant l'affichage correct du menu, créant un artefact visuel
    # if mode == 0:
    #     if shutters[speed] < 0:
    #         text(0,2,3,1,1,"1/" + str(abs(shutters[speed])),fv,10)
    #     else:
    #         text(0,2,3,1,1,str(shutters[speed]),fv,10)
    # else:
    #     if shutters[speed] < 0:
    #         text(0,2,0,1,1,"1/" + str(abs(shutters[speed])),fv,10)
    #     else:
    #         text(0,2,0,1,1,str(shutters[speed]),fv,10)
# if mode == 0:
#     draw_bar(0,2,lgrnColor,'speed',speed)
pygame.display.update()
time.sleep(.25)

xx = int(preview_width/2)
xy = int(preview_height/2)

fxx = 0
fxy = 0
fxz = 1
fyz = 1
old_histarea = histarea

# start preview
text(0,0,6,2,1,"Please Wait for preview...",int(fv*1.7),1)
preview()

# Créer une image noire par défaut pour éviter NameError au premier passage
if igw/igh > 1.5:
    image = pygame.Surface((preview_width, int(preview_height * 0.75)))
else:
    image = pygame.Surface((preview_width, preview_height))
image.fill((0, 0, 0))  # Remplir en noir

# main loop
while True:
    time.sleep(0.01)
    # focus UP button
    if (Pi_Cam == 3 and v3_af == 1) or Pi_Cam == 8 or ((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6):
      if buttonFUP.is_pressed:
        if v3_f_mode != 1:
            focus_mode = 1
            v3_f_mode = 1 # manual focus
            foc_man = 1
            if menu == 0:
                button(0,7,1,9)
                text(0,7,3,1,1,str(v3_f_modes[v3_f_mode]),fv,0)
        v3_focus += 10
        for f in range(0,len(video_limits)-1,3):
          if video_limits[f] == 'v3_focus':
            v3_pmin = video_limits[f+1]
            v3_pmax = video_limits[f+2]
        v3_focus = min(v3_focus,v3_pmax)
        focus = v3_focus
        if Pi_Cam == 3 and menu == 0:
            draw_Vbar(0,7,dgryColor,'v3_focus',v3_focus * 4)
            fd = 1/(v3_focus/100)
            text(0,7,3,0,1,'<<< ' + str(fd)[0:5] + "m" + ' >>>',fv,0)
        elif menu == 0:
            draw_Vbar(0,7,dgryColor,'v3_focus',v3_focus)
            text(0,7,3,0,1,'<<< ' + str(v3_focus) + ' >>>',fv,0)

        # ===== MODE PICAMERA2 : Changement focus en temps réel =====
        if use_picamera2 and picam2 is not None:
            try:
                picam2.set_controls({
                    "AfMode": controls.AfModeEnum.Manual,
                    "LensPosition": v3_focus / 100 if Pi_Cam == 3 else focus / 100
                })
            except:
                pass
        # ===== MODE RPICAM-VID : Redémarrer le subprocess =====
        else:
            kill_preview_process()
            text(0,0,6,2,1,"Waiting for preview ...",int(fv*1.7),1)
            preview()

    # focus DOWN button
    if (Pi_Cam == 3 and v3_af == 1) or Pi_Cam == 8 or ((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6):
      if buttonFDN.is_pressed:
        if v3_f_mode != 1:
            focus_mode = 1
            v3_f_mode = 1 # manual focus
            foc_man = 1
            if menu == 0:
                button(0,7,1,9)
                text(0,7,3,1,1,str(v3_f_modes[v3_f_mode]),fv,0)
        v3_focus -= 10
        for f in range(0,len(video_limits)-1,3):
          if video_limits[f] == 'v3_focus':
            v3_pmin = video_limits[f+1]
            v3_pmax = video_limits[f+2]
        v3_focus = max(v3_focus,v3_pmin)
        focus = v3_focus
        if Pi_Cam == 3 and menu == 0:
            draw_Vbar(0,7,dgryColor,'v3_focus',v3_focus * 4)
            fd = 1/(v3_focus/100)
            text(0,7,3,0,1,'<<< ' + str(fd)[0:5] + "m" + ' >>>',fv,0)
        elif menu == 0:
            draw_Vbar(0,7,dgryColor,'v3_focus',v3_focus)
            text(0,7,3,0,1,'<<< ' + str(v3_focus) + ' >>>',fv,0)

        # ===== MODE PICAMERA2 : Changement focus en temps réel =====
        if use_picamera2 and picam2 is not None:
            try:
                picam2.set_controls({
                    "AfMode": controls.AfModeEnum.Manual,
                    "LensPosition": v3_focus / 100 if Pi_Cam == 3 else focus / 100
                })
            except:
                pass
        # ===== MODE RPICAM-VID : Redémarrer le subprocess =====
        else:
            kill_preview_process()
            text(0,0,6,2,1,"Waiting for preview ...",int(fv*1.7),1)
            preview()    

       
    # ===== CAPTURE AVEC PICAMERA2 =====
    if use_picamera2 and picam2 is not None:
        # DEBUG: afficher une fois qu'on rentre dans cette section
        if not hasattr(pygame, '_picam2_section_debug'):
            print("DEBUG: Entering Picamera2 capture section")
            pygame._picam2_section_debug = True
        try:
            # Capturer l'image directement depuis Picamera2
            array = picam2.capture_array("main")

            # DEBUG: afficher une fois au démarrage
            if not hasattr(pygame, '_picam2_debug_done'):
                print(f"DEBUG: Array shape: {array.shape}, dtype: {array.dtype}")
                # Sauvegarder une frame de test pour vérifier
                import cv2
                cv2.imwrite('/home/admin/debug_picamera2_frame.jpg', cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
                print("DEBUG: Frame saved to /home/admin/debug_picamera2_frame.jpg")
                pygame._picam2_debug_done = True

            # ===== LIVE STACK PROCESSING =====
            livestack_display_done = False
            if livestack_active and livestack is not None:
                # Traiter la frame avec LiveStack
                preview_updated = livestack.process_frame(array)

                # Si le preview est rafraîchi, utiliser l'image stackée
                if preview_updated:
                    # TOUJOURS utiliser le résultat brut (haute précision float 0-65535)
                    # Le stretch sera appliqué par le programme principal si activé
                    stacked_array = livestack.session.get_current_stack()

                    if stacked_array is not None:
                        # Convertir float (0-65535) en uint8
                        stacked_array = (stacked_array / 256).astype(np.uint8)

                        # Appliquer le stretch du programme principal si activé
                        if stretch_mode == 1 and stretch_preset != 0:
                            stacked_array = astro_stretch(stacked_array)

                        # Convertir en surface pygame
                        # last_preview est en RGB (H, W, 3) ou MONO (H, W)
                        if len(stacked_array.shape) == 3:
                            # RGB: transposer et échanger R/B pour pygame
                            image = pygame.surfarray.make_surface(
                                np.swapaxes(stacked_array, 0, 1)[:,:,[2,1,0]]
                            )
                        else:
                            # MONO
                            image = pygame.surfarray.make_surface(stacked_array.T)

                        # Redimensionner en fullscreen si stretch activé, sinon preview normal
                        if stretch_mode == 1:
                            # Mode fullscreen (comme pour stretch normal)
                            display_modes = pygame.display.list_modes()
                            if display_modes and display_modes != -1:
                                max_width, max_height = display_modes[0]
                            else:
                                screen_info = pygame.display.Info()
                                max_width, max_height = screen_info.current_w, screen_info.current_h
                            image = pygame.transform.scale(image, (max_width, max_height))
                        elif image.get_width() != preview_width or image.get_height() != preview_height:
                            image = pygame.transform.scale(image, (preview_width, preview_height))

                        # Afficher
                        windowSurfaceObj.blit(image, (0, 0))

                        # Afficher les statistiques Live Stack
                        stats = livestack.get_stats()
                        stats_text = f"LiveStack: {stats['accepted']}/{stats['total_frames']} | Rejected: {stats['rejected']} | SNR: x{stats['snr_gain']:.1f}"
                        text(0, 0, 2, 0, 1, stats_text, ft, 1)

                        # Marquer que l'affichage est fait
                        livestack_display_done = True

            # Traitement normal seulement si LiveStack n'a pas déjà affiché
            if not livestack_display_done:
                # Appliquer le stretch astro si le mode est activé ET que le preset n'est pas OFF
                if stretch_mode == 1 and stretch_preset != 0:
                    array = astro_stretch(array)

                # Convertir numpy array → pygame surface
                # Picamera2 retourne (height, width, 3) en RGB
                # pygame.surfarray.make_surface attend (width, height, 3)
                # MAIS pygame interprète les couleurs comme (R,G,B) dans l'ordre (0,1,2)
                # On doit transposer width/height ET échanger R et B pour pygame
                image = pygame.surfarray.make_surface(np.swapaxes(array, 0, 1)[:,:,[2,1,0]])

                # Scaling si nécessaire
                if stretch_mode == 1:
                    # En mode stretch, afficher en VRAI plein écran (cache la barre de tâches)
                    # Obtenir la résolution maximale de l'écran
                    display_modes = pygame.display.list_modes()
                    if display_modes and display_modes != -1:
                        # Prendre la résolution la plus grande (la première de la liste)
                        max_width, max_height = display_modes[0]
                    else:
                        # Fallback si list_modes ne fonctionne pas
                        screen_info = pygame.display.Info()
                        max_width, max_height = screen_info.current_w, screen_info.current_h
                    image = pygame.transform.scale(image, (max_width, max_height))
                elif image.get_width() != preview_width or image.get_height() != preview_height:
                    if igw/igh > 1.5:
                        image = pygame.transform.scale(image, (preview_width, int(preview_height * 0.75)))
                    else:
                        image = pygame.transform.scale(image, (preview_width, preview_height))

                # Affichage
                windowSurfaceObj.blit(image, (0, 0))

        except Exception as e:
            # En cas d'erreur, afficher dans la console
            import traceback
            if show_cmds == 1:
                print(f"Erreur capture Picamera2: {e}")
                traceback.print_exc()
            pass

    # ===== CAPTURE AVEC RPICAM-VID (méthode originale) =====
    else:
        # MÉTHODE OPTIMISÉE POUR FLUIDITÉ
        # Charger l'image la plus récente (pics[0]) pour réduire le délai
        # Garder un petit buffer (3 images) pour éviter suppressions constantes
        pics = glob.glob('/run/shm/*.jpg')
        if len(pics) > 0:
            # Tri alphabétique descendant (test0003, test0002, test0001, test0000)
            pics.sort(reverse=True)
            try:
                # Charger la DERNIÈRE image (pics[0]) pour minimiser le délai
                # Avec --segment 1, les fichiers sont écrits très rapidement
                image = pygame.image.load(pics[0])

                # Garder seulement les 3 images les plus récentes (réduit les I/O)
                # Au lieu de tout supprimer à chaque fois
                if len(pics) > 3:
                    for tt in range(3, len(pics)):
                        try:
                            os.remove(pics[tt])
                        except OSError:
                            pass  # Ignorer si déjà supprimé
            except (pygame.error, OSError):
                # Si pics[0] échoue (en cours d'écriture), essayer pics[1]
                try:
                    if len(pics) > 1:
                        image = pygame.image.load(pics[1])
                except (pygame.error, OSError):
                    pass  # Garder l'ancienne image affichée

            # Appliquer le stretch astro si le mode est activé ET que le preset n'est pas OFF
            if stretch_mode == 1 and stretch_preset != 0:
                # Convertir pygame surface → numpy array
                img_array = pygame.surfarray.array3d(image)
                # Transposer de (width, height, channels) à (height, width, channels)
                img_array = np.transpose(img_array, (1, 0, 2))
                # Appliquer le stretch
                img_array = astro_stretch(img_array)
                # Reconvertir en pygame surface
                image = pygame.surfarray.make_surface(np.swapaxes(img_array, 0, 1))

            # Scaling et affichage
            # En mode zoom, l'image arrive déjà à la bonne taille via ROI, ne pas rescaler
            if stretch_mode == 1:
                # En mode stretch, afficher en VRAI plein écran (cache la barre de tâches)
                # Obtenir la résolution maximale de l'écran
                display_modes = pygame.display.list_modes()
                if display_modes and display_modes != -1:
                    # Prendre la résolution la plus grande (la première de la liste)
                    max_width, max_height = display_modes[0]
                else:
                    # Fallback si list_modes ne fonctionne pas
                    screen_info = pygame.display.Info()
                    max_width, max_height = screen_info.current_w, screen_info.current_h
                image = pygame.transform.scale(image, (max_width, max_height))
            elif zoom == 0:
                if igw/igh > 1.5:
                    image = pygame.transform.scale(image, (preview_width,int(preview_height * 0.75)))
                else:
                    image = pygame.transform.scale(image, (preview_width,preview_height))
            windowSurfaceObj.blit(image, (0,0))
    # Ne pas afficher les overlays en mode stretch
    if (zoom > 0 or foc_man == 1 or focus_mode == 1 or histogram > 0) and stretch_mode == 0:
        # Utiliser array3d au lieu de pixels3d pour ne pas verrouiller la surface
        # Cela améliore grandement la fluidité de l'affichage en mode focus et histogram
        image2 = pygame.surfarray.array3d(image)
        # Transposer car array3d retourne (width, height, channels) au lieu de (height, width, channels)
        image2_transposed = np.transpose(image2, (1, 0, 2))
        crop2 = image2_transposed[xy-histarea:xy+histarea,xx-histarea:xx+histarea]
        gray = cv2.cvtColor(crop2,cv2.COLOR_RGB2GRAY)
        
        # Histogramme OPTIMISÉ avec numpy vectorisé (80-95% plus rapide)
        # Ne pas afficher l'histogramme en mode focus
        if histogram > 0 and focus_mode == 0:
            # Calculer les histogrammes avec numpy (100-1000x plus rapide que les boucles Python)
            bins = np.arange(257)  # 0-256 pour np.histogram

            # Initialiser les histogrammes
            rede = greene = bluee = lume = None

            if histogram == 1 or histogram == 5:
                rede, _ = np.histogram(crop2[:,:,0].ravel(), bins=bins)
            if histogram == 2 or histogram == 5:
                greene, _ = np.histogram(crop2[:,:,1].ravel(), bins=bins)
            if histogram == 3 or histogram == 5:
                bluee, _ = np.histogram(crop2[:,:,2].ravel(), bins=bins)
            if histogram == 4 or histogram == 5:
                lume, _ = np.histogram(gray.ravel(), bins=bins)

            # Normalisation linéaire vectorisée (0-100) - évite les warnings et plus robuste
            if lume is not None:
                max_val = lume.max()
                lume = (lume / max_val * 100).astype(int) if max_val > 0 else lume
            if rede is not None:
                max_val = rede.max()
                rede = (rede / max_val * 100).astype(int) if max_val > 0 else rede
            if greene is not None:
                max_val = greene.max()
                greene = (greene / max_val * 100).astype(int) if max_val > 0 else greene
            if bluee is not None:
                max_val = bluee.max()
                bluee = (bluee / max_val * 100).astype(int) if max_val > 0 else bluee

            # Calculer la hauteur du bandeau noir pour utiliser toute la hauteur disponible
            hist_y = int(preview_height * 0.75) + 1
            hist_height = preview_height - hist_y - 2  # -2 pour une petite marge en bas
            # L'histogramme occupe toute la largeur de la zone preview (sans le menu)
            hist_width = preview_width

            # Créer le graphique DIRECTEMENT à la largeur cible (sans scaling)
            # Cela évite les marges natives du scaling pygame
            output = np.zeros((hist_width, hist_height, 3), dtype=np.uint8)

            # Normaliser les valeurs pour la hauteur disponible
            scale_factor = hist_height / 100  # Adapter l'échelle à la nouvelle hauteur

            # Calculer la largeur de chaque bin (256 bins répartis sur hist_width pixels)
            bin_width = hist_width / 256.0

            # Dessiner chaque courbe directement à la bonne largeur
            # IMPORTANT: Dessiner les 256 bins (0-255) pour couvrir toute la largeur
            for i in range(256):
                # Calculer la position x pour ce bin
                x_start = int(i * bin_width)
                x_end = int((i + 1) * bin_width)

                # S'assurer que le dernier bin atteint bien le bord droit
                if i == 255:
                    x_end = hist_width

                if lume is not None and i < len(lume) and lume[i] > 0:
                    if i < 255:
                        y_start = min(int(lume[i] * scale_factor), int(lume[i+1] * scale_factor))
                        y_end = max(int(lume[i] * scale_factor), int(lume[i+1] * scale_factor))
                    else:
                        y_start = y_end = int(lume[i] * scale_factor)
                    output[x_start:x_end, y_start:min(y_end+1, hist_height), :] = 255  # Blanc pour luminance

                if rede is not None and i < len(rede) and rede[i] > 0:
                    if i < 255:
                        y_start = min(int(rede[i] * scale_factor), int(rede[i+1] * scale_factor))
                        y_end = max(int(rede[i] * scale_factor), int(rede[i+1] * scale_factor))
                    else:
                        y_start = y_end = int(rede[i] * scale_factor)
                    output[x_start:x_end, y_start:min(y_end+1, hist_height), 0] = 255  # Rouge

                if greene is not None and i < len(greene) and greene[i] > 0:
                    if i < 255:
                        y_start = min(int(greene[i] * scale_factor), int(greene[i+1] * scale_factor))
                        y_end = max(int(greene[i] * scale_factor), int(greene[i+1] * scale_factor))
                    else:
                        y_start = y_end = int(greene[i] * scale_factor)
                    output[x_start:x_end, y_start:min(y_end+1, hist_height), 1] = 255  # Vert

                if bluee is not None and i < len(bluee) and bluee[i] > 0:
                    if i < 255:
                        y_start = min(int(bluee[i] * scale_factor), int(bluee[i+1] * scale_factor))
                        y_end = max(int(bluee[i] * scale_factor), int(bluee[i+1] * scale_factor))
                    else:
                        y_start = y_end = int(bluee[i] * scale_factor)
                    output[x_start:x_end, y_start:min(y_end+1, hist_height), 2] = 255  # Bleu

            graph = pygame.surfarray.make_surface(output)
            graph = pygame.transform.flip(graph, 0, 1)
            graph.set_alpha(180)  # Légèrement plus opaque pour meilleure lisibilité
            # Plus besoin de scaling - déjà à la bonne taille !
            # DEBUG: Cadre rouge pour visualiser les limites de l'histogramme
            pygame.draw.rect(windowSurfaceObj, (255, 0, 0), Rect(0, hist_y, hist_width, hist_height), 2)
            # Afficher l'histogramme sur toute la largeur sans cadre
            windowSurfaceObj.blit(graph, (0, hist_y))
        
        # Nettoyage des variables numpy (array3d ne crée pas de verrou)
        del image2
        del image2_transposed
        del crop2
        
        # Calcul du focus Laplacian avec indicateur de qualité coloré
        foc = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Déterminer la couleur du texte selon la qualité du focus
        # Seuils typiques pour astrophotographie (dépendent de la taille de la zone)
        focus_color = 0  # couleur par défaut (gris foncé)
        if foc > 500:
            focus_color = 1  # vert - excellent focus
        elif foc > 200:
            focus_color = 2  # jaune - bon focus
        elif foc > 50:
            focus_color = 3  # rouge - focus moyen/mauvais
        else:
            focus_color = 3  # rouge - mauvais focus
        
        text(20,1,focus_color,2,0,"Focus: " + str(int(foc)),fv* 2,0)
        
        # Calcul et affichage du SNR - convertir la surface pygame en array numpy
        try:
            # Convertir la surface pygame en array pour le calcul SNR
            image_array = pygame.surfarray.array3d(image)
            # Transposer car pygame utilise (width, height, channels) et opencv utilise (height, width, channels)
            image_array = np.transpose(image_array, (1, 0, 2))
            snr_value = calculate_snr(image_array)
            
            # Seuils pour ratio linéaire (adapté à l'astrophotographie)
            snr_color = 0  # couleur par défaut (gris foncé)
            if snr_value > 20:
                snr_color = 1  # vert si excellent SNR
            elif snr_value > 10:
                snr_color = 2  # jaune si bon SNR
            elif snr_value > 5:
                snr_color = 2  # jaune si SNR moyen
            else:
                snr_color = 3  # rouge si mauvais SNR
            
            # Affichage au format ratio (ex: 15.2:1)
            text(20,2,snr_color,2,0,"SNR = " + str(round(snr_value, 1)) + ":1",fv* 2,0)
        except:
            text(20,2,0,2,0,"SNR = N/A",fv* 2,0)
        
        # *** HFR : Calcul et affichage permanent (robuste aux aigrettes) ***
        if focus_mode == 1 or histogram > 0:
            # Utiliser la même zone que le réticule (histarea) pour cohérence avec FWHM
            # L'algorithme avec seuil à 20% filtre le bruit efficacement
            hfr_val = calculate_hfr(image, xx, xy, histarea)

            if hfr_val is not None:
                # Déterminer la couleur selon la qualité HFR
                # HFR en pixels : plus petit = meilleur
                # Valeurs typiques pour astrophotographie
                if hfr_val < 2:
                    hfr_color = 1  # vert (excellent - étoile très concentrée)
                elif hfr_val < 3.5:
                    hfr_color = 2  # jaune (bon)
                elif hfr_val < 5:
                    hfr_color = 2  # jaune (moyen)
                else:
                    hfr_color = 3  # rouge (mauvais - étoile diffuse)

                # Affichage texte HFR - position adaptée selon le mode
                # En preview: ligne 3 (juste sous SNR), en focus: ligne 4 (sous FWHM)
                hfr_line = 3 if focus_mode == 0 else 4
                text(20, hfr_line, hfr_color, 2, 0, "HFR: " + str(round(hfr_val, 2)), fv * 2, 0)
            else:
                # Afficher "N/A" si pas d'étoile détectée
                hfr_line = 3 if focus_mode == 0 else 4
                text(20, hfr_line, 0, 2, 0, "HFR: N/A", fv * 2, 0)

        # *** Éléments affichés SEULEMENT en mode focus ***
        if focus_mode == 1:
            # FWHM
            fwhm_val = calculate_fwhm(image, xx, xy, histarea)

            if fwhm_val is not None:
                # Déterminer la couleur selon la qualité FWHM
                # Note: FWHM est en pixels, pas en arcsec
                # Les seuils dépendent de la taille de histarea
                # Pour histarea=50, des valeurs typiques sont 2-20 pixels
                if fwhm_val < 5:
                    fwhm_color = 1  # vert (excellente - étoile très fine)
                elif fwhm_val < 10:
                    fwhm_color = 2  # jaune (bonne)
                elif fwhm_val < 20:
                    fwhm_color = 2  # jaune (moyenne)
                else:
                    fwhm_color = 3  # rouge (mauvaise - étoile très large)

                # Affichage texte FWHM
                text(20, 3, fwhm_color, 2, 0, "FWHM: " + str(round(fwhm_val, 1)), fv * 2, 0)
            else:
                # Afficher "N/A" si pas d'étoile détectée
                text(20, 3, 0, 2, 0, "FWHM: N/A", fv * 2, 0)

            # En mode focus : afficher 2 graphiques élégants dans le bandeau noir
            # Graphique 1 (gauche) : HFR + FWHM combinés
            # Graphique 2 (droite) : Focus (Laplacian variance)
            try:
                graph_y = int(preview_height * 0.75) + 1
                graph_height = preview_height - graph_y - 2

                # Largeur de chaque graphique = moitié du bandeau moins marges
                graph_width = int((preview_width - 40) / 2)  # -40 pour marges (10+10+10+10)

                # Graphique HFR+FWHM combiné (gauche)
                hfr_fwhm_surface = update_combined_hfr_fwhm_graph(hfr_val, fwhm_val)
                if hfr_fwhm_surface is not None and alt_dis < 2:
                    graph1_x = 10
                    graph1_resized = pygame.transform.scale(hfr_fwhm_surface, (graph_width, graph_height))

                    # Cadre élégant avec couleur gradient
                    pygame.draw.rect(windowSurfaceObj, (100, 255, 200),
                                   Rect(graph1_x - 2, graph_y - 2,
                                        graph1_resized.get_width() + 4,
                                        graph1_resized.get_height() + 4), 2)

                    windowSurfaceObj.blit(graph1_resized, (graph1_x, graph_y))

                # Graphique Focus (droite)
                focus_surface = update_focus_graph(foc)
                if focus_surface is not None and alt_dis < 2:
                    graph2_x = 10 + graph_width + 20  # Après le premier graphique + marge
                    graph2_resized = pygame.transform.scale(focus_surface, (graph_width, graph_height))

                    # Cadre élégant avec couleur gradient
                    pygame.draw.rect(windowSurfaceObj, (100, 200, 255),
                                   Rect(graph2_x - 2, graph_y - 2,
                                        graph2_resized.get_width() + 4,
                                        graph2_resized.get_height() + 4), 2)

                    windowSurfaceObj.blit(graph2_resized, (graph2_x, graph_y))

            except Exception as e:
                pass  # Ignorer les erreurs des graphiques

            # Rectangle rouge et croix d'analyse
            pygame.draw.rect(windowSurfaceObj,redColor,Rect(xx-histarea,xy-histarea,histarea*2,histarea*2),1)
            pygame.draw.line(windowSurfaceObj,(255,255,255),(xx-int(histarea/2),xy),(xx+int(histarea/2),xy),1)
            pygame.draw.line(windowSurfaceObj,(255,255,255),(xx,xy-int(histarea/2)),(xx,xy+int(histarea/2)),1)
    
    # Mode preview (zoom == 0 ET focus_mode == 0) - Ne pas afficher en mode stretch
    if zoom == 0 and focus_mode == 0 and stretch_mode == 0:
        text(0,0,6,2,0,"Preview",fv* 2,0)

    # Mode preview (zoom == 0)
    if zoom == 0:
        #pygame.draw.rect(windowSurfaceObj,blackColor,Rect(0,0,int(preview_width/4.5),int(preview_height/8)),0)
        # Ne pas afficher le texte en mode stretch
        if stretch_mode == 0:
            text(0,0,6,2,0,"Preview",fv* 2,0)
        zxp = (zx -((preview_width/2) / (igw/preview_width)))
        zyp = (zy -((preview_height/2) / (igh/preview_height)))
        zxq = (zx - zxp) * 2
        zyq = (zy - zyp) * 2
        if zxp + zxq > preview_width:
            zx = preview_width - int(zxq/2)
            zxp = (zx -((preview_width/2) / (igw/preview_width)))
            zxq = (zx - zxp) * 2
        if zyp + zyq > preview_height:
            zy = preview_height - int(zyq/2)
            zyp = (zy -((preview_height/2) / (igh/preview_height)))
            zyq = (zy - zyp) * 2
        if zxp < 0:
            zx = int(zxq/2) + 1
            zxp = 0
            zxq = (zx - zxp) * 2
        if zyp < 0:
            zy = int(zyq/2) + 1
            zyp = 0
            zyq = (zy - zyp) * 2
        if preview_width < 800:
            gw = 2
        else:
            gw = 1
        # Ne pas afficher le rectangle de focus en mode stretch (déjà géré plus bas)
        if ((Pi_Cam == 3 and v3_af == 1) or ((Pi_Cam == 5 or Pi_Cam == 6)) or Pi_Cam == 8) and fxz != 1 and zoom == 0 and stretch_mode == 0:
            pygame.draw.rect(windowSurfaceObj,(200,0,0),Rect(int(fxx*preview_width),int(fxy*preview_height*.75),int(fxz*preview_width),int(fyz*preview_height)),1)

    # *** CRUCIAL : Mettre à jour l'affichage pygame ***
    pygame.display.update()
    
    # *** CETTE PARTIE EST CRUCIALE : Mode preview (zoom == 0) ***
    if zoom == 0:
        #pygame.draw.rect(windowSurfaceObj,blackColor,Rect(0,0,int(preview_width/4.5),int(preview_height/8)),0)
        # Ne pas afficher le texte en mode stretch
        if stretch_mode == 0:
            text(0,0,6,2,0,"Preview",fv* 2,0)
        zxp = (zx -((preview_width/2) / (igw/preview_width)))
        zyp = (zy -((preview_height/2) / (igh/preview_height)))
        zxq = (zx - zxp) * 2
        zyq = (zy - zyp) * 2
        if zxp + zxq > preview_width:
            zx = preview_width - int(zxq/2)
            zxp = (zx -((preview_width/2) / (igw/preview_width)))
            zxq = (zx - zxp) * 2
        if zyp + zyq > preview_height:
            zy = preview_height - int(zyq/2)
            zyp = (zy -((preview_height/2) / (igh/preview_height)))
            zyq = (zy - zyp) * 2
        if zxp < 0:
            zx = int(zxq/2) + 1
            zxp = 0
            zxq = (zx - zxp) * 2
        if zyp < 0:
            zy = int(zyq/2) + 1
            zyp = 0
            zyq = (zy - zyp) * 2
        if preview_width < 800:
            gw = 2
        else:
            gw = 1
        # Ne pas afficher le rectangle de focus en mode stretch
        if ((Pi_Cam == 3 and v3_af == 1) or ((Pi_Cam == 5 or Pi_Cam == 6)) or Pi_Cam == 8) and fxz != 1 and zoom == 0 and stretch_mode == 0:
            pygame.draw.rect(windowSurfaceObj,(200,0,0),Rect(int(fxx*preview_width),int(fxy*preview_height*.75),int(fxz*preview_width),int(fyz*preview_height)),1)

        pygame.display.update()

    if buttonSTR.is_pressed:
        type = pygame.MOUSEBUTTONUP
        if str_cap == 2:
            click_event = pygame.event.Event(type, {"button": 3, "pos": (0,0)})
        else:
            click_event = pygame.event.Event(type, {"button": 1, "pos": (0,0)})
        pygame.event.post(click_event)
    
    #check for any mouse button presses
    for event in pygame.event.get():
      #QUIT
      if event.type == QUIT:
          # Arrêter proprement l'extracteur MJPEG
          if mjpeg_extractor is not None:
              mjpeg_extractor.stop()
          if not use_picamera2 and p is not None:
              os.killpg(p.pid, signal.SIGTERM)
          pygame.quit()
      # MOVE HISTAREA
      elif (event.type == MOUSEBUTTONUP):
        mousex, mousey = event.pos

        # Si on est en mode LiveStack actif, un clic quitte ce mode
        if livestack_active:
            livestack_active = False
            stretch_mode = 0  # Désactiver aussi le stretch
            if livestack is not None:
                livestack.stop()
            print("[LIVESTACK] Mode désactivé (clic)")
            # Restaurer le mode d'affichage normal (avec l'interface)
            if frame == 1:
                if fullscreen == 1:
                    windowSurfaceObj = pygame.display.set_mode((preview_width + bw, dis_height), pygame.FULLSCREEN, 24)
                else:
                    windowSurfaceObj = pygame.display.set_mode((preview_width + bw, dis_height), 0, 24)
            else:
                windowSurfaceObj = pygame.display.set_mode((preview_width + bw, dis_height), pygame.NOFRAME, 24)
            # Effacer l'écran (remplir de noir)
            windowSurfaceObj.fill((0, 0, 0))
            # Redessiner le menu pour restaurer l'affichage normal
            Menu()
            pygame.display.update()
            continue

        # Si on est en mode stretch, un clic quitte ce mode
        if stretch_mode == 1:
            stretch_mode = 0
            # Restaurer le mode d'affichage normal (avec l'interface)
            if frame == 1:
                if fullscreen == 1:
                    windowSurfaceObj = pygame.display.set_mode((preview_width + bw, dis_height), pygame.FULLSCREEN, 24)
                else:
                    windowSurfaceObj = pygame.display.set_mode((preview_width + bw, dis_height), 0, 24)
            else:
                windowSurfaceObj = pygame.display.set_mode((preview_width + bw, dis_height), pygame.NOFRAME, 24)
            # Effacer l'écran (remplir de noir)
            windowSurfaceObj.fill((0, 0, 0))
            # Redessiner le menu pour restaurer l'affichage normal
            Menu()
            pygame.display.update()
            continue

        if mousex < preview_width and mousey < preview_height and mousex != 0 and mousey != 0 and event.button != 3 and menu == 0:
            xx = mousex
            xx = min(xx,preview_width - histarea)
            xx = max(xx,histarea)
            xy = mousey
            if igw/igh > 1.5 and zoom < 5:
                xy = min(xy,int(preview_height * .75) - histarea)
            else:
                xy = min(xy,preview_height - histarea)
            xy = max(xy,histarea)
            if ((Pi_Cam == 3 and v3_af == 1) or ((Pi_Cam == 5 or Pi_Cam == 6)) or Pi_Cam == 8) and mousex < preview_width and mousey < preview_height *.75 and zoom == 0 and (v3_f_mode == 0 or v3_f_mode == 2):
                fxx = (xx - 25)/preview_width
                xy  = min(xy,int((preview_height - 25) * .75))
                fxy = ((xy - 20) * 1.3333)/preview_height
                fxz = 50/preview_width
                fyz = fxz
                #if fxz != 1 and menu == 0:
                #    text(0,3,3,1,1,"Spot",fv,7)
            elif ((Pi_Cam == 3 and v3_af == 1) or ((Pi_Cam == 5 or Pi_Cam ==6)) or Pi_Cam == 8) and zoom == 0:
                fxx = 0
                fxy = 0
                fxz = 1
                fzy = 1
                if (v3_f_mode == 0 or v3_f_mode == 2) and menu == 0:
                    text(0,3,3,1,1,str(v3_f_modes[v3_f_mode]),fv,7)
            if ((Pi_Cam == 3 and v3_af == 1) or ((Pi_Cam == 5 or Pi_Cam == 6)) or Pi_Cam == 8) and zoom == 0:
                restart = 1
        
        # external trigger
        if mousex == 0 and mousey == 0:
            str_btn = 1
            
        # determine button pressed
        if mousex > preview_width or str_btn == 1:
            button_row = int(mousey/bh)
            if mousex > preview_width + bw/2:
                button_pos = 1
            else:
                button_pos = 0
                      
            # capture on STR button press
            if str_btn == 1:
                if str_cap == 0:
                    button_row = 0
                elif str_cap == 1 or str_cap == 2:
                    button_row = 1
                elif str_cap == 3:
                    button_row = 2
                str_btn = 0
              
            if button_row == 0 and menu > 0:
                menu = 0
                Menu()
            # MENU 0
            elif menu == 0: 
                if button_row == 0:
                    # TAKE STILL
                    still = 1
                    if not use_picamera2 and p is not None:
                        os.killpg(p.pid, signal.SIGTERM)
                        # Attendre que le processus de preview se termine complètement
                        poll = p.poll()
                        while poll == None:
                            poll = p.poll()
                            time.sleep(0.1)

                    button(0,0,1,4)
                    if os.path.exists("PiLibtext.txt"):
                         os.remove("PiLibtext.txt")
                    text(0,0,2,0,1,"    CAPTURING",ft,0)
                    text(0,0,6,2,1,"Please Wait, taking still ...",int(fv*1.7),1)
                    now = datetime.datetime.now()
                    timestamp = now.strftime("%y%m%d%H%M%S")
                    if extns[extn] != 'raw':
                        fname =  pic_dir + str(timestamp) + '.' + extns2[extn]
                        if lver != "bookwo" and lver != "trixie":
                            datastr = "libcamera-still"
                        else:
                            datastr = "rpicam-still"
                        datastr += " --camera " + str(camera) + " -e " + extns[extn] + " -n "
                        datastr += "-t " + str(timet) + " -o " + fname
                    else:
                        fname =  pic_dir + str(timestamp) + '.' + extns2[extn]
                        if lver != "bookwo" and lver != "trixie":
                            datastr = "libcamera-still"
                        else:
                             datastr = "rpicam-still"
                        datastr += " --camera " + str(camera) + " -r -n "
                        datastr += "-t " + str(timet) + " -o " + fname
                    datastr += " --brightness " + str(brightness/100) + " --contrast " + str(contrast/100) 
                    if mode == 0:
                        datastr += " --shutter " + str(sspeed)
                    else:
                        datastr += " --exposure " + str(modes[mode])
                    if ev != 0:
                        datastr += " --ev " + str(ev)
                    if sspeed > 1000000 and mode == 0:
                        datastr += " --gain " + str(gain) + " --immediate --awbgains " + str(red/10) + "," + str(blue/10)
                    else:    
                        datastr += " --gain " + str(gain)
                        if awb == 0:
                            datastr += " --awbgains " + str(red/10) + "," + str(blue/10)
                        else:
                            datastr += " --awb " + awbs[awb]
                    datastr += " --metering " + meters[meter]
                    datastr += " --saturation " + str(saturation/10)
                    datastr += " --sharpness " + str(sharpness/10)
                    datastr += " --quality " + str(quality)
                    if vflip == 1:
                        datastr += " --vflip"
                    if hflip == 1:
                        datastr += " --hflip"
                    datastr += " --denoise " + denoises[denoise]
                    if Pi_Cam == 9 and os.path.exists("/home/" + Home_Files[0] + "/imx290a.json") and Pi == 5:
                        datastr += " --tuning-file /home/" + Home_Files[0] + "/imx290a.json"
                    if Pi_Cam == 10 and os.path.exists("/home/" + Home_Files[0] + "/imx585_lowlight.json") and Pi == 5:
                        datastr += " --tuning-file /home/" + Home_Files[0] + "/imx585_lowlight.json"
                    if Pi_Cam == 4 and scientific == 1:
                        if os.path.exists('/usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json') and Pi == 4:
                            datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json"
                        if os.path.exists('/usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json') and Pi == 5:
                            datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json"
                    if ((Pi_Cam == 3 and v3_af == 1) and v3_f_mode > 0 and fxx == 0) or (((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6)) or Pi_Cam == 8:
                        datastr += " --autofocus-mode " + v3_f_modes[v3_f_mode]
                        if v3_f_mode == 1:
                            if Pi_Cam == 3:
                                datastr += " --lens-position " + str(v3_focus/100)
                            if Pi_Cam == 8 or ((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6):
                                datastr += " --lens-position " + str(focus/100)
                    elif (Pi_Cam == 3 and v3_af == 1) and v3_f_mode == 0 and fxz == 1:
                        datastr += " --autofocus-mode " + v3_f_modes[v3_f_mode] + " --autofocus-on-capture"
                    if ((Pi_Cam == 3 and v3_af == 1) or (((Pi_Cam == 5 and v5_af == 1)or Pi_Cam == 6)) or Pi_Cam == 8) and zoom == 0 and fxz != 1:
                        datastr += " --autofocus-window " + str(fxx) + "," + str(fxy) + "," + str(fxz) + "," + str(fxz)
                    if Pi_Cam == 3 or Pi == 5:
                        datastr += " --hdr " + v3_hdrs[v3_hdr]
                    if (Pi_Cam == 6 or Pi_Cam == 8) and mode == 0 and button_pos == 1:
                        datastr += " --width 4624 --height 3472 " # 16MP superpixel mode for higher light sensitivity
                    elif Pi_Cam == 6 or Pi_Cam == 8:
                        if Pi != 5 and lo_res == 1:
                            datastr += " --width 4624 --height 3472"
                        elif Pi_Cam == 6:
                            datastr += " --width 9152 --height 6944"
                        elif Pi_Cam == 8:
                            datastr += " --width 9248 --height 6944"
                    if zoom > 0 and zoom <= 5 and zoom != 3:  # Zoom fixe 1x à 6x (zoom 3 désactivé)
                        # Arrondir à un nombre PAIR pour compatibilité formats vidéo
                        zws = int(igw * zfs[zoom])
                        zhs = int(igh * zfs[zoom])
                        if zws % 2 != 0:
                            zws -= 1  # Forcer pair
                        if zhs % 2 != 0:
                            zhs -= 1  # Forcer pair
                        zxo = ((igw-zws)/2)/igw
                        zyo = ((igh-zhs)/2)/igh
                        datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(zws/igw) + "," + str(zhs/igh)
                        datastr += " --width " + str(zws) + " --height " + str(zhs)
                    # Supprimé: ancien zoom manuel (zoom == 5)
                    if False and zoom == 5:
                        zxo = ((igw/2)-(preview_width/2))/igw
                        if igw/igh > 1.5:
                            zyo = ((igh/2)-((preview_height * .75)/2))/igh
                        else:
                            zyo = ((igh/2)-(preview_height/2))/igh
                        if igw/igh > 1.5:
                            datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(int(preview_width)/igw) + "," + str(int(preview_height * .75)/igh)
                        else:
                            datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(preview_width/igw) + "," + str(preview_height/igh)
                    datastr += " --metadata - --metadata-format txt >> PiLibtext.txt"
                    if show_cmds == 1:
                        print (datastr)

                    # MODE PICAMERA2 : Capture directe
                    if use_picamera2 and picam2 is not None:
                        try:
                            # Déterminer la configuration de capture selon l'extension
                            if extns[extn] == 'raw' or extns2[extn] == 'dng':
                                # Capture RAW - utiliser le stream 'raw'
                                picam2.capture_file(fname, name='raw', format=extns[extn])
                            else:
                                # Capture JPEG/PNG/BMP - utiliser le stream 'main'
                                picam2.capture_file(fname, name='main')

                            # Attendre que le fichier soit créé
                            import time
                            timeout = 5  # 5 secondes max
                            start_wait = time.time()
                            while not os.path.exists(fname) and (time.time() - start_wait) < timeout:
                                time.sleep(0.1)

                        except Exception as e:
                            print(f"Erreur capture Picamera2: {e}")
                            # En cas d'erreur, afficher un message
                            text(0,0,3,2,1,"Erreur capture",int(fv*1.5),1)

                    # MODE RPICAM-STILL : Commande système
                    else:
                        os.system(datastr)
                        while not os.path.exists(fname):
                            pass
                    if extns2[extn] == 'jpg' or extns2[extn] == 'bmp' or extns2[extn] == 'png':
                        image = pygame.image.load(fname)
                        if igw/igh > 1.5: 
                            image = pygame.transform.scale(image, (preview_width,int(preview_height * 0.75)))
                        else:
                            image = pygame.transform.scale(image, (preview_width,preview_height))
                        windowSurfaceObj.blit(image, (0,0))
                    dgain = 0
                    again = 0
                    etime = 0
                    if os.path.exists("PiLibtext.txt"):
                        with open("PiLibtext.txt", "r") as file:
                            line = file.readline()
                            check = line.split("=")
                            if check[0] == "DigitalGain":
                                dgain = check[1][:-1]
                            if check[0] == "AnalogueGain":
                                again = check[1][:-1]
                            if check[0] == "ExposureTime":
                                etime = check[1][:-1]
                            while line:
                                line = file.readline()
                                check = line.split("=")
                                if check[0] == "DigitalGain":
                                    dgain = check[1][:-1]
                                if check[0] == "AnalogueGain":
                                    again = check[1][:-1]
                                if check[0] == "ExposureTime":
                                    etime = check[1][:-1]
                    text(0,22,6,2,1,"Ana Gain: " + str(again) + " Dig Gain: " + str(dgain) + " Exp Time: " + str(etime) +"uS",int(fv*1.5),1)
                    text(0,0,6,2,1,fname,int(fv*1.5),1)
                    pygame.display.update()
                    time.sleep(2)
                    pygame.draw.rect(windowSurfaceObj,blackColor,Rect(0,int(preview_height * .75),preview_width,preview_height /4),0)
                    still = 0
                    menu = 0
                    Menu()
                    restart = 2
                        
                elif button_row == 1 and event.button != 3:

                    # TAKE VIDEO
                    video = 1
                    picam2_was_paused = False  # Initialiser pour éviter NameError
                    if not use_picamera2 and p is not None:
                        os.killpg(p.pid, signal.SIGTERM)
                        # Attendre que le processus de preview se termine complètement
                        poll = p.poll()
                        while poll == None:
                            poll = p.poll()
                            time.sleep(0.1)

                    button(0,1,1,3)
                    if Pi == 5:
                        text(0,1,2,0,1,"    RECORDING",ft,1)
                    else:
                        text(0,1,2,0,1,"       STOP ",ft,0)
                    text(0,0,6,2,1,"Please Wait, taking video ...",int(fv*1.7),1)
                    now = datetime.datetime.now()
                    timestamp = now.strftime("%y%m%d%H%M%S")
                    vname =  vid_dir + str(timestamp) + "." + codecs2[codec]
                    
                    if codecs[codec] == 'ser':
                        # Capture pour format SER - méthode YUV420 (SIMPLE ET FIABLE !)
                        text(0,0,6,2,1,"Recording YUV420 for SER...",int(fv*1.7),1)

                        # Construire la commande pour capturer en YUV420
                        if lver != "bookwo" and lver != "trixie":
                            datastr = "libcamera-vid"
                        else:
                            datastr = "rpicam-vid"

                        # Capturer en fichier YUV420 dans /run/shm
                        temp_video = "/run/shm/ser_temp_video.yuv"

                        datastr += " --camera " + str(camera) + " -t " + str(vlen * 1000)
                        datastr += " -o " + temp_video
                        datastr += " --codec yuv420"

                        # IMPORTANT: Pour un vrai ROI (méthode Test2), l'ordre doit être:
                        # --codec yuv420 --mode ... --roi ... --width ... --height ...

                        # Gestion du ROI (Region Of Interest) - DOIT venir AVANT --width/--height
                        if zoom > 0 and zoom <= 5 and zoom != 3:  # Zoom fixe 1x à 6x (zoom 3 désactivé)
                            # ROI centré sur le mode capteur natif (vrai ROI)
                            # Déterminer la profondeur de bits selon la caméra
                            if Pi_Cam == 4 or Pi_Cam == 10:  # Pi HQ ou IMX585
                                sensor_bits = 12
                            else:
                                sensor_bits = 10

                            # Arrondir à un nombre PAIR pour compatibilité YUV420
                            zws = int(igw * zfs[zoom])
                            zhs = int(igh * zfs[zoom])
                            if zws % 2 != 0:
                                zws -= 1  # Forcer pair
                            if zhs % 2 != 0:
                                zhs -= 1  # Forcer pair
                            zxo = ((igw-zws)/2)/igw
                            zyo = ((igh-zhs)/2)/igh
                            datastr += " --mode " + str(igw) + ":" + str(igh) + ":" + str(sensor_bits) + "  --roi " + str(zxo) + "," + str(zyo) + "," + str(zws/igw) + "," + str(zhs/igh)
                        # Supprimé: ancien zoom manuel (zoom == 5)
                        elif False and zoom == 5:
                            # ROI manuel centré basé sur preview_width/height
                            # Déterminer la profondeur de bits selon la caméra
                            if Pi_Cam == 4 or Pi_Cam == 10:  # Pi HQ ou IMX585
                                sensor_bits = 12
                            else:
                                sensor_bits = 10

                            zxo = ((igw/2)-(preview_width/2))/igw
                            if igw/igh > 1.5:
                                zyo = ((igh/2)-((preview_height * .75)/2))/igh
                            else:
                                zyo = ((igh/2)-(preview_height/2))/igh

                            datastr += " --mode " + str(igw) + ":" + str(igh) + ":" + str(sensor_bits)
                            if igw/igh > 1.5:
                                datastr += "  --roi " + str(zxo) + "," + str(zyo) + "," + str(int(preview_width)/igw) + "," + str(int(preview_height * .75)/igh)
                            else:
                                datastr += "  --roi " + str(zxo) + "," + str(zyo) + "," + str(preview_width/igw) + "," + str(preview_height/igh)

                        # Résolution et dimensions - Utiliser résolutions STANDARDS pour compatibilité SER
                        # Stratégie: ROI pour zoom (crop capteur) + rescale vers résolution standard
                        if zoom > 0 and zoom <= 5 and zoom != 3:  # Zoom fixe avec ROI (zoom 3 désactivé)
                            # Choisir une résolution standard compatible SER
                            # IMPORTANT: Utiliser uniquement des résolutions testées et validées pour SER
                            zoom_resolutions = {
                                1: (1920, 1080),  # 2x zoom -> Full HD (16:9, standard)
                                2: (1280, 720),   # 3x zoom -> HD (16:9, standard)
                                4: (640, 480),    # 5x zoom -> VGA (4:3, confirmé compatible SER)
                                5: (640, 368),    # 6x zoom -> nHD (16:9, confirmé compatible SER)
                            }

                            if zoom in zoom_resolutions:
                                actual_vwidth, actual_vheight = zoom_resolutions[zoom]
                            else:
                                # Fallback: utiliser la résolution du ROI
                                actual_vwidth = zws
                                actual_vheight = zhs

                            # Ajouter --width et --height pour rescaler vers résolution standard
                            datastr += " --width " + str(actual_vwidth) + " --height " + str(actual_vheight)
                        # Supprimé: ancien zoom manuel (zoom == 5)
                        elif False and zoom == 5:
                            # Zoom manuel : utiliser preview_width/height
                            if igw/igh > 1.5:
                                actual_vwidth = preview_width
                                actual_vheight = int(preview_height * .75)
                            else:
                                actual_vwidth = preview_width
                                actual_vheight = preview_height
                            datastr += " --width " + str(actual_vwidth) + " --height " + str(actual_vheight)
                        else:
                            # Pas de zoom : utiliser les dimensions demandées
                            actual_vwidth = vwidth
                            actual_vheight = vheight
                            # Spécifier le mode natif pour les résolutions natives de l'IMX585
                            if Pi_Cam == 10 and vwidth == 1928 and vheight == 1090:
                                datastr += " --mode 1928:1090:12"
                            elif Pi_Cam == 10 and vwidth == 3856 and vheight == 2180:
                                datastr += " --mode 3856:2180:12"
                            datastr += " --width " + str(vwidth) + " --height " + str(vheight)

                        if mode != 0:
                            datastr += " --framerate " + str(fps)
                        else:
                            speed7 = sspeed
                            speed7 = max(speed7,int((1/fps)*1000000))
                            datastr += " --framerate " + str(max(1, min(120, int(1000000/speed7))))

                        if vpreview == 0:
                            datastr += " -n "
                        datastr += " --brightness " + str(brightness/100) + " --contrast " + str(contrast/100)
                        
                        if mode == 0:
                            datastr += " --shutter " + str(sspeed)
                        else:
                            datastr += " --exposure " + modes[mode]
                        
                        datastr += " --gain " + str(gain)
                        
                        if ev != 0:
                            datastr += " --ev " + str(ev)
                        
                        if awb == 0:
                            datastr += " --awbgains " + str(red/10) + "," + str(blue/10)
                        else:
                            datastr += " --awb " + awbs[awb]
                        
                        datastr += " --metering " + meters[meter]
                        datastr += " --saturation " + str(saturation/10)
                        datastr += " --sharpness " + str(sharpness/10)
                        datastr += " --denoise " + denoises[denoise]
                        
                        if vflip == 1:
                            datastr += " --vflip"
                        if hflip == 1:
                            datastr += " --hflip"
                        
                        if Pi_Cam == 9 and os.path.exists("/home/" + Home_Files[0] + "/imx290a.json") and Pi == 5:
                            datastr += " --tuning-file /home/" + Home_Files[0] + "/imx290a.json"
                        if Pi_Cam == 10 and os.path.exists("/home/" + Home_Files[0] + "/imx585_lowlight.json") and Pi == 5:
                            datastr += " --tuning-file /home/" + Home_Files[0] + "/imx585_lowlight.json"

                        if show_cmds == 1:
                            print(datastr)

                        # Arrêter temporairement Picamera2 pour libérer la caméra
                        picam2_was_paused = pause_picamera2()

                        print(f"[DEBUG SER] Starting SER video recording: {vname}")
                        print(f"[DEBUG SER] Temp file: {temp_video}")
                        print(f"[DEBUG SER] Command: {datastr}")

                        # Capturer la vidéo
                        os.system(datastr)

                        print(f"[DEBUG SER] Recording finished")
                        print(f"[DEBUG SER] Temp file exists: {os.path.exists(temp_video)}")
                        if os.path.exists(temp_video):
                            print(f"[DEBUG SER] Temp file size: {os.path.getsize(temp_video)} bytes")

                        # Convertir RAW unpacked (SRGGB16) en SER avec le script qui marche !
                        text(0,0,6,2,1,"Converting SRGGB16 to RGB24...",int(fv*1.7),1)

                        # Calculer le FPS réel
                        if mode != 0:
                            actual_fps = fps
                        else:
                            speed7 = sspeed
                            speed7 = max(speed7,int((1/fps)*1000000))
                            actual_fps = max(1, min(120, int(1000000/speed7)))

                        # Vérifier que le fichier vidéo YUV420 existe
                        if not os.path.exists(temp_video):
                            text(0,0,6,2,1,"Error: YUV420 video file not created!",int(fv*1.7),1)
                            time.sleep(2)
                        else:
                            # Avec résolutions standards, pas besoin de détection automatique
                            # La résolution est exactement celle spécifiée dans actual_vwidth/actual_vheight
                            print(f"[DEBUG SER] Converting with standard resolution: {actual_vwidth}×{actual_vheight}")

                            # Vérification optionnelle: calculer taille attendue du fichier
                            file_size = os.path.getsize(temp_video)
                            duration_sec = vlen / 1000.0
                            num_frames = int(duration_sec * actual_fps)
                            expected_size = int(actual_vwidth * actual_vheight * 1.5 * num_frames)

                            print(f"[DEBUG SER] File size: {file_size} bytes, Expected: {expected_size} bytes, Frames: {num_frames}")

                            if expected_size > 0 and abs(file_size - expected_size) / expected_size > 0.1:  # Plus de 10% de différence
                                print(f"[DEBUG SER] Warning: File size mismatch ({100*abs(file_size-expected_size)/expected_size:.1f}% difference)")
                                print(f"[DEBUG SER] This may indicate incorrect resolution or frame count")

                            # Convertir YUV420 → SER avec la résolution standard
                            text(0,0,6,2,1,f"Converting YUV420 to SER ({actual_vwidth}×{actual_vheight}) @ {actual_fps} fps...",int(fv*1.7),1)

                            success, frame_count, msg = convert_yuv420_to_ser(
                                temp_video, vname, actual_vwidth, actual_vheight, fps=actual_fps
                            )

                            if success:
                                text(0,0,6,2,1,vname + f" - {frame_count} frames @ {actual_fps} fps",int(fv*1.5),1)
                            else:
                                text(0,0,6,2,1,f"Conversion error: {msg}",int(fv*1.7),1)

                            # Supprimer le fichier vidéo temporaire
                            try:
                                os.remove(temp_video)
                            except:
                                pass

                            time.sleep(2)

                        # Redémarrer Picamera2 si il avait été mis en pause
                        if picam2_was_paused:
                            resume_picamera2()


                    elif codecs2[codec] != 'raw' and codecs[codec] != 'ser':
                        # Code existant pour les autres formats (sauf SER et RAW)
                        if lver != "bookwo" and lver != "trixie":
                            datastr = "libcamera-vid"


                    if codecs2[codec] != 'raw' and codecs[codec] != 'ser':
                        if lver != "bookwo" and lver != "trixie":
                            datastr = "libcamera-vid"
                        else:
                            datastr = "rpicam-vid"
                        datastr += " --camera " + str(camera) + " -t " + str(vlen * 1000)
                        # Ajouter framerate AVANT -o (output)
                        if mode != 0:
                            datastr += " --framerate " + str(fps)
                        else:
                            speed7 = sspeed
                            speed7 = max(speed7,int((1/fps)*1000000))
                            datastr += " --framerate " + str(max(1, min(120, int(1000000/speed7))))
                        if codecs[codec] != 'h264' and codecs[codec] != 'mp4':
                            datastr += " --codec " + codecs[codec]
                        elif codecs[codec] != 'mp4':
                            prof = h264profiles[profile].split(" ")
                            #datastr += " --profile " + str(prof[0]) + " --level " + str(prof[1])
                            datastr += " --level " + str(prof[1])
                    elif codecs2[codec] == 'raw':
                        if lver != "bookwo" and lver != "trixie":
                            datastr = "libcamera-raw"
                        else:
                            datastr = "rpicam-raw"
                        datastr += " --camera " + str(camera) + " -t " + str(vlen * 1000) + " -o " + vname + " --framerate " + str(fps)

                    # Le code ci-dessous ne doit PAS s'exécuter pour le codec 'ser' qui construit sa propre commande
                    if codecs[codec] != 'ser':
                        if vpreview == 0:
                            datastr += " -n "
                        datastr += " --brightness " + str(brightness/100) + " --contrast " + str(contrast/100)

                        # IMPORTANT: Pour un vrai ROI (méthode Test2), ajouter --mode et --roi AVANT --width/--height
                        if zoom > 0 and zoom <= 5 and zoom != 3:  # Zoom fixe 1x à 6x (zoom 3 désactivé)
                            # Déterminer la profondeur de bits selon la caméra
                            if Pi_Cam == 4 or Pi_Cam == 10:  # Pi HQ ou IMX585
                                sensor_bits = 12
                            else:
                                sensor_bits = 10

                            # Arrondir à un nombre PAIR pour compatibilité YUV420
                            zws = int(igw * zfs[zoom])
                            zhs = int(igh * zfs[zoom])
                            if zws % 2 != 0:
                                zws -= 1  # Forcer pair
                            if zhs % 2 != 0:
                                zhs -= 1  # Forcer pair
                            zxo = ((igw-zws)/2)/igw
                            zyo = ((igh-zhs)/2)/igh
                            datastr += " --mode " + str(igw) + ":" + str(igh) + ":" + str(sensor_bits) + "  --roi " + str(zxo) + "," + str(zyo) + "," + str(zws/igw) + "," + str(zhs/igh)
                        # Supprimé: ancien zoom manuel (zoom == 5)
                        elif False and zoom == 5:
                            # Déterminer la profondeur de bits selon la caméra
                            if Pi_Cam == 4 or Pi_Cam == 10:  # Pi HQ ou IMX585
                                sensor_bits = 12
                            else:
                                sensor_bits = 10

                            zxo = ((igw/2)-(preview_width/2))/igw
                            if igw/igh > 1.5:
                                zyo = ((igh/2)-((preview_height * .75)/2))/igh
                            else:
                                zyo = ((igh/2)-(preview_height/2))/igh

                            datastr += " --mode " + str(igw) + ":" + str(igh) + ":" + str(sensor_bits)
                            if igw/igh > 1.5:
                                datastr += "  --roi " + str(zxo) + "," + str(zyo) + "," + str(int(preview_width / .75)/igw) + "," + str(preview_height/igh)
                            else:
                                datastr += "  --roi " + str(zxo) + "," + str(zyo) + "," + str(preview_width/igw) + "," + str(preview_height/igh)

                        # Résolution et dimensions - SEULEMENT si ROI non appliqué
                        # IMPORTANT: Quand --roi est utilisé, NE PAS spécifier --width/--height
                        # Le ROI définit déjà la taille de sortie (crop sans rescale = vrai zoom)
                        # Supprimé: ancien zoom manuel (zoom == 5)
                        if False and zoom == 5:
                            if igw/igh > 1.5:
                                datastr += " --width " + str(preview_width) + " --height " + str(int(preview_height * .75))
                            else:
                                datastr += " --width " + str(preview_width) + " --height " + str(preview_height)
                        elif zoom > 0 and zoom <= 5 and zoom != 3:  # Zoom fixe avec ROI (zoom 3 désactivé)
                            # NE RIEN FAIRE : le ROI a déjà défini la taille de sortie
                            pass
                        elif Pi_Cam == 4 and vwidth == 2028:
                            datastr += " --mode 2028:1520:12"
                        elif Pi_Cam == 3 and vwidth == 2304 and codec == 0:
                            datastr += " --mode 2304:1296:10 --width 2304 --height 1296"
                        elif Pi_Cam == 3 and vwidth == 2028 and codec == 0:
                            datastr += " --mode 2028:1520:10 --width 2028 --height 1520"
                        elif Pi_Cam == 10 and vwidth == 1928:
                            datastr += " --mode 1928:1090:12 --width 1928 --height 1090"
                        elif Pi_Cam == 10 and vwidth == 3856:
                            datastr += " --mode 3856:2180:12 --width 3856 --height 2180"
                        elif Pi_Cam == 10:
                            # Pour IMX585, forcer le mode natif le plus proche
                            if vwidth * vheight <= 2100000:  # Résolution <= 1928x1090
                                datastr += " --mode 1928:1090:12 --width 1928 --height 1090"
                            else:
                                datastr += " --mode 3856:2180:12 --width 3856 --height 2180"
                        else:
                            datastr += " --width " + str(vwidth) + " --height " + str(vheight)

                        # Ajouter les paramètres communs à tous les cas (zoom ou pas)
                        if mode == 0:
                            datastr += " --shutter " + str(sspeed)
                        else:
                            datastr += " --exposure " + modes[mode]
                        datastr += " --gain " + str(gain)
                        if ev != 0:
                            datastr += " --ev " + str(ev)
                        if awb == 0:
                            datastr += " --awbgains " + str(red/10) + "," + str(blue/10)
                        else:
                            datastr += " --awb " + awbs[awb]
                        datastr += " --metering " + meters[meter]
                        datastr += " --saturation " + str(saturation/10)
                        datastr += " --sharpness " + str(sharpness/10)
                        datastr += " --denoise "    + denoises[denoise]
                        if vflip == 1:
                            datastr += " --vflip"
                        if hflip == 1:
                            datastr += " --hflip"
                        if Pi_Cam == 9 and os.path.exists("/home/" + Home_Files[0] + "/imx290a.json") and Pi == 5:
                            datastr += " --tuning-file /home/" + Home_Files[0] + "/imx290a.json"
                        if Pi_Cam == 4 and scientific == 1:
                            if os.path.exists('/usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json') and Pi == 4:
                                datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json"
                            if os.path.exists('/usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json') and Pi == 5:
                                datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json"
                        if Pi_Cam == 5 and foc_man == 1 and Pi == 5:
                            if os.path.exists('/usr/share/libcamera/ipa/rpi/pisp/imx519mf.json'):
                                datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/pisp/imx519mf.json"
                        elif Pi_Cam == 5  and foc_man == 1 and Pi != 5:
                            if os.path.exists('/usr/share/libcamera/ipa/rpi/vc4/imx519mff.json'):
                                datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/vc4/imx519mff.json"
                        if Pi_Cam == 6  and foc_man == 1 and Pi == 5:
                            if os.path.exists('/usr/share/libcamera/ipa/rpi/pisp/arducam_64mf.json'):
                                datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/pisp/arducam_64mf.json"
                        if Pi_Cam == 6  and foc_man == 1 and Pi != 5:
                            if os.path.exists('/usr/share/libcamera/ipa/rpi/vc4/arducam_64mff.json'):
                                datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/vc4/arducam_64mff.json"
                        if ((Pi_Cam == 3 and v3_af == 1) and v3_f_mode > 0 and fxx == 0) or (((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6)) or Pi_Cam == 8:
                            if v3_f_mode == 1:
                                if Pi_Cam == 3:
                                    datastr += " --lens-position " + str(v3_focus/100)
                                if Pi_Cam == 8 or ((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6):
                                    datastr += " --lens-position " + str(focus/100)
                        if ((Pi_Cam == 3 and v3_af == 1) or (((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6)) or Pi_Cam == 8) and zoom == 0 and fxx != 0 and v3_f_mode != 1:
                            datastr += " --autofocus-window " + str(fxx) + "," + str(fxy) + "," + str(fxz) + "," + str(fxz)
                        if (Pi_Cam == 3 and v3_af == 1) and v3_f_speed != 0:
                            datastr += " --autofocus-speed " + v3_f_speeds[v3_f_speed]
                        if (Pi_Cam == 3 and v3_af == 1) and v3_f_range != 0:
                            datastr += " --autofocus-range " + v3_f_ranges[v3_f_range]
                        if Pi_Cam == 3 or Pi == 5:
                            datastr += " --hdr " + v3_hdrs[v3_hdr]
                        datastr += " -p 0,0," + str(preview_width) + "," + str(preview_height)
                        # Ajouter le fichier de sortie (-o) APRES tous les autres paramètres
                        datastr += " -o " + vname
                        # ROI déjà ajouté plus haut (après brightness/contrast) pour respecter l'ordre de Test2
                        if show_cmds == 1:
                            print (datastr)

                        # Arrêter temporairement Picamera2 pour libérer la caméra
                        picam2_was_paused = pause_picamera2()

                        print(f"[DEBUG] Starting video recording: {vname}")
                        print(f"[DEBUG] Command: {datastr}")
                        print(f"[DEBUG] Pi={Pi}, codec={codecs[codec]}")

                        if Pi == 5 and codecs[codec] == 'mp4':
                            print("[DEBUG] Using os.system() for MP4 on Pi 5")
                            os.system(datastr)
                        else:
                            print("[DEBUG] Using subprocess.Popen()")
                            p = subprocess.Popen(datastr, shell=True, preexec_fn=os.setsid)
                        start_video = time.monotonic()
                        stop = 0
                        while (time.monotonic() - start_video < vlen or vlen == 0) and stop == 0:
                          if vlen != 0:
                              vlength = int(vlen - (time.monotonic()-start_video))
                          else:
                              vlength = int(time.monotonic()-start_video)
                          td = timedelta(seconds=vlength)
                          text(0,1,1,1,1,str(td),fv,0)
                          for event in pygame.event.get():
                              if (event.type == MOUSEBUTTONUP):
                                  mousex, mousey = event.pos
                                  # stop video recording
                                  if mousex > preview_width:
                                      button_row = int((mousey)/bh)
                                      if mousex > preview_width + (bw/2):
                                          button_pos = 1
                                      else:
                                          button_pos = 0
                                  if button_row == 1:
                                      if p is not None:
                                          os.killpg(p.pid, signal.SIGTERM)
                                      stop = 1

                    print(f"[DEBUG] Video recording stopped")
                    print(f"[DEBUG] Checking if file exists: {vname}")
                    print(f"[DEBUG] File exists: {os.path.exists(vname)}")
                    if os.path.exists(vname):
                        print(f"[DEBUG] File size: {os.path.getsize(vname)} bytes")

                    text(0,0,6,2,1,vname,int(fv*1.5),1)
                    time.sleep(1)

                    # Redémarrer Picamera2 si il avait été mis en pause
                    if picam2_was_paused:
                        text(0,0,2,2,1,"Redémarrage Picamera2...",int(fv*1.5),1)
                        pygame.display.update()
                        resume_picamera2()
                        time.sleep(0.5)

                    # Post-traitement pour Pi 5 avec MP4/H264 (correction timestamps)
                    if Pi == 5 and (codecs[codec] == 'mp4' or codecs[codec] == 'h264'):
                        text(0,0,2,2,1,"Post-traitement ffmpeg en cours...",int(fv*1.5),1)
                        pygame.display.update()

                        # Calculer le framerate utilisé
                        if mode != 0:
                            fps_used = fps
                        else:
                            speed7 = sspeed
                            speed7 = max(speed7, int((1/fps)*1000000))
                            fps_used = max(1, min(120, int(1000000/speed7)))

                        # Appeler la fonction de correction des timestamps
                        success = fix_video_timestamps(vname, fps_used, quality_preset="ultrafast")

                        if success:
                            text(0,0,1,2,1,"Vidéo corrigée avec succès",int(fv*1.5),1)
                        else:
                            text(0,0,3,2,1,"Avertissement: correction timestamps échouée",int(fv*1.5),1)

                        pygame.display.update()
                        time.sleep(2)

                    td = timedelta(seconds=vlen)
                    text(0,1,3,1,1,str(td),fv,0)
                    video = 0
                    menu = 0
                    Menu()
                    restart = 2
                                       
                elif button_row == 1 and event.button == 3:
                    # STREAM VIDEO
                    stream = 1
                    picam2_was_paused = False  # Initialiser pour éviter NameError
                    if not use_picamera2 and p is not None:
                        os.killpg(p.pid, signal.SIGTERM)
                        # Attendre que le processus de preview se termine complètement
                        poll = p.poll()
                        while poll == None:
                            poll = p.poll()
                            time.sleep(0.1)

                    button(0,1,1,3)
                    text(0,1,2,0,1,"           STOP ",ft,0)
                    text(0,0,6,2,1,"Please Wait, streaming video ...",int(fv*1.7),1)
                    now = datetime.datetime.now()
                    timestamp = now.strftime("%y%m%d%H%M%S")
                    vname =  vid_dir + str(timestamp) + "." + codecs2[codec]
                    if lver != "bookwo" and lver != "trixie":
                        datastr = "libcamera-vid "
                    else:
                        datastr = "rpicam-vid "
                    datastr += "--camera " + str(camera) + " -t " + str(vlen * 1000)
                    # Ajouter framerate AVANT les options de sortie
                    if mode != 0:
                        datastr += " --framerate " + str(fps)
                    else:
                        speed7 = sspeed
                        speed7 = max(speed7,int((1/fps)*1000000))
                        datastr += " --framerate " + str(max(1, min(120, int(1000000/speed7))))
                    # Ajouter les options de sortie APRES framerate
                    if stream_type == 0:
                        datastr += " --inline --listen -o tcp://0.0.0.0:" + str(stream_port)
                    elif stream_type == 1:
                        datastr += " --inline -o udp://" + udp_ip_addr + ":" + str(stream_port)
                    prof = h264profiles[profile].split(" ")
                    #datastr += " --profile " + str(prof[0]) + " --level " + str(prof[1])
                    datastr += " --level " + str(prof[1])
                    if vpreview == 0:
                        datastr += " -n "
                    datastr += " --brightness " + str(brightness/100) + " --contrast " + str(contrast/100)

                    # IMPORTANT: Pour un vrai ROI (méthode Test2), ajouter --mode et --roi AVANT --width/--height
                    if zoom > 0 and zoom <= 5 and zoom != 3:  # Zoom fixe 1x à 6x (zoom 3 désactivé)
                        # Déterminer la profondeur de bits selon la caméra
                        if Pi_Cam == 4 or Pi_Cam == 10:  # Pi HQ ou IMX585
                            sensor_bits = 12
                        else:
                            sensor_bits = 10

                        zws = int(igw * zfs[zoom])
                        zhs = int(igh * zfs[zoom])
                        zxo = ((igw-zws)/2)/igw
                        zyo = ((igh-zhs)/2)/igh
                        datastr += " --mode " + str(igw) + ":" + str(igh) + ":" + str(sensor_bits) + "  --roi " + str(zxo) + "," + str(zyo) + "," + str(zws/igw) + "," + str(zhs/igh)
                    # Supprimé: ancien zoom manuel (zoom == 5)
                    elif False and zoom == 5:
                        # Déterminer la profondeur de bits selon la caméra
                        if Pi_Cam == 4 or Pi_Cam == 10:  # Pi HQ ou IMX585
                            sensor_bits = 12
                        else:
                            sensor_bits = 10

                        zxo = ((igw/2)-(preview_width/2))/igw
                        if igw/igh > 1.5:
                            zyo = ((igh/2)-((preview_height * .75)/2))/igh
                        else:
                            zyo = ((igh/2)-(preview_height/2))/igh

                        datastr += " --mode " + str(igw) + ":" + str(igh) + ":" + str(sensor_bits)
                        if igw/igh > 1.5:
                            datastr += "  --roi " + str(zxo) + "," + str(zyo) + "," + str(int(preview_width / .75)/igw) + "," + str(preview_height/igh)
                        else:
                            datastr += "  --roi " + str(zxo) + "," + str(zyo) + "," + str(preview_width/igw) + "," + str(preview_height/igh)

                    # Résolution et dimensions - SEULEMENT si ROI non appliqué
                    # IMPORTANT: Quand --roi est utilisé, NE PAS spécifier --width/--height
                    # Supprimé: ancien zoom manuel (zoom == 5)
                    if False and zoom == 5:
                        datastr += " --width " + str(preview_width) + " --height " + str(preview_height)
                    elif zoom > 0 and zoom <= 5 and zoom != 3:  # Zoom fixe avec ROI (zoom 3 désactivé)
                        # NE RIEN FAIRE : le ROI a déjà défini la taille de sortie
                        pass
                    elif Pi_Cam == 4 and vwidth == 2028:
                        datastr += " --mode 2028:1520:12"
                    elif Pi_Cam == 3 and vwidth == 2304 and codec == 0:
                        datastr += " --mode 2304:1296:10 --width 2304 --height 1296"
                    elif Pi_Cam == 3 and vwidth == 2028 and codec == 0:
                        datastr += " --mode 2028:1520:10 --width 2028 --height 1520"
                    elif Pi_Cam == 10 and vwidth == 1928:
                        datastr += " --mode 1928:1090:12 --width 1928 --height 1090"
                    elif Pi_Cam == 10 and vwidth == 3856:
                        datastr += " --mode 3856:2180:12 --width 3856 --height 2180"
                    elif Pi_Cam == 10:
                        # Pour IMX585, forcer le mode natif le plus proche
                        if vwidth * vheight <= 2100000:  # Résolution <= 1928x1090
                            datastr += " --mode 1928:1090:12 --width 1928 --height 1090"
                        else:
                            datastr += " --mode 3856:2180:12 --width 3856 --height 2180"
                    else:
                        datastr += " --width " + str(vwidth) + " --height " + str(vheight)
                    if mode == 0:
                        datastr += " --shutter " + str(sspeed)
                    else:
                        datastr += " --exposure " + modes[mode]
                    datastr += " --gain " + str(gain)
                    if ev != 0:
                        datastr += " --ev " + str(ev)
                    if awb == 0:
                        datastr += " --awbgains " + str(red/10) + "," + str(blue/10)
                    else:
                        datastr += " --awb " + awbs[awb]
                    datastr += " --metering " + meters[meter]
                    datastr += " --saturation " + str(saturation/10)
                    datastr += " --sharpness " + str(sharpness/10)
                    datastr += " --denoise "    + denoises[denoise]
                    if vflip == 1:
                        datastr += " --vflip"
                    if hflip == 1:
                        datastr += " --hflip"
                    if Pi_Cam == 9 and os.path.exists("/home/" + Home_Files[0] + "/imx290a.json") and Pi == 5:
                        datastr += " --tuning-file /home/" + Home_Files[0] + "/imx290a.json"
                    if Pi_Cam == 10 and os.path.exists("/home/" + Home_Files[0] + "/imx585_lowlight.json") and Pi == 5:
                        datastr += " --tuning-file /home/" + Home_Files[0] + "/imx585_lowlight.json"
                    if Pi_Cam == 4 and scientific == 1:
                        if os.path.exists('/usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json') and Pi == 4:
                            datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json"
                        if os.path.exists('/usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json') and Pi == 5:
                            datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json"
                    if ((Pi_Cam == 3 and v3_af == 1) and v3_f_mode > 0 and fxx == 0) or (((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6) ) or Pi_Cam == 8:
                        datastr += " --autofocus-mode " + v3_f_modes[v3_f_mode]
                        if v3_f_mode == 1:
                            if Pi_Cam == 3:
                                datastr += " --lens-position " + str(v3_focus/100)
                            if Pi_Cam == 8 or ((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6):
                                datastr += " --lens-position " + str(focus/100)
                    if ((Pi_Cam == 3 and v3_af == 1) or ((Pi_Cam == 5 or Pi_Cam == 6) ) or Pi_Cam == 8)  and zoom == 0 and fxx != 0 and v3_f_mode != 1:
                        datastr += " --autofocus-window " + str(fxx) + "," + str(fxy) + "," + str(fxz) + "," + str(fxz)
                    if (Pi_Cam == 3 and v3_af == 1) and v3_f_speed != 0:
                        datastr += " --autofocus-speed " + v3_f_speeds[v3_f_speed]
                    if (Pi_Cam == 3 and v3_af == 1) and v3_f_range != 0:
                        datastr += " --autofocus-range " + v3_f_ranges[v3_f_range]
                    if Pi_Cam == 3 or Pi == 5:
                        datastr += " --hdr " + v3_hdrs[v3_hdr]
                    datastr += " -p 0,0," + str(preview_width) + "," + str(preview_height)
                    # ROI déjà ajouté plus haut (après brightness/contrast) pour respecter l'ordre de Test2
                    if stream_type == 2:
                        data = "#rtp{sdp=rtsp://:" + str(stream_port) + "/stream1}"
                        datastr += " --inline -o | cvlc stream:///dev/stdin --sout '" + data + "' :demux=h264" ###
                    if show_cmds == 1:
                        print (datastr)

                    # Arrêter temporairement Picamera2 pour libérer la caméra
                    picam2_was_paused = pause_picamera2()

                    p = subprocess.Popen(datastr, shell=True, preexec_fn=os.setsid)
                    start_video = time.monotonic()
                    stop = 0
                    while (time.monotonic() - start_video < vlen or vlen == 0) and stop == 0:
                        if vlen != 0:
                            vlength = int(vlen - (time.monotonic()-start_video))
                        else:
                            vlength = int(time.monotonic()-start_video)
                        td = timedelta(seconds=vlength)
                        text(0,1,1,1,1,str(td),fv,0)
                        for event in pygame.event.get():
                            if (event.type == MOUSEBUTTONUP):
                                mousex, mousey = event.pos
                                # stop video streaming
                                if mousex > preview_width:
                                    button_row = int((mousey)/bh)
                                    if mousex > preview_width + (bw/2):
                                        button_pos = 1
                                    else:
                                        button_pos = 0
                                if button_row == 1:
                                   if p is not None:
                                       os.killpg(p.pid, signal.SIGTERM)
                                   stop = 1

                    # Redémarrer Picamera2 si il avait été mis en pause
                    if picam2_was_paused:
                        resume_picamera2()

                    td = timedelta(seconds=vlen)
                    text(0,1,3,1,1,str(td),fv,0)
                    stream = 0
                    menu = 0
                    Menu()
                    restart = 2
                        
                elif button_row == 2:
                    # TAKE TIMELAPSE
                    if not use_picamera2 and p is not None:
                        os.killpg(p.pid, signal.SIGTERM)
                    # Fermer Picamera2 si actif pour libérer la caméra pour rpicam-still
                    pause_picamera2()
                    restart = 1
                    timelapse = 1
                    button(0,2,1,4)
                    text(0,2,2,0,1,"           STOP",ft,0)
                    tcount = 0
                      
                    if tinterval > 0 and mode != 0: # normal mode
                        text(0,0,6,2,1,"Please Wait, taking Timelapse ...",int(fv*1.7),1)
                        now = datetime.datetime.now()
                        timestamp = now.strftime("%y%m%d%H%M%S")
                        count = 0
                        fname =  pic_dir + str(timestamp) + '_%04d.' + extns2[extn]
                        if lver != "bookwo" and lver != "trixie":
                            datastr = "libcamera-still"
                        else:
                            datastr = "rpicam-still"
                        if extns[extn] != 'raw':
                            datastr += " --camera " + str(camera) + " -e " + extns[extn] + " -s -t 0 -o " + fname
                            datastr += " -n"
                        else:
                            datastr += " --camera " + str(camera) + " -r -s -t 0 -o " + fname 
                            datastr += " -n"
                        datastr += " --brightness " + str(brightness/100) + " --contrast " + str(contrast/100)
                        if mode == 0:
                            datastr += " --shutter " + str(sspeed)
                        else:
                            datastr += " --exposure " + modes[mode]
                        if ev != 0:
                            datastr += " --ev " + str(ev)
                        if sspeed > 1000000 and mode == 0:
                            datastr += " --gain " + str(gain) + " --immediate --awbgains " + str(red/10) + "," + str(blue/10)
                        else:
                            datastr += " --gain " + str(gain)
                            if awb == 0:
                                datastr += " --awbgains " + str(red/10) + "," + str(blue/10)
                            else:
                                datastr += " --awb " + awbs[awb]
                        datastr += " --metering " + meters[meter]
                        datastr += " --saturation " + str(saturation/10)
                        datastr += " --sharpness " + str(sharpness/10)
                        datastr += " --quality " + str(quality)
                        datastr += " --denoise "    + denoises[denoise]
                        if vflip == 1:
                            datastr += " --vflip"
                        if hflip == 1:
                            datastr += " --hflip"
                        if Pi_Cam == 9 and os.path.exists("/home/" + Home_Files[0] + "/imx290a.json") and Pi == 5:
                            datastr += " --tuning-file /home/" + Home_Files[0] + "/imx290a.json"
                        if Pi_Cam == 4 and scientific == 1:
                            if os.path.exists('/usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json') and Pi == 4:
                                datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json"
                            if os.path.exists('/usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json') and Pi == 5:
                                datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json"
                        if ((Pi_Cam == 3 and v3_af == 1) and v3_f_mode > 0 and fxx == 0) or (((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6)) or Pi_Cam == 8:
                            datastr += " --autofocus-mode " + v3_f_modes[v3_f_mode]
                            if v3_f_mode == 1:
                                if Pi_Cam == 3 and v3_af == 1:
                                    datastr += " --lens-position " + str(v3_focus/100)
                                if Pi_Cam == 8 or ((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6):
                                    datastr += " --lens-position " + str(focus/100)
                        elif (Pi_Cam == 3 and v3_af == 1) and v3_f_mode == 0 and fxz == 1:
                            datastr += " --autofocus-mode " + v3_f_modes[v3_f_mode] + " --autofocus-on-capture"
                        if ((Pi_Cam == 3 and v3_af == 1) or (((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6)) or Pi_Cam == 8) and zoom == 0:
                            datastr += " --autofocus-window " + str(fxx) + "," + str(fxy) + "," + str(fxz) + "," + str(fxz)
                        if Pi_Cam == 3 or Pi == 5:
                            datastr += " --hdr " + v3_hdrs[v3_hdr]
                        if (Pi_Cam == 6 or Pi_Cam == 8) and mode == 0 and button_pos == 3:
                            datastr += " --width 4624 --height 3472 " # 16MP superpixel mode for higher light sensitivity
                        elif (Pi_Cam == 5 or Pi_Cam == 6 or Pi_Cam == 8):
                            if Pi != 5 and lo_res == 1:
                                datastr += " --width 4624 --height 3472"
                            elif Pi_Cam == 6:
                                datastr += " --width 9152 --height 6944"
                            elif Pi_Cam == 8:
                                datastr += " --width 9248 --height 6944"
                        # Zoom fixe 1x à 6x
                        if zoom > 0 and zoom <= 5 and zoom != 3:  # Zoom 3 désactivé
                            # Arrondir à un nombre PAIR pour compatibilité formats vidéo
                            zws = int(igw * zfs[zoom])
                            zhs = int(igh * zfs[zoom])
                            if zws % 2 != 0:
                                zws -= 1  # Forcer pair
                            if zhs % 2 != 0:
                                zhs -= 1  # Forcer pair
                            zxo = ((igw-zws)/2)/igw
                            zyo = ((igh-zhs)/2)/igh
                            datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(zws/igw) + "," + str(zhs/igh)
                            datastr += " --width " + str(zws) + " --height " + str(zhs)
                        # Supprimé: ancien zoom manuel (zoom == 5)
                        if False and zoom == 5:
                            zxo = ((igw/2)-(preview_width/2))/igw
                            if igw/igh > 1.5:
                                zyo = ((igh/2)-((preview_height * .75)/2))/igh
                            else:
                                zyo = ((igh/2)-(preview_height/2))/igh
                            if igw/igh > 1.5:
                                datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(int(preview_width)/igw) + "," + str(int(preview_height * .75)/igh)
                            else:
                                datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(preview_width/igw) + "," + str(preview_height/igh)
                        p = subprocess.Popen(datastr, shell=True, preexec_fn=os.setsid)
                        if show_cmds == 1:
                            print (datastr)
                        start_timelapse = time.monotonic()
                        start2 = time.monotonic()
                        stop = 0
                        pics3 = []
                        count = 0
                        old_count = 0
                        while count < tshots and stop == 0:
                            if time.monotonic() - start2 >= tinterval:
                                if lver != "bookwo" and lver != "trixie":
                                    os.system('pkill -SIGUSR1 libcamera-still')
                                else:
                                    os.system('pkill -SIGUSR1 rpicam-still')
                                start2 = time.monotonic()
                                text(0,0,6,2,1,"Please Wait, taking Timelapse ..."  + " " + str(count+1),int(fv*1.7),1)
                                show = 0
                                while count == old_count:
                                    time.sleep(0.1)
                                    pics3 = glob.glob(pic_dir + "*.*")
                                    counts = []
                                    for xu in range(0,len(pics3)):
                                        ww = pics3[xu].split("/")
                                        if ww[-1][0:12] == timestamp:
                                            counts.append(pics3[xu])
                                    count = len(counts)
                                    counts.sort()
                                    for event in pygame.event.get():
                                        if (event.type == MOUSEBUTTONUP):
                                            mousex, mousey = event.pos
                                            # stop timelapse
                                            if mousex > preview_width:
                                                button_row = int((mousey)/bh)
                                            if button_row == 2:
                                                if p is not None:
                                                    os.killpg(p.pid, signal.SIGTERM)
                                                text(0,2,3,1,1,str(tshots),fv,12)
                                                stop = 1
                                                count = tshots
                                                                                        
                                old_count = count
                                text(0,2,1,1,1,str(tshots - count),fv,0)
                                tdur = tinterval * (tshots - count)
                                td = timedelta(seconds=tdur)
                            time.sleep(0.1)
                            if buttonSTR.is_pressed: # 
                                type = pygame.MOUSEBUTTONUP
                                if str_cap == 2:
                                    click_event = pygame.event.Event(type, {"button": 3, "pos": (0,0)})
                                else:
                                    click_event = pygame.event.Event(type, {"button": 1, "pos": (0,0)})
                                pygame.event.post(click_event)
                            for event in pygame.event.get():
                                if (event.type == MOUSEBUTTONUP):
                                    mousex, mousey = event.pos
                                    # stop timelapse or capture STILL
                                    if mousex > preview_width:
                                        button_row = int((mousey)/bh)
                                    if button_row == 2:
                                        if p is not None:
                                            os.killpg(p.pid, signal.SIGTERM)
                                        text(0,2,3,1,1,str(tshots),fv,12)
                                        stop = 1
                                        count = tshots
                                    if button_row == 0:
                                        if lver != "bookwo" and lver != "trixie":
                                            os.system('pkill -SIGUSR1 libcamera-still')
                                        else:
                                            os.system('pkill -SIGUSR1 rpicam-still')
                                        text(0,0,3,0,1,"CAPTURE",ft,7)
                        if lver != "bookwo" and lver != "trixie":
                            os.system('pkill -SIGUSR2 libcamera-still')
                        else:
                            os.system('pkill -SIGUSR2 rpicam-still')
                            
                    elif tinterval > 0 and mode == 0: # manual mode
                        text(0,0,6,2,1,"Please Wait, taking Timelapse ...",int(fv*1.7),1)
                        now = datetime.datetime.now()
                        timestamp = now.strftime("%y%m%d%H%M%S")
                        start2 = time.monotonic()
                        stop = 0
                        pics3 = []
                        count = 0
                        old_count = 0
                        trig = 1
                        p = None
                        while count < tshots and stop == 0:
                            if time.monotonic() - start2 > tinterval:
                                start2 = time.monotonic()
                                if p is not None:
                                    poll = p.poll()
                                    while poll == None:
                                        poll = p.poll()
                                        time.sleep(0.1)
                                fname =  pic_dir + str(timestamp) + "_" + str(count) + "." + extns2[extn]
                                if lver != "bookwo" and lver != "trixie":
                                    datastr = "libcamera-still"
                                else:
                                    datastr = "rpicam-still"
                                if extns[extn] != 'raw':
                                    datastr += " --camera " + str(camera) + " -e " + extns[extn] + " -t " + str(timet) + " -o " + fname + " -n"
                                else:
                                    datastr += " --camera " + str(camera) + " -r -t 1000 -o " + fname + " -n " 
                                datastr += " --brightness " + str(brightness/100) + " --contrast " + str(contrast/100)
                                datastr += " --shutter " + str(sspeed)
                                if ev != 0:
                                    datastr += " --ev " + str(ev)
                                if sspeed > 1000000 and mode == 0:
                                    datastr += " --gain " + str(gain) + " --immediate --awbgains " + str(red/10) + "," + str(blue/10)
                                else:
                                    datastr += " --gain " + str(gain)
                                    if awb == 0:
                                        datastr += " --awbgains " + str(red/10) + "," + str(blue/10)
                                    else:
                                        datastr += " --awb " + awbs[awb]
                                datastr += " --metering " + meters[meter]
                                datastr += " --saturation " + str(saturation/10)
                                datastr += " --sharpness " + str(sharpness/10)
                                datastr += " --quality " + str(quality)
                                datastr += " --denoise "    + denoises[denoise]
                                if vflip == 1:
                                    datastr += " --vflip"
                                if hflip == 1:
                                    datastr += " --hflip"
                                if Pi_Cam == 9 and os.path.exists("/home/" + Home_Files[0] + "/imx290a.json") and Pi == 5:
                                    datastr += " --tuning-file /home/" + Home_Files[0] + "/imx290a.json"
                                if Pi_Cam == 4 and scientific == 1:
                                    if os.path.exists('/usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json') and Pi == 4:
                                        datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json"
                                    if os.path.exists('/usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json') and Pi == 5:
                                        datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json"
                                if ((Pi_Cam == 3 and v3_af == 1) and v3_f_mode > 0 and fxx == 0) or (((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6)) or Pi_Cam == 8:
                                    datastr += " --autofocus-mode " + v3_f_modes[v3_f_mode]
                                    if v3_f_mode == 1:
                                        if Pi_Cam == 3 and v3_af == 1:
                                            datastr += " --lens-position " + str(v3_focus/100)
                                        if Pi_Cam == 8 or ((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6):
                                            datastr += " --lens-position " + str(focus/100)
                                elif (Pi_Cam == 3 and v3_af == 1) and v3_f_mode == 0 and fxz == 1:
                                    datastr += " --autofocus-mode " + v3_f_modes[v3_f_mode] + " --autofocus-on-capture"
                                if ((Pi_Cam == 3 and v3_af == 1) or (((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6)) or Pi_Cam == 8)  and zoom == 0:
                                    datastr += " --autofocus-window " + str(fxx) + "," + str(fxy) + "," + str(fxz) + "," + str(fxz)
                                if Pi_Cam == 3:
                                    datastr += " --hdr " + v3_hdrs[v3_hdr]
                                if (Pi_Cam == 6 or Pi_Cam == 8) and mode == 0 and button_pos == 3:
                                    datastr += " --width 4624 --height 3472 " # 16MP superpixel mode for higher light sensitivity
                                elif (Pi_Cam == 5 or Pi_Cam == 6 or Pi_Cam == 8):
                                    if Pi != 5 and lo_res == 1:
                                        datastr += " --width 4624 --height 3472"
                                    elif Pi_Cam == 6:
                                        datastr += " --width 9152 --height 6944"
                                    elif Pi_Cam == 8:
                                        datastr += " --width 9248 --height 6944"
                                # Zoom fixe 1x à 6x
                                if zoom > 0 and zoom <= 5:
                                    zws = int(igw * zfs[zoom])
                                    zhs = int(igh * zfs[zoom])
                                    zxo = ((igw-zws)/2)/igw
                                    zyo = ((igh-zhs)/2)/igh
                                    datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(zws/igw) + "," + str(zhs/igh)
                                # Supprimé: ancien zoom manuel (zoom == 5)
                                if False and zoom == 5:
                                    zxo = ((igw/2)-(preview_width/2))/igw
                                    if igw/igh > 1.5:
                                        zyo = ((igh/2)-((preview_height * .75)/2))/igh
                                    else:
                                        zyo = ((igh/2)-(preview_height/2))/igh
                                    if igw/igh > 1.5:
                                        datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(int(preview_width)/igw) + "," + str(int(preview_height * .75)/igh)
                                    else:
                                        datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(preview_width/igw) + "," + str(preview_height/igh)
                                if show_cmds == 1:
                                    print (datastr)
                                p = subprocess.Popen(datastr, shell=True, preexec_fn=os.setsid)
                                text(0,0,6,2,1,"Please Wait, taking Timelapse ..."  + " " + str(count+1),int(fv*1.7),1)
                                show = 0
                                while count == old_count:
                                    time.sleep(0.1)
                                    pics3 = glob.glob(pic_dir + "*.*")
                                    counts = []
                                    for xu in range(0,len(pics3)):
                                        ww = pics3[xu].split("/")
                                        if ww[-1][0:12] == timestamp:
                                            counts.append(pics3[xu])
                                    count = len(counts)
                                    counts.sort()
                                    if (extns2[extn] == 'jpg' or extns2[extn] == 'bmp' or extns2[extn] == 'png') and count > 0 and show == 0:
                                        image = pygame.image.load(counts[count-1])
                                        # Zoom manuel supprimé: simplifié la condition
                                        if (Pi_Cam != 3 and Pi_Cam != 10 and Pi_Cam != 15):
                                            catSurfacesmall = pygame.transform.scale(image, (preview_width,preview_height))
                                        else:
                                            catSurfacesmall = pygame.transform.scale(image, (preview_width,int(preview_height * 0.75)))
                                        windowSurfaceObj.blit(catSurfacesmall, (0, 0))
                                        text(0,0,6,2,1,counts[count-1],int(fv*1.5),1)
                                        pygame.display.update()
                                        show == 1
                                    for event in pygame.event.get():
                                        if (event.type == MOUSEBUTTONUP):
                                            mousex, mousey = event.pos
                                            # stop timelapse
                                            if mousex > preview_width:
                                                button_row = int((mousey)/bh)
                                            if button_row == 2:
                                                if p is not None:
                                                    os.killpg(p.pid, signal.SIGTERM)
                                                text(0,2,3,1,1,str(tshots),fv,0)
                                                stop = 1
                                                count = tshots
                                        
                                old_count = count
                                text(0,2,1,1,1,str(tshots - count),fv,12)
                                tdur = tinterval * (tshots - count)
                                td = timedelta(seconds=tdur)
                            time.sleep(0.1)
                            for event in pygame.event.get():
                                if (event.type == MOUSEBUTTONUP):
                                    mousex, mousey = event.pos
                                    # stop timelapse
                                    if mousex > preview_width:
                                        button_row = int((mousey)/bh)
                                    if button_row == 2:
                                        if p is not None:
                                            os.killpg(p.pid, signal.SIGTERM)
                                        text(0,2,3,1,1,str(tshots),fv,0)
                                        stop = 1
                                        count = tshots

                    elif tinterval == 0:
                        if tduration == 0:
                            tduration = 1
                        text(0,0,6,2,1,"Please Wait, taking Timelapse ...",int(fv*1.7),1)
                        now = datetime.datetime.now()
                        timestamp = now.strftime("%y%m%d%H%M%S")
                        fname =  pic_dir + str(timestamp) + '_%04d.' + extns2[extn]
                        if codecs2[codec] != 'raw':
                            if lver != "bookwo" and lver != "trixie":
                                datastr = "libcamera-vid"
                            else:
                                datastr = "rpicam-vid"
                            datastr += " --camera " + str(camera) + " -n --codec mjpeg -t " + str(tduration*1000) + " --segment 400"
                            # Ajouter le framerate AVANT les autres paramètres
                            if mode == 0:
                                # Permettre FPS < 1 pour longues expositions (0.01 fps min)
                                if sspeed > 0:
                                    calc_fps = max(min(1000000/sspeed, 120), 0.01)
                                else:
                                    calc_fps = 30
                                datastr += " --framerate " + str(calc_fps)
                            else:
                                datastr += " --framerate " + str(fps)
                        else:
                            fname =  pic_dir + str(timestamp) + '_%04d.' + codecs2[codec]
                            if lver != "bookwo" and lver != "trixie":
                                datastr = "libcamera-raw"
                            else:
                                datastr = "rpicam-raw"
                            datastr += " --camera " + str(camera) + " -n -t " + str(tduration*1000) + " --segment 400"
                            # Ajouter le framerate pour raw aussi
                            if mode == 0:
                                # Permettre FPS < 1 pour longues expositions (0.01 fps min)
                                if sspeed > 0:
                                    calc_fps = max(min(1000000/sspeed, 120), 0.01)
                                else:
                                    calc_fps = 30
                                datastr += " --framerate " + str(calc_fps)
                            else:
                                datastr += " --framerate " + str(fps)
                        if zoom > 0:
                            if igw/igh > 1.5:
                                datastr += " --width " + str(int(preview_width)) + " --height " + str(int(preview_height * .75))
                            else:
                                datastr += " --width " + str(preview_width) + " --height " + str(preview_height)
                        else:
                            datastr += " --width " + str(vwidth) + " --height " + str(vheight)
                        datastr += " --brightness " + str(brightness/100) + " --contrast " + str(contrast/100)
                        if mode == 0:
                            datastr += " --shutter " + str(sspeed)
                        else:
                            datastr += " --exposure " + str(modes[mode])
                        if ev != 0:
                            datastr += " --ev " + str(ev)
                        if sspeed > 5000000 and mode == 0 and (Pi_Cam < 5 or Pi_Cam == 7):
                            datastr += " --gain 1 --immediate --awbgains " + str(red/10) + "," + str(blue/10)
                        else:
                            datastr += " --gain " + str(gain)
                            if awb == 0:
                                datastr += " --awbgains " + str(red/10) + "," + str(blue/10)
                            else:
                                datastr += " --awb " + awbs[awb]
                        datastr += " --metering "   + meters[meter]
                        datastr += " --saturation " + str(saturation/10)
                        datastr += " --sharpness "  + str(sharpness/10)
                        datastr += " --denoise "    + denoises[denoise]
                        if vflip == 1:
                            datastr += " --vflip"
                        if hflip == 1:
                            datastr += " --hflip"
                        if Pi_Cam == 9 and os.path.exists("/home/" + Home_Files[0] + "/imx290a.json") and Pi == 5:
                            datastr += " --tuning-file /home/" + Home_Files[0] + "/imx290a.json"
                        if Pi_Cam == 4 and scientific == 1:
                            if os.path.exists('/usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json') and Pi == 4:
                                datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/vc4/imx477_scientific.json"
                            if os.path.exists('/usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json') and Pi == 5:
                                datastr += " --tuning-file /usr/share/libcamera/ipa/rpi/pisp/imx477_scientific.json"
                        if ((Pi_Cam == 3 and v3_af == 1) and v3_f_mode > 0 and fxx == 0) or (((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6)) or Pi_Cam == 8:
                            datastr += " --autofocus-mode " + v3_f_modes[v3_f_mode]
                            if v3_f_mode == 1:
                                if Pi_Cam == 3:
                                    datastr += " --lens-position " + str(v3_focus/100)
                                if Pi_Cam == 8 or ((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6):
                                    datastr += " --lens-position " + str(focus/100)
                        if ((Pi_Cam == 3 and v3_af == 1) or (((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6) ) or Pi_Cam == 8) and zoom == 0:
                            datastr += " --autofocus-window " + str(fxx) + "," + str(fxy) + "," + str(fxz) + "," + str(fxz)
                        if Pi_Cam == 3 or Pi == 5:
                            datastr += " --hdr " + v3_hdrs[v3_hdr]
                        # Zoom fixe 1x à 6x
                        if zoom > 0 and zoom <= 5 and zoom != 3:  # Zoom 3 désactivé
                            # Arrondir à un nombre PAIR pour compatibilité formats vidéo
                            zws = int(igw * zfs[zoom])
                            zhs = int(igh * zfs[zoom])
                            if zws % 2 != 0:
                                zws -= 1  # Forcer pair
                            if zhs % 2 != 0:
                                zhs -= 1  # Forcer pair
                            zxo = ((igw-zws)/2)/igw
                            zyo = ((igh-zhs)/2)/igh
                            datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(zws/igw) + "," + str(zhs/igh)
                        # Supprimé: ancien zoom manuel (zoom == 5)
                        if False and zoom == 5:
                            zxo = ((igw/2)-(preview_width/2))/igw
                            if igw/igh > 1.5:
                                zyo = ((igh/2)-((preview_height * .75)/2))/igh
                            else:
                                zyo = ((igh/2)-(preview_height/2))/igh
                            if igw/igh > 1.5:
                                datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(int(preview_width / .75)/igw) + "," + str(preview_height/igh)
                            else:
                                datastr += " --roi " + str(zxo) + "," + str(zyo) + "," + str(preview_width/igw) + "," + str(preview_height/igh)
                        # Ajouter le fichier de sortie à la fin
                        datastr += " -o " + fname
                        if show_cmds == 1:
                            print (datastr)
                        p = subprocess.Popen(datastr, shell=True, preexec_fn=os.setsid)
                        start_timelapse = time.monotonic()
                        stop = 0
                        while time.monotonic() - start_timelapse < tduration+1 and stop == 0:
                            tdur = int(tduration - (time.monotonic() - start_timelapse))
                            td = timedelta(seconds=tdur)
                            text(0,2,1,1,1,str(td),fv,0)
                        # Attendre que le processus rpicam-vid se termine complètement
                        if p is not None:
                            print("[DEBUG] Waiting for rpicam-vid to finish...")
                            p.wait()
                            print("[DEBUG] rpicam-vid finished, waiting for camera release...")
                            time.sleep(2.0)  # Délai supplémentaire pour libération complète de la caméra
                    # Redémarrer Picamera2 si nécessaire après le timelapse
                    resume_picamera2()
                    timelapse = 0
                    menu = 0
                    Menu()
                    restart = 2 
                        
                elif button_row == 3:
                    # LIVE STACK - Active/désactive le mode Live Stacking
                    if not livestack_active:
                        # Activer Live Stack
                        livestack_active = True

                        # Activer le mode stretch pour affichage fullscreen
                        stretch_mode = 1

                        # Passer en mode fullscreen (comme pour STRETCH)
                        display_modes = pygame.display.list_modes()
                        if display_modes and display_modes != -1:
                            max_width, max_height = display_modes[0]
                        else:
                            screen_info = pygame.display.Info()
                            max_width, max_height = screen_info.current_w, screen_info.current_h
                        windowSurfaceObj = pygame.display.set_mode((max_width, max_height), pygame.FULLSCREEN, 24)

                        # Créer l'instance livestack si nécessaire
                        if livestack is None:
                            # Récupérer les paramètres actuels de la caméra (comme pour timelapse)
                            camera_params = {
                                'exposure': sspeed,  # Déjà en microsecondes
                                'gain': gain,
                                'red': red / 10,  # Divisé par 10 pour ColourGains
                                'blue': blue / 10
                            }
                            livestack = create_livestack_session(camera_params)

                            # Configurer avec paramètres utilisateur
                            # Le stretch sera TOUJOURS appliqué par le programme principal
                            # pour garantir des paramètres identiques et éviter le double stretch
                            # OFF (valeur 0) = désactive le contrôle (valeur très grande)

                            livestack.configure(
                                alignment_mode=ls_alignment_modes[ls_alignment_mode],
                                enable_qc=bool(ls_enable_qc),
                                max_fwhm=ls_max_fwhm / 10.0 if ls_max_fwhm > 0 else 999.0,
                                min_sharpness=ls_min_sharpness / 1000.0 if ls_min_sharpness > 0 else 0.0,
                                max_drift=float(ls_max_drift) if ls_max_drift > 0 else 999999.0,
                                min_stars=int(ls_min_stars),  # 0 = désactivé (accepté par libastrostack)
                                # Pas de stretch dans libastrostack - sera fait par le programme principal
                                png_stretch="linear",
                                png_factor=1.0,
                                preview_refresh=ls_preview_refresh,
                                save_dng="none"
                            )

                        # Démarrer la session
                        livestack.start()
                        print("[LIVESTACK] Mode activé")
                    else:
                        # Désactiver Live Stack
                        livestack_active = False
                        stretch_mode = 0  # Désactiver aussi le stretch
                        if livestack is not None:
                            livestack.stop()

                        # Restaurer le mode d'affichage normal (avec l'interface)
                        if frame == 1:
                            if fullscreen == 1:
                                windowSurfaceObj = pygame.display.set_mode((preview_width + bw, dis_height), pygame.FULLSCREEN, 24)
                            else:
                                windowSurfaceObj = pygame.display.set_mode((preview_width + bw, dis_height), 0, 24)
                        else:
                            windowSurfaceObj = pygame.display.set_mode((preview_width + bw, dis_height), pygame.NOFRAME, 24)

                        # Effacer l'écran et redessiner le menu
                        windowSurfaceObj.fill((0, 0, 0))
                        Menu()
                        pygame.display.update()
                        print("[LIVESTACK] Mode désactivé")

                elif button_row == 4:
                    # STRETCH - Active le mode stretch astro pour le preview
                    stretch_mode = 1
                    # Passer en mode fullscreen pour couvrir aussi la barre de tâches
                    display_modes = pygame.display.list_modes()
                    if display_modes and display_modes != -1:
                        max_width, max_height = display_modes[0]
                    else:
                        screen_info = pygame.display.Info()
                        max_width, max_height = screen_info.current_w, screen_info.current_h
                    windowSurfaceObj = pygame.display.set_mode((max_width, max_height), pygame.FULLSCREEN, 24)

                elif button_row == 5:
                    menu = 1
                    Menu()

                elif button_row == 6:
                    menu = 2
                    Menu()

                elif button_row == 7:
                    # EXIT
                    kill_preview_process()
                    if picam2 is not None:
                        try:
                            picam2.stop()
                            picam2.close()
                        except:
                            pass
                    pygame.display.quit()
                    sys.exit()
                   
                                           
            # MENU 1
            elif menu == 1:
              if button_row == 1:
                  menu = 3
                  Menu()
              elif button_row == 2:
                  menu = 5
                  Menu()
              elif button_row == 3:
                  menu = 6
                  Menu()
              elif button_row == 6:
                  menu = 8
                  Menu()
              elif button_row == 7:
                  menu = 7
                  Menu()

              elif button_row == 4:
                # ZOOM
                for f in range(0,len(video_limits)-1,3):
                    if video_limits[f] == 'zoom':
                        pmin = video_limits[f+1]
                        pmax = video_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    zoom = int(((mousex-preview_width) / bw) * (pmax+1-pmin))###
                    if zoom == 3:
                        zoom = 4  # Sauter le zoom 3
                    # Zoom manuel supprimé: simplifié la condition
                    if igw/igh > 1.5 and alt_dis == 0:
                        pygame.draw.rect(windowSurfaceObj,(0,0,0),Rect(0,int(preview_height * .75),preview_width,preview_height))
                elif (mousey > preview_height + (bh*3)  and mousey < preview_height + (bh*3) + int(bh/3)) and alt_dis == 1:
                    zoom = int(((mousex-((button_row -9)*bw)) / bw) * (pmax+1-pmin))
                    if zoom == 3:
                        zoom = 4  # Sauter le zoom 3
                    # Zoom manuel supprimé: simplifié la condition
                    if igw/igh > 1.5:
                        pygame.draw.rect(windowSurfaceObj,(0,0,0),Rect(0,int(preview_height * .75),preview_width,preview_height/4))
                elif (mousey > preview_height * .75 + (bh*3)  and mousey < preview_height * .75 + (bh*3) + int(bh/3)) and alt_dis == 2:
                    zoom = int(((mousex-((button_row -9)*bw)) / bw) * (pmax+1-pmin))
                    if zoom == 3:
                        zoom = 4  # Sauter le zoom 3
                # Zoom manuel supprimé: simplifié la condition
                elif (alt_dis == 0 and mousex > preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 1):
                    zoom +=1
                    zoom = min(zoom,pmax)
                    if zoom == 3:
                        zoom = 4  # Sauter le zoom 3
                elif alt_dis == 0 and mousex < preview_width + (bw/2)  and zoom > 0:
                    zoom -=1
                    if zoom == 3:
                        zoom = 2  # Sauter le zoom 3
                    # Zoom manuel supprimé: simplifié la condition
                    if igw/igh > 1.5 and alt_dis == 0:
                        pygame.draw.rect(windowSurfaceObj,(0,0,0),Rect(0,int(preview_height * .75),preview_width,preview_height))
                # Synchroniser la résolution vidéo avec le zoom
                sync_video_resolution_with_zoom()
                print(zoom)
                if zoom < 2:
                    if zoom == 0:
                        button(0,4,0,9)
                        text(0,4,5,0,1,"Zoom",ft,7)
                        text(0,4,3,1,1,"",fv,7)
                        draw_Vbar(0,4,greyColor,'zoom',zoom)
                    else:
                        button(0,4,1,9)
                        text(0,4,2,0,1,"ZOOMED",ft,0)
                        text(0,4,3,1,1,zoom_res_labels.get(zoom, str(zoom)),fv,0)
                        draw_Vbar(0,4,dgryColor,'zoom',zoom)

                    if foc_man == 0 and focus_mode == 0:
                        button(0,5,0,9)
                        text(0,5,5,0,1,"FOCUS",ft,7)
                    # determine if camera native format
                    vw = 0
                    x = 0
                    while x < len(vwidths2) and vw == 0:
                        if vwidth == vwidths2[x]:
                             if vheight == vheights2[x]:
                                vw = 1
                        x += 1
                    
                else:
                    button(0,4,1,9)
                    text(0,4,2,0,1,"ZOOMED",ft,0)
                    text(0,4,3,1,1,zoom_res_labels.get(zoom, str(zoom)),fv,0)
                    draw_Vbar(0,4,dgryColor,'zoom',zoom)

                # Maintenir le bouton Focus actif si on est en mode focus
                if focus_mode == 1:
                    # Recentrer le réticule lors du changement de zoom en mode focus
                    xx = int(preview_width/2)
                    xy = int(preview_height/2)
                    button(0,5,1,9)
                    text(0,5,3,0,1,"FOCUS",ft,0)

                if zoom > 0:
                    fxx = 0
                    fxy = 0
                    fxz = 1
                    fyz = 1
                    if (Pi_Cam == 3 and v3_af == 1) and v3_f_mode == 0:
                        text(0,5,3,1,1,str(v3_f_modes[v3_f_mode]),fv,7)
                restart = 1
                time.sleep(.2)
                                         
                             
              elif button_row == 5:
                # FOCUS
                if (Pi_Cam == 3 and v3_af == 1):
                    for f in range(0,len(video_limits)-1,3):
                        if video_limits[f] == 'v3_focus':
                            pmin = video_limits[f+1]
                            pmax = video_limits[f+2]
                if Pi_Cam == 5 or Pi_Cam == 6 or Pi_Cam == 8:
                    if Pi_Cam == 5:
                      for f in range(0,len(video_limits)-1,3):
                        if video_limits[f] == 'v5_focus':
                            pmin = video_limits[f+1]
                            pmax = video_limits[f+2]
                    if Pi_Cam == 6 or Pi_Cam == 8:
                      for f in range(0,len(video_limits)-1,3):
                        if video_limits[f] == 'v6_focus':
                            pmin = video_limits[f+1]
                            pmax = video_limits[f+2]
                # arducam manual focus slider
                if (mousex > preview_width and mousey < ((button_row)*bh) + (bh/3)) and ((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6 or Pi_Cam == 8) and foc_man == 1:
                    focus = int(((mousex-preview_width) / bw) * pmax)
                    if Pi_Cam == 5:
                        draw_Vbar(0,5,dgryColor,'v5_focus',focus)
                    elif Pi_Cam == 6 or Pi_Cam == 8:
                        draw_Vbar(0,5,dgryColor,'v6_focus',focus)
                    v3_focus = focus
                    restart = 1
                    text(0,5,3,0,1,'<<< ' + str(focus) + ' >>>',fv,0)
                #arducam manual focus buttons    
                elif mousex > preview_width and mousey > ((button_row)*bh) + (bh/3) and mousey < ((button_row)*bh) + (bh/1.5) and ((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6 or Pi_Cam == 8) and foc_man == 1:
                    if button_pos == 2:
                        focus -= 10
                        focus = max(focus,pmin)
                    elif button_pos == 3:
                        focus += 10
                        focus = min(focus,pmax)
                    if Pi_Cam == 5:
                        draw_Vbar(0,3,dgryColor,'v5_focus',focus)
                    elif Pi_Cam == 6 or Pi_Cam == 8:
                        draw_Vbar(0,3,dgryColor,'v6_focus',focus)
                    v3_focus = focus
                    restart = 1
                    text(0,5,3,0,1,'<<< ' + str(focus) + ' >>>',fv,0)
                # Pi v3 manual focus slider
                elif (mousex > preview_width and mousey < ((button_row)*bh) + (bh/3)) and (Pi_Cam == 3 and v3_af == 1) and foc_man == 1:
                    v3_focus = int(((mousex-preview_width) / bw) * (pmax+1-pmin)) + pmin
                    draw_Vbar(0,5,dgryColor,'v3_focus',v3_focus-pmin)
                    fd = 1/(v3_focus/100)
                    text(0,5,3,0,1,'<<< ' + str(fd)[0:5] + "m" + ' >>>',fv,0)
                    restart = 1
                # Pi v3 manual focus buttons
                elif mousex > preview_width and mousey > ((button_row)*bh) + (bh/3) and mousey < ((button_row)*bh) + (bh/1.5) and (Pi_Cam == 3 and v3_af == 1)  and foc_man == 1:
                    if button_pos == 2:
                        v3_focus -= 1
                        v3_focus = max(v3_focus,pmin)
                    elif button_pos == 3:
                        v3_focus += 1
                        v3_focus = min(v3_focus,pmax)
                    draw_Vbar(0,3,dgryColor,'v3_focus',v3_focus-pmin)
                    fd = 1/(v3_focus/100)
                    text(0,5,3,0,1,'<<< ' + str(fd)[0:5] + "m" + ' >>>',fv,0)
                    restart = 1

                elif alt_dis == 0:
                    # determine if camera native format
                    vw = 0
                    x = 0
                    while x < len(vwidths2) and vw == 0:
                        if vwidth == vwidths2[x]:
                             if vheight == vheights2[x]:
                                vw = 1
                        x += 1
                    # FOCUS button NON AF camera (ajout IMX585 = Pi_Cam 10)
                    if (Pi_Cam < 3 or Pi_Cam == 4 or Pi_Cam == 7 or Pi_Cam == 9 or Pi_Cam == 10 or (Pi_Cam ==3 and v3_af == 0)) and focus_mode == 0:
                        zoom = 4
                        sync_video_resolution_with_zoom()
                        focus_mode = 1
                        # Recentrer le réticule
                        xx = int(preview_width/2)
                        xy = int(preview_height/2)
                        button(0,5,1,9)
                        text(0,5,3,0,1,"FOCUS",ft,0)
                        button(0,4,1,9)
                        text(0,4,2,0,1,"ZOOMED",ft,0)
                        text(0,4,3,1,1,zoom_res_labels.get(zoom, str(zoom)),fv,0)
                        draw_Vbar(0,4,dgryColor,'zoom',zoom)
                        time.sleep(0.25)
                        restart = 1
                    # CANCEL FOCUS NON AF camera (ajout IMX585 = Pi_Cam 10)
                    elif (Pi_Cam < 3 or Pi_Cam == 4 or Pi_Cam == 7 or Pi_Cam == 9 or Pi_Cam == 10 or (Pi_Cam ==3 and v3_af == 0)) and focus_mode == 1:
                        zoom = 0
                        sync_video_resolution_with_zoom()
                        focus_mode = 0
                        reset_fwhm_history()
                        reset_hfr_history()
                        pygame.draw.rect(windowSurfaceObj,(0,0,0),Rect(0,int(preview_height * .75),preview_width,preview_height/4))
                        button(0,5,0,9)
                        button(0,4,0,9)
                        text(0,5,5,0,1,"FOCUS",ft,7)
                        text(0,5,3,1,1,"",fv,7)
                        text(0,4,5,0,1,"Zoom",ft,7)
                        text(0,4,3,1,1,"",fv,7)
                        draw_Vbar(0,4,greyColor,'zoom',zoom)
                        restart = 1
                    # Pi V3 manual focus
                    elif Pi_Cam == 3 and v3_af == 1 and v3_f_mode == 0:
                        focus_mode = 1
                        v3_f_mode = 1
                        foc_man = 1
                        # Recentrer le réticule
                        xx = int(preview_width/2)
                        xy = int(preview_height/2)
                        restart = 1
                        button(0,5,1,9)
                        time.sleep(0.25)
                        draw_Vbar(0,5,dgryColor,'v3_focus',v3_focus-pmin)
                        fd = 1/(v3_focus/100)
                        text(0,5,3,0,1,'<<< ' + str(fd)[0:5] + "m" + ' >>>',fv,0)
                        text(0,5,3,1,1,str(v3_f_modes[v3_f_mode]),fv,0)
                        time.sleep(0.25)
                    # ARDUCAM manual focus
                    elif ((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6 or Pi_Cam == 8) and v3_f_mode == 0:
                        focus_mode = 1
                        v3_f_mode = 1
                        foc_man = 1
                        # Recentrer le réticule
                        xx = int(preview_width/2)
                        xy = int(preview_height/2)
                        text(0,5,3,0,1,'<<< ' + str(focus) + ' >>>',fv,0)
                        if Pi_Cam == 5:
                            draw_Vbar(0,5,dgryColor,'v5_focus',focus)
                        if Pi_Cam == 6 or Pi_Cam == 8:
                            draw_Vbar(0,5,dgryColor,'v6_focus',focus)
                        text(0,5,3,1,1,"manual",fv,0)
                        time.sleep(0.25)
                        restart = 1
                    # ARDUCAM cancel manual focus
                    elif ((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6 or Pi_Cam == 8) and foc_man == 1:
                        focus_mode = 0
                        reset_fwhm_history()
                        reset_hfr_history()
                        foc_man = 0
                        if Pi_Cam == 8:
                            v3_f_mode = 2 # continuous focus
                        else:
                            v3_f_mode = 0
                        zoom = 0
                        sync_video_resolution_with_zoom()
                        fxx = 0
                        fxy = 0
                        fxz = 1
                        fyz = 0.75
                        button(0,5,0,9)
                        text(0,5,5,0,1,"FOCUS",ft,7)
                        text(0,5,3,1,1,str(v3_f_modes[v3_f_mode]),fv,7)
                        button(0,4,0,9)
                        text(0,4,5,0,1,"Zoom",ft,7)
                        text(0,4,3,1,1,"",fv,7)
                        time.sleep(0.25)
                        restart = 1
                    # Pi V3 cancel manual focus
                    elif (Pi_Cam == 3 and v3_af == 1)  and v3_f_mode == 1:
                        focus_mode = 0
                        reset_fwhm_history()
                        reset_hfr_history()
                        v3_f_mode = 2 # continuous focus
                        foc_man = 0
                        zoom = 0
                        sync_video_resolution_with_zoom()
                        fxx = 0
                        fxy = 0
                        fxz = 1
                        fyz = 0.75
                        pygame.draw.rect(windowSurfaceObj,(0,0,0),Rect(0,0,preview_width,preview_height))
                        button(0,5,0,9)
                        text(0,5,5,0,1,"FOCUS",ft,7)
                        text(0,5,3,1,1,str(v3_f_modes[v3_f_mode]),fv,7)
                        button(0,4,0,9)
                        text(0,4,5,0,1,"Zoom",ft,7)
                        text(0,4,3,1,1,"",fv,7)
                        time.sleep(0.25)
                        restart = 1
                    # AF camera to AUTO
                    elif ((Pi_Cam == 3 and v3_af == 1) or (((Pi_Cam == 5 and v5_af == 1) or Pi_Cam == 6 or Pi_Cam == 8))) and v3_f_mode == 2:
                        focus_mode = 0
                        reset_fwhm_history()
                        reset_hfr_history()
                        v3_f_mode = 0 # auto focus
                        foc_man = 0
                        zoom = 0
                        sync_video_resolution_with_zoom()
                        fxx = 0
                        fxy = 0
                        fxz = 1
                        fyz = 0.75
                        pygame.draw.rect(windowSurfaceObj,(0,0,0),Rect(0,0,preview_width,preview_height))
                        button(0,5,0,9)
                        text(0,5,5,0,1,"FOCUS",ft,7)
                        text(0,5,3,1,1,str(v3_f_modes[v3_f_mode]),fv,7)
                        button(0,4,0,9)
                        text(0,4,5,0,1,"Zoom",ft,7)
                        text(0,4,3,1,1,"",fv,7)
                        time.sleep(0.25)
                        restart = 1
                time.sleep(.25)
                
                              
              elif button_row == 6 and Pi_Cam == 3 and v3_af == 1:
                # V3 FOCUS SPEED 
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'v3_f_speed':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    v3_f_speed = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height + bh  and mousey < preview_height + bh + int(bh/3)) and alt_dis == 1:
                    v3_f_speed = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75 + bh  and mousey < preview_height * .75 + bh + int(bh/3)) and alt_dis == 2:
                    v3_f_speed = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        v3_f_speed-=1
                        v3_f_speed = max(v3_f_speed,pmin)
                    else:
                        v3_f_speed +=1
                        v3_f_speed = min(v3_f_speed,pmax)
                text(0,7,3,1,1,v3_f_speeds[v3_f_speed],fv,7)
                draw_bar(0,6,greyColor,'v3_f_speed',v3_f_speed)
                restart = 1
                time.sleep(.25)
                
              elif button_row == 7 and Pi_Cam == 3 and v3_af == 1:
                # V3 FOCUS RANGE 
                for f in range(0,len(video_limits)-1,3):
                    if video_limits[f] == 'v3_f_range':
                        pmin = video_limits[f+1]
                        pmax = video_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    v3_f_range = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height + bh  and mousey < preview_height + (bh*3) + int(bh/3)) and alt_dis == 1:
                    v3_f_range = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75 + bh  and mousey < preview_height * .75 + (bh*3) + int(bh/3)) and alt_dis == 2:
                    v3_f_range = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        v3_f_range-=1
                        v3_f_range = max(v3_f_range,pmin)
                    else:
                        v3_f_range +=1
                        v3_f_range = min(v3_f_range,pmax)
                text(0,7,3,1,1,v3_f_ranges[v3_f_range],fv,7)
                draw_Vbar(0,7,greyColor,'v3_f_range',v3_f_range)
                restart = 1
                time.sleep(.25)
            
            elif menu == 2:
              if button_row == 1 and cam1 != "1":
                # SWITCH CAMERA
                camera += 1
                if camera > max_camera:
                    camera = 0
                text(0,1,3,1,1,str(camera),fv,7)
                if not use_picamera2 and p is not None:
                    poll = p.poll()
                    if poll == None:
                        os.killpg(p.pid, signal.SIGTERM)
                focus_mode = 0
                reset_fwhm_history()
                reset_hfr_history()
                v3_f_mode = 0 
                foc_man = 0
                fxx = 0
                fxy = 0
                fxz = 1
                fyz = 1
                Camera_Version()
                Menu()
                restart = 1
                
              elif button_row == 2:
                # EXT TRIGGER 
                for f in range(0,len(video_limits)-1,3):
                    if video_limits[f] == 'str_cap':
                        pmin = video_limits[f+1]
                        pmax = video_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    str_cap = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height + bh  and mousey < preview_height + (bh*3) + int(bh/3)) and alt_dis == 1:
                    str_cap = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75 + bh  and mousey < preview_height * .75 + (bh*3) + int(bh/3)) and alt_dis == 2:
                    str_cap = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        str_cap -=1
                        str_cap = max(str_cap,pmin)
                    else:
                        str_cap +=1
                        str_cap = min(str_cap,pmax)
                text(0,2,3,1,1,strs[str_cap],fv,7)
                draw_Vbar(0,2,greyColor,'str_cap',str_cap)
                restart = 1
                time.sleep(.25)
                
              elif button_row == 3:
                # HISTOGRAM 
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'histogram':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    histogram = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height + bh  and mousey < preview_height + bh + int(bh/3)) and alt_dis == 1:
                    histogram = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75 + bh  and mousey < preview_height * .75 + bh + int(bh/3)) and alt_dis == 2:
                    histogram = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        histogram -=1
                        histogram = max(histogram,pmin)
                    else:
                        histogram +=1
                        histogram = min(histogram,pmax)
                text(0,3,3,1,1,histograms[histogram],fv,7)
                draw_bar(0,3,greyColor,'histogram',histogram)
                time.sleep(.25)
                
              elif button_row == 4:
                # HISTOGRAM SIZE
                for f in range(0,len(video_limits)-1,3):
                    if video_limits[f] == 'histarea':
                        pmin = video_limits[f+1]
                        pmax = video_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    histarea = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                    histarea = max(histarea,pmin)
                elif (mousey > preview_height + (bh*3)  and mousey < preview_height + (bh*3) + int(bh/3)) and alt_dis == 1:
                    histarea = int(((mousex-((button_row -9)*bw)) / bw) * (pmax+1-pmin))
                    histarea = max(histarea,pmin)
                elif (mousey > preview_height * .75 + (bh*3)  and mousey < preview_height * .75 + (bh*3) + int(bh/3)) and alt_dis == 2:
                    histarea = int(((mousex-((button_row -9)*bw)) / bw) * (pmax+1-pmin))
                    histarea = max(histarea,pmin)
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        histarea -=1
                        histarea = max(histarea,pmin)
                    else:
                        histarea +=1
                        histarea = min(histarea,pmax)
                if xx - histarea < 0 or xy - histarea < 0:
                    histarea = old_histarea
                if xy + histarea > preview_height or xx + histarea > preview_width:
                    histarea = old_histarea
                if (Pi_Cam == 3 and v3_af == 1) and (xy + histarea > preview_height * 0.75 or xx + histarea > preview_width):
                    histarea = old_histarea
                text(0,4,3,1,1,str(histarea),fv,7)
                draw_Vbar(0,4,greyColor,'histarea',histarea)
                old_histarea = histarea
                time.sleep(.25) 
                
              elif button_row == 5:
                # VERTICAL FLIP
                vflip +=1
                if vflip > 1:
                    vflip = 0
                text(0,5,3,1,1,str(vflip),fv,7)
                restart = 1
                time.sleep(.25)
                
              elif button_row == 6:
                # HORIZONTAL FLIP
                hflip += 1
                if hflip > 1:
                    hflip = 0
                text(0,6,3,1,1,str(hflip),fv,7)
                restart = 1
                time.sleep(.25)
                        
              elif button_row == 7:
                # timet
                if alt_dis == 0 and mousex < preview_width + (bw/2):
                    timet -=100
                    timet  = max(timet ,100)
                else:
                    timet  +=100
                    timet = min(timet ,10000)
                text(0,7,3,1,1,str(timet),fv,7)
                time.sleep(0.05) 
                
              elif button_row == 8:
                   # SAVE CONFIG
                   text(0,8,3,0,1,"SAVE Config",fv,7)
                   config[0] = mode
                   config[1] = speed
                   config[2] = gain
                   config[3] = int(brightness)
                   config[4] = int(contrast)
                   config[5] = frame
                   config[6] = int(red)
                   config[7] = int(blue)
                   config[8] = ev
                   config[9] = vlen
                   config[10] = fps
                   config[11] = vformat
                   config[12] = codec
                   config[13] = tinterval
                   config[14] = tshots
                   config[15] = extn
                   config[16] = zx
                   config[17] = zy
                   config[18] = zoom
                   config[19] = int(saturation)
                   config[20] = meter
                   config[21] = awb
                   config[22] = sharpness
                   config[23] = int(denoise)
                   config[24] = quality
                   config[25] = profile
                   config[26] = level
                   config[27] = histogram
                   config[28] = histarea
                   config[29] = v3_f_speed
                   config[30] = v3_f_range
                   config[31] = rotate
                   config[32] = IRF
                   config[33] = str_cap
                   config[34] = v3_hdr
                   config[35] = timet
                   config[36] = vflip
                   config[37] = hflip
                   config[38] = stretch_p_low
                   config[39] = stretch_p_high
                   config[40] = stretch_factor
                   config[41] = stretch_preset
                   with open(config_file, 'w') as f:
                      for item in range(0,len(titles)):
                          f.write(titles[item] + " : " + str(config[item]) + "\n")
                   time.sleep(1)
                   text(0,8,2,0,1,"SAVE CONFIG",fv,7)

            # MENU 3    
            elif menu == 3: 
              if button_row == 9:
                  menu = 4
                  Menu()
              elif button_row == 1:
                # MODE
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'mode':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    mode = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height  and mousey < preview_height + int(bh/3)) and alt_dis == 1:
                    mode = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75 and mousey < preview_height * .75 + int(bh/3)) and alt_dis == 2:
                    mode = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        mode -=1
                        mode  = max(mode ,pmin)
                    else:
                        mode  +=1
                        mode = min(mode ,pmax)
                if mode == 0:
                    text(0,2,5,0,1,"Shutter S",ft,10)
                    draw_bar(0,2,lgrnColor,'speed',speed)
                    if shutters[speed] < 0:
                        text(0,2,3,1,1,"1/" + str(abs(shutters[speed])),fv,10)
                    else:
                        text(0,2,3,1,1,str(shutters[speed]),fv,10)
                    if gain == 0:
                        gain = 1
                        text(0,3,5,0,1,"Gain    A/D",ft,10)
                        if gain <= mag:
                            text(0,3,3,1,1,str(gain) + " :  " + str(gain) + "/1",fv,10)
                        else:
                            text(0,3,3,1,1,str(gain) + " :  " + str(int(mag)) + "/" + str(((gain/mag)*10)/10)[0:3],fv,10)
                        draw_bar(0,3,lgrnColor,'gain',gain)
                else:
                    text(0,2,5,0,1,"eV",ft,10)
                    text(0,2,3,1,1,str(ev),fv,10)
                    draw_bar(0,2,lgrnColor,'ev',ev)
                    gain = 0
                    text(0,3,5,0,1,"Gain ",ft,10)
                    text(0,3,3,1,1,"Auto",fv,10)
                    draw_bar(0,3,lgrnColor,'gain',gain)
                text(0,1,3,1,1,modes[mode],fv,10)
                draw_bar(0,2,lgrnColor,'mode',mode)
                td = timedelta(seconds=tinterval)
                if tinterval > 0:
                    tduration = tinterval * tshots
                if mode == 0 and tinterval == 0 :
                    speed = 15
                    shutter = shutters[speed]
                    if shutter < 0:
                        shutter = abs(1/shutter)
                    sspeed = int(shutter * 1000000)
                    if (shutter * 1000000) - int(shutter * 1000000) > 0.5:
                        sspeed +=1
                    if shutters[speed] < 0:
                        text(0,2,3,1,1,"1/" + str(abs(shutters[speed])),fv,10)
                    else:
                        text(0,2,3,1,1,str(shutters[speed]),fv,10)
                    draw_bar(0,2,lgrnColor,'speed',speed)

                time.sleep(.25)
                restart = 1

              elif button_row == 2:
                # SHUTTER SPEED or EV (dependent on MODE set)
                if mode == 0 :
                    for f in range(0,len(still_limits)-1,3):
                        if still_limits[f] == 'speed':
                            pmin = still_limits[f+1]
                            pmax = max_speed
                    if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                        speed = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                    elif (mousey > preview_height  and mousey < preview_height + int(bh/3)) and alt_dis == 1:
                        speed = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                    elif (mousey > preview_height * .75 and mousey < preview_height * .75  + int(bh/3)) and alt_dis == 2:
                        speed = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                    else:
                        if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                            speed -=1
                            speed  = max(speed ,pmin)
                        else:
                            speed  +=1
                            speed = min(speed ,pmax)
                    shutter = shutters[speed]
                    if shutter < 0:
                        shutter = abs(1/shutter)
                    sspeed = int(shutter * 1000000)
                    if (shutter * 1000000) - int(shutter * 1000000) > 0.5:
                        sspeed +=1
                    if shutters[speed] < 0:
                        text(0,2,3,1,1,"1/" + str(abs(shutters[speed])),fv,10)
                    else:
                        text(0,2,3,1,1,str(shutters[speed]),fv,10)
                    draw_bar(0,2,lgrnColor,'speed',speed)
                    if tinterval > 0:
                        tinterval = int(sspeed/1000000)
                        tinterval = max(tinterval,1)
                        td = timedelta(seconds=tinterval)
                        tduration = tinterval * tshots
                        td = timedelta(seconds=tduration)
                        
                    time.sleep(.25)
                    restart = 1
                else:
                    # EV
                    for f in range(0,len(still_limits)-1,3):
                        if still_limits[f] == 'ev':
                            pmin = still_limits[f+1]
                            pmax = still_limits[f+2]
                    if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                        ev = int(((mousex-preview_width) / bw) * (pmax+1-pmin)) + pmin 
                    elif (mousey > preview_height  and mousey < preview_height + int(bh/3)) and alt_dis == 1:
                        ev = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin)) + pmin
                    elif (mousey > preview_height * .75  and mousey < preview_height * .75 + int(bh/3)) and alt_dis == 2:
                        ev = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin)) + pmin
                    else:
                        if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                            ev -=1
                            ev  = max(ev ,pmin)
                        else:
                            ev  +=1
                            ev = min(ev ,pmax)
                    text(0,2,3,1,1,str(ev),fv,10)
                    draw_bar(0,2,lgrnColor,'ev',ev)
                    time.sleep(0.25)
                    restart = 1
                    
              elif button_row == 3:
                # GAIN
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'gain':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    slider_val = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                    gain = slider_to_gain_nonlinear(slider_val, pmax)
                elif (mousey > preview_height  and mousey < preview_height + int(bh/3)) and alt_dis == 1:
                    slider_val = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                    gain = slider_to_gain_nonlinear(slider_val, pmax)
                elif (mousey > preview_height * .75  and mousey < preview_height * .75 + int(bh/3)) and alt_dis == 2:
                    slider_val = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                    gain = slider_to_gain_nonlinear(slider_val, pmax)
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        # Bouton - : ajuster avec courbe non-linéaire
                        slider_pos = gain_to_slider_nonlinear(gain, pmax)
                        slider_pos = max(slider_pos - 1, pmin)
                        gain = slider_to_gain_nonlinear(slider_pos, pmax)
                    else:
                        # Bouton + : ajuster avec courbe non-linéaire
                        slider_pos = gain_to_slider_nonlinear(gain, pmax)
                        slider_pos = min(slider_pos + 1, pmax)
                        gain = slider_to_gain_nonlinear(slider_pos, pmax)
                if gain > 0:
                    text(0,3,5,0,1,"Gain    A/D",ft,10)
                    if gain <= mag:
                        text(0,3,3,1,1,str(gain) + " :  " + str(gain) + "/1",fv,10)
                    else:
                        text(0,3,3,1,1,str(gain) + " :  " + str(int(mag)) + "/" + str(((gain/mag)*10)/10)[0:3],fv,10)
                else:
                    if gain == 0:
                        text(0,3,5,0,1,"Gain ",ft,10)
                    else:
                        text(0,3,5,0,1,"Gain    A/D",ft,10)
                    text(0,3,3,1,1,"Auto",fv,10)
                time.sleep(.25)
                draw_bar(0,3,lgrnColor,'gain',gain)
                restart = 1
                
              elif button_row == 4:
                # BRIGHTNESS
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'brightness':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    brightness = int(((mousex-preview_width) / bw) * (pmax+1-pmin)) + pmin 
                elif (mousey > preview_height  and mousey < preview_height + int(bh/3)) and alt_dis == 1:
                    brightness = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin)) + pmin
                elif (mousey > preview_height * .75  and mousey < preview_height * .75+ int(bh/3)) and alt_dis == 2:
                    brightness = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin)) + pmin 
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        brightness -=1
                        brightness  = max(brightness ,pmin)
                    else:
                        brightness  +=1
                        brightness = min(brightness ,pmax)
                text(0,4,3,1,1,str(brightness/100),fv,10)
                draw_bar(0,4,lgrnColor,'brightness',brightness)
                time.sleep(0.025)
                restart = 1
                
              elif button_row == 5:
                # CONTRAST
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'contrast':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    contrast = int(((mousex-preview_width) / bw) * (pmax+1-pmin)) 
                elif (mousey > preview_height  and mousey < preview_height + int(bh/3)) and alt_dis == 1:
                    contrast = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75  and mousey < preview_height * .75 + int(bh/3)) and alt_dis == 2:
                    contrast = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        contrast -=1
                        contrast  = max(contrast ,pmin)
                    else:
                        contrast  +=1
                        contrast = min(contrast ,pmax)
                text(0,5,3,1,1,str(contrast/100)[0:4],fv,10)
                draw_bar(0,5,lgrnColor,'contrast',contrast)
                time.sleep(0.025)
                restart = 1
                
              elif button_row == 6:
                # AWB
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'awb':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    awb = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height and mousey < preview_height + int(bh/3)) and alt_dis == 1:
                    awb = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75 and mousey < preview_height * .75 + int(bh/3)) and alt_dis == 2:
                    awb = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        awb -=1
                        awb  = max(awb ,pmin)
                    else:
                        awb  +=1
                        awb = min(awb ,pmax)
                text(0,7,3,1,1,awbs[awb],fv,10)
                draw_bar(0,6,lgrnColor,'awb',awb)
                text(0,7,5,0,1,"Blue",ft,10)
                text(0,8,5,0,1,"Red",ft,10)
                text(0,8,3,1,1,str(red/10)[0:3],fv,10)
                text(0,7,3,1,1,str(blue/10)[0:3],fv,10)
                draw_bar(0,7,lgrnColor,'blue',blue)
                draw_bar(0,8,lgrnColor,'red',red)
                time.sleep(.25)
                restart = 1
                
              elif button_row == 7:
                # BLUE
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'blue':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    blue = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height  and mousey < preview_height + int(bh/3)) and alt_dis == 1:
                    blue = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75  and mousey < preview_height * .75 + int(bh/3)) and alt_dis == 2:
                    blue = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        blue -=1
                        blue  = max(blue ,pmin)
                    else:
                        blue  +=1
                        blue = min(blue ,pmax)
                text(0,7,3,1,1,str(blue/10)[0:3],fv,10)
                draw_bar(0,7,lgrnColor,'blue',blue)
                time.sleep(.25)
                restart = 1


              elif button_row == 8 :
                # RED
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'red':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    red = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height + bh  and mousey < preview_height + bh + int(bh/3)) and alt_dis == 1:
                    red = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75 + bh  and mousey < preview_height * .75 + bh + int(bh/3)) and alt_dis == 2:
                    red = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        red -=1
                        red  = max(red ,pmin)
                    else:
                        red  +=1
                        red = min(red ,pmax)
                text(0,8,3,1,1,str(red/10)[0:3],fv,10)
                draw_bar(0,8,lgrnColor,'red',red)
                time.sleep(.25)
                restart = 1
                           
            # MENU 4
            elif menu == 4:
              if button_row == 1:
                  # Page 1 - retour au menu 3
                  menu = 3
                  Menu()

              elif button_row == 2:
                # METER
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'meter':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    meter = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height + bh  and mousey < preview_height + bh + int(bh/3)) and alt_dis == 1:
                    meter = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75 + bh  and mousey < preview_height * .75 + bh + int(bh/3)) and alt_dis == 2:
                    meter = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        meter -=1
                        meter  = max(meter ,pmin)
                    else:
                        meter  +=1
                        meter = min(meter ,pmax)
                text(0,2,3,1,1,meters[meter],fv,10)
                draw_bar(0,2,lgrnColor,'meter',meter)
                time.sleep(.25)
                restart = 1
              
              elif button_row == 3:
                # QUALITY
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'quality':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    quality = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height + bh  and mousey < preview_height + bh + int(bh/3)) and alt_dis == 1:
                    quality = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75 + bh  and mousey < preview_height * .75 + bh + int(bh/3)) and alt_dis == 2:
                    quality = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        quality -=1
                        quality  = max(quality ,pmin)
                    else:
                        quality  +=1
                        quality = min(quality ,pmax)
                text(0,3,3,1,1,str(quality)[0:3],fv,10)
                draw_bar(0,3,lgrnColor,'quality',quality)
                time.sleep(.25)
                restart = 1

              elif button_row == 4:
                # SATURATION
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'saturation':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    saturation = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height + bh  and mousey < preview_height + bh + int(bh/3)) and alt_dis == 1:
                    saturation = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75 + bh  and mousey < preview_height * .75 + bh + int(bh/3)) and alt_dis == 2:
                    saturation = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        saturation -=1
                        saturation  = max(saturation ,pmin)
                    else:
                        saturation  +=1
                        saturation = min(saturation ,pmax)
                text(0,4,3,1,1,str(saturation/10),fv,10)
                draw_bar(0,4,lgrnColor,'saturation',saturation)
                time.sleep(.25)
                restart = 1

              elif button_row == 5:
                # DENOISE
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'denoise':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    denoise = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height  and mousey < preview_height + int(bh/3)) and alt_dis == 1:
                    denoise = int(((mousex-((button_row -1)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75  and mousey < preview_height * .75 + int(bh/3)) and alt_dis == 2:
                    denoise = int(((mousex-((button_row -1)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        denoise -=1
                        denoise = max(denoise,pmin)
                    else:
                        denoise +=1
                        denoise = min(denoise,pmax)
                text(0,5,3,1,1,denoises[denoise],fv,10)
                draw_bar(0,5,lgrnColor,'denoise',denoise)
                time.sleep(.25)
                restart = 1

              elif button_row == 6:
                # SHARPNESS
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'sharpness':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    sharpness = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height + (bh)  and mousey < preview_height + (bh) + int(bh/3)) and alt_dis == 1:
                    sharpness = int(((mousex-((button_row -9)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75 + (bh)  and mousey < preview_height * .75 + (bh) + int(bh/3)) and alt_dis == 2:
                    sharpness = int(((mousex-((button_row -9)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        sharpness -=1
                        sharpness = max(sharpness,pmin)
                    else:
                        sharpness +=1
                        sharpness = min(sharpness,pmax)

                text(0,6,3,1,1,str(sharpness/10),fv,10)
                draw_bar(0,6,lgrnColor,'sharpness',sharpness)
                time.sleep(.25)
                restart = 1
                
              elif button_row == 7 and Pi_Cam == 9:
                # Waveshare imx290 IR Filter
                if mousex < preview_width + (bw/2):
                    IRF -=1
                    IRF = max(IRF ,0)
                else:
                    IRF  +=1
                    IRF = min(IRF ,1)
                if IRF == 0:
                    text(0,7,3,1,1,"Off",fv,10)
                    led_sw_ir.off()
                else:
                    text(0,7,3,1,1,"ON ",fv,10)
                    led_sw_ir.on()
                time.sleep(0.25)
                restart = 1
                   
              elif button_row == 7 and Pi_Cam == 4 and scientif == 1:
                # v4 (HQ) CAMERA Scientific.json
                if alt_dis == 0 and mousex < preview_width + (bw/2):
                    scientific -=1
                    scientific = max(scientific ,0)
                else:
                    scientific  +=1
                    scientific = min(scientific ,1)
                text(0,7,5,0,1,"Scientific",fv,10)
                if scientific == 0:
                    text(0,7,3,1,1,"Off",fv,10)
                else:
                    text(0,7,3,1,1,"ON ",fv,10)
                time.sleep(0.25)
                restart = 1

                            
              elif button_row == 7 and Pi_Cam == 3:
                # PI V3 CAMERA HDR
                if alt_dis == 0 and mousex < preview_width + (bw/2):
                    v3_hdr -=1
                    v3_hdr  = max(v3_hdr ,0)
                else:
                    v3_hdr  +=1
                    v3_hdr = min(v3_hdr ,3)

                text(0,7,5,0,1,"HDR",fv,10)
                text(0,7,3,1,1,v3_hdrs[v3_hdr],fv,10)
                time.sleep(0.25)
                restart = 1

              elif button_row == 7 and Pi_Cam != 3 and Pi == 5:
                # PI5 and NON V3 CAMERA HDR
                if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                    v3_hdr -=1
                    v3_hdr  = max(v3_hdr ,0)
                else:
                    v3_hdr  +=1
                    v3_hdr = min(v3_hdr ,3)  # Correction: permettre tous les modes HDR (0-3) incluant "sensor"

                text(0,7,5,0,1,"HDR",fv,10)
                text(0,7,3,1,1,v3_hdrs[v3_hdr],fv,10)
                time.sleep(0.25)
                restart = 1

              elif button_row == 8:
                # FILE FORMAT
                for f in range(0,len(still_limits)-1,3):
                    if still_limits[f] == 'extn':
                        pmin = still_limits[f+1]
                        pmax = still_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    extn = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height + bh  and mousey < preview_height + bh + int(bh/3)) and alt_dis == 1:
                    extn = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75 + bh  and mousey < preview_height * .75 + bh + int(bh/3)) and alt_dis == 2:
                    extn = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        extn -=1
                        extn  = max(extn ,pmin)
                    else:
                        extn  +=1
                        extn = min(extn ,pmax)
                text(0,8,3,1,1,extns[extn],fv,10)
                draw_bar(0,8,lgrnColor,'extn',extn)
                time.sleep(.25)
                restart = 1

              elif button_row == 9:
                   # SAVE CONFIG
                   text(0,9,3,0,1,"SAVE Config",fv,10)
                   config[0] = mode
                   config[1] = speed
                   config[2] = gain
                   config[3] = int(brightness)
                   config[4] = int(contrast)
                   config[5] = frame
                   config[6] = int(red)
                   config[7] = int(blue)
                   config[8] = ev
                   config[9] = vlen
                   config[10] = fps
                   config[11] = vformat
                   config[12] = codec
                   config[13] = tinterval
                   config[14] = tshots
                   config[15] = extn
                   config[16] = zx
                   config[17] = zy
                   config[18] = zoom
                   config[19] = int(saturation)
                   config[20] = meter
                   config[21] = awb
                   config[22] = sharpness
                   config[23] = int(denoise)
                   config[24] = quality
                   config[25] = profile
                   config[26] = level
                   config[27] = histogram
                   config[28] = histarea
                   config[29] = v3_f_speed
                   config[30] = v3_f_range
                   config[31] = rotate
                   config[32] = IRF
                   config[33] = str_cap
                   config[34] = v3_hdr
                   config[35] = timet
                   config[36] = vflip
                   config[37] = hflip
                   config[38] = stretch_p_low
                   config[39] = stretch_p_high
                   config[40] = stretch_factor
                   config[41] = stretch_preset
                   with open(config_file, 'w') as f:
                      for item in range(0,len(titles)):
                          f.write(titles[item] + " : " + str(config[item]) + "\n")
                   time.sleep(1)
                   text(0,8,2,0,1,"SAVE CONFIG",fv,10)    
                                        
            # MENU 5
            elif menu == 5:   
              if button_row == 1:
                # VIDEO LENGTH
                for f in range(0,len(video_limits)-1,3):
                    if video_limits[f] == 'vlen':
                        pmin = video_limits[f+1]
                        pmax = video_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    vlen = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height  + (bh*2) and mousey < preview_height + (bh*2) + int(bh/3)) and alt_dis == 1:
                    vlen = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75  + (bh*2) and mousey < preview_height * .75 + (bh*2) + int(bh/3)) and alt_dis == 2:
                    vlen = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if mousex < preview_width + (bw/2):
                        vlen -=1
                        vlen  = max(vlen ,pmin)
                    else:
                        vlen  +=1
                        vlen = min(vlen ,pmax)
                td = timedelta(seconds=vlen)
                text(0,1,3,1,1,str(td),fv,11)
                draw_Vbar(0,1,lpurColor,'vlen',vlen)
                time.sleep(.25)
 
              elif button_row == 2:
                # FPS
                for f in range(0,len(video_limits)-1,3):
                    if video_limits[f] == 'fps':
                        pmin = video_limits[f+1]
                        pmax = video_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    fps = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                    fps = min(fps,vfps)
                    fps = max(fps,pmin)
                elif (mousey > preview_height  + (bh*2) and mousey < preview_height + (bh*2) + int(bh/3)) and alt_dis == 1:
                    fps = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                    fps = min(fps,vfps)
                    fps = max(fps,pmin)
                elif (mousey > preview_height * .75  + (bh*2) and mousey < preview_height  * .75 + (bh*2) + int(bh/3)) and alt_dis == 2:
                    fps = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                    fps = min(fps,vfps)
                    fps = max(fps,pmin)
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        fps -=1
                        fps  = max(fps ,pmin)
                    else:
                        fps  +=1
                        fps = min(fps ,pmax)
                
                text(0,2,3,1,1,str(fps),fv,11)
                draw_Vbar(0,2,lpurColor,'fps',fps)
                time.sleep(.25)
                restart = 1
                   
              elif button_row == 3:
                # VFORMAT
                for f in range(0,len(video_limits)-1,3):
                    if video_limits[f] == 'vformat':
                        pmin = video_limits[f+1]
                        pmax = video_limits[f+2]
                
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    # set max video format
                    setmaxvformat()
                    pmax = max_vformat
                    vformat = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height  + (bh*2) and mousey < preview_height + (bh*2) + int(bh/3)) and alt_dis == 1:
                    # set max video format    
                    setmaxvformat()
                    pmax = max_vformat
                    vformat = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75  + (bh*2) and mousey < preview_height * .75 + (bh*2) + int(bh/3)) and alt_dis == 2:
                    # set max video format    
                    setmaxvformat()
                    pmax = max_vformat
                    vformat = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        vformat -=1
                        # set max video format    
                        setmaxvformat()
                        vformat = min(vformat,max_vformat)
                    else:
                        vformat +=1
                        # set max video format
                        setmaxvformat()
                        vformat = min(vformat,max_vformat)
                draw_Vbar(0,3,lpurColor,'vformat',vformat)
                vwidth  = vwidths[vformat]
                vheight = vheights[vformat]
                # Activer l'affichage temporaire du ROI (5 secondes pour compenser le restart)
                show_roi_until = time.time() + 5.0
                if Pi_Cam == 3:
                    vfps = v3_max_fps[vformat]
                    if vwidth == 1920 and codec == 0:
                        prof = h264profiles[profile].split(" ")
                        if str(prof[1]) == "4.2":
                            if vpreview == 1:
                                vfps = 45
                            else:
                                vfps = 60
                    elif vwidth == 1536 and codec == 0:
                        prof = h264profiles[profile].split(" ")
                        if str(prof[1]) == "4.2":
                            if vpreview == 1:
                                vfps = 60
                            else:
                                vfps = 90
                elif Pi_Cam == 9:
                    vfps = v9_max_fps[vformat]
                elif Pi_Cam == 15:
                    vfps = v15_max_fps[vformat]
                else:
                    vfps = v_max_fps[vformat]
                fps = min(fps,vfps)
                video_limits[5] = vfps
                text(0,2,3,1,1,str(fps),fv,11)
                draw_Vbar(0,2,lpurColor,'fps',fps)
                # determine if camera native format
                vw = 0
                x = 0
                while x < len(vwidths2) and vw == 0:
                    if vwidth == vwidths2[x]:
                        if vheight == vheights2[x]:
                            vw = 1
                    x += 1
                if vw == 0:
                    text(0,3,3,1,1,str(vwidth) + "x" + str(vheight),fv,11)
                if vw == 1:
                    text(0,3,1,1,1,str(vwidth) + "x" + str(vheight),fv,11)
                time.sleep(.25)
                restart = 1

              elif button_row == 4:
                # CODEC
                for f in range(0,len(video_limits)-1,3):
                    if video_limits[f] == 'codec':
                        pmin = video_limits[f+1]
                        pmax = video_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    codec = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height + (bh*2) and mousey < preview_height + (bh*2) + int(bh/3)) and alt_dis == 1:
                    codec = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75 + (bh*2) and mousey < preview_height * .75 + (bh*2) + int(bh/3)) and alt_dis == 2:
                    codec = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        codec -=1
                        codec  = max(codec ,pmin)
                    else:
                        codec  +=1
                        codec = min(codec ,pmax)
                # set max video format
                setmaxvformat()
                vformat = min(vformat,max_vformat)
                text(0,4,3,1,1,codecs[codec],fv,11)
                draw_Vbar(0,4,lpurColor,'codec',codec)
                draw_Vbar(0,3,lpurColor,'vformat',vformat)
                vwidth  = vwidths[vformat]
                vheight = vheights[vformat]
                # Activer l'affichage temporaire du ROI (5 secondes pour compenser le restart)
                show_roi_until = time.time() + 5.0
                if Pi_Cam == 3:
                    vfps = v3_max_fps[vformat]
                    if vwidth == 1920 and codec == 0:
                        prof = h264profiles[profile].split(" ")
                        if str(prof[1]) == "4.2":
                            if vpreview == 1:
                                vfps = 45
                            else:
                                vfps = 60
                    elif vwidth == 1536 and codec == 0:
                        prof = h264profiles[profile].split(" ")
                        if str(prof[1]) == "4.2":
                            if vpreview == 1:
                                vfps = 60
                            else:
                                vfps = 90
                elif Pi_Cam == 9:
                    vfps = v9_max_fps[vformat]
                elif Pi_Cam == 15:
                    vfps = v15_max_fps[vformat]
                else:
                    vfps = v_max_fps[vformat]
                fps = min(fps,vfps)
                video_limits[5] = vfps
                text(0,2,3,1,1,str(fps),fv,11)
                draw_Vbar(0,2,lpurColor,'fps',fps)
                # determine if camera native format
                vw = 0
                x = 0
                while x < len(vwidths2) and vw == 0:
                    if vwidth == vwidths2[x]:
                        if vheight == vheights2[x]:
                            vw = 1
                    x += 1
                if vw == 0:
                    text(0,3,3,1,1,str(vwidth) + "x" + str(vheight),fv,11)
                if vw == 1:
                    text(0,3,1,1,1,str(vwidth) + "x" + str(vheight),fv,11)
                time.sleep(.25)

              elif button_row == 5:
                # H264 PROFILE
                for f in range(0,len(video_limits)-1,3):
                    if video_limits[f] == 'profile':
                        pmin = video_limits[f+1]
                        pmax = video_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    profile = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height + (bh*2) and mousey < preview_height + (bh*2) + int(bh/3)) and alt_dis == 1:
                    profile = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75 + (bh*2) and mousey < preview_height * .75 + (bh*2) + int(bh/3)) and alt_dis == 2:
                    profile = int(((mousex-((button_row - 1)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        profile -=1
                        profile  = max(profile ,pmin)
                    else:
                        profile  +=1
                        profile = min(profile ,pmax)
                text(0,5,3,1,1,h264profiles[profile],fv,11)
                draw_Vbar(0,5,lpurColor,'profile',profile)
                vwidth  = vwidths[vformat]
                vheight = vheights[vformat]
                if Pi_Cam == 3:
                    vfps = v3_max_fps[vformat]
                    if vwidth == 1920 and codec == 0:
                        prof = h264profiles[profile].split(" ")
                        if str(prof[1]) == "4.2":
                            if vpreview == 1:
                                vfps = 45
                            else:
                                vfps = 60
                    elif vwidth == 1536 and codec == 0:
                        prof = h264profiles[profile].split(" ")
                        if str(prof[1]) == "4.2":
                            if vpreview == 1:
                                vfps = 60
                            else:
                                vfps = 90
                elif Pi_Cam == 9:
                    vfps = v9_max_fps[vformat]
                elif Pi_Cam == 15:
                    vfps = v15_max_fps[vformat]
                else:
                    vfps = v_max_fps[vformat]
                fps = min(fps,vfps)
                video_limits[5] = vfps
                text(0,2,3,1,1,str(fps),fv,11)
                draw_Vbar(0,2,lpurColor,'fps',fps)
                time.sleep(.25)

              elif button_row == 6:
                # V_PREVIEW
                if (alt_dis == 0 and mousex > preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                    vpreview +=1
                    vpreview  = min(vpreview ,1)
                else:
                    vpreview  -=1
                    vpreview = max(vpreview ,0)

                if vpreview == 0:
                    text(0,7,3,1,1,"Off",fv,11)
                else:
                    text(0,7,3,1,1,"ON ",fv,11)
                vwidth  = vwidths[vformat]
                vheight = vheights[vformat]
                if Pi_Cam == 3:
                    vfps = v3_max_fps[vformat]
                    if vwidth == 1920 and codec == 0:
                        prof = h264profiles[profile].split(" ")
                        if str(prof[1]) == "4.2":
                            if vpreview == 1:
                                vfps = 45
                            else:
                                vfps = 60
                    elif vwidth == 1536 and codec == 0:
                        prof = h264profiles[profile].split(" ")
                        if str(prof[1]) == "4.2":
                            if vpreview == 1:
                                vfps = 60
                            else:
                                vfps = 90
                elif Pi_Cam == 9:
                    vfps = v9_max_fps[vformat]
                elif Pi_Cam == 15:
                    vfps = v15_max_fps[vformat]
                else:
                    vfps = v_max_fps[vformat]
                fps = min(fps,vfps)
                video_limits[5] = vfps
                text(0,2,3,1,1,str(fps),fv,11)
                draw_Vbar(0,2,lpurColor,'fps',fps)
                time.sleep(0.25)
                
              elif button_row == 8:
                   # SAVE CONFIG
                   text(0,8,3,0,1,"SAVE Config",fv,11)
                   config[0] = mode
                   config[1] = speed
                   config[2] = gain
                   config[3] = int(brightness)
                   config[4] = int(contrast)
                   config[5] = frame
                   config[6] = int(red)
                   config[7] = int(blue)
                   config[8] = ev
                   config[9] = vlen
                   config[10] = fps
                   config[11] = vformat
                   config[12] = codec
                   config[13] = tinterval
                   config[14] = tshots
                   config[15] = extn
                   config[16] = zx
                   config[17] = zy
                   config[18] = zoom
                   config[19] = int(saturation)
                   config[20] = meter
                   config[21] = awb
                   config[22] = sharpness
                   config[23] = int(denoise)
                   config[24] = quality
                   config[25] = profile
                   config[26] = level
                   config[27] = histogram
                   config[28] = histarea
                   config[29] = v3_f_speed
                   config[30] = v3_f_range
                   config[31] = rotate
                   config[32] = IRF
                   config[33] = str_cap
                   config[34] = v3_hdr
                   config[35] = timet
                   config[36] = vflip
                   config[37] = hflip
                   config[38] = stretch_p_low
                   config[39] = stretch_p_high
                   config[40] = stretch_factor
                   config[41] = stretch_preset
                   with open(config_file, 'w') as f:
                      for item in range(0,len(titles)):
                          f.write(titles[item] + " : " + str(config[item]) + "\n")
                   time.sleep(1)
                   text(0,8,2,0,1,"SAVE CONFIG",fv,11)

            # MENU 6
            elif menu == 6:
              if button_row == 1:
                # TIMELAPSE DURATION
                for f in range(0,len(video_limits)-1,3):
                    if video_limits[f] == 'tduration':
                        pmin = video_limits[f+1]
                        pmax = video_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    tduration = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height + (bh*3)  and mousey < preview_height + (bh*3) + int(bh/3)) and alt_dis == 1:
                    tduration = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75 + (bh*3)  and mousey < preview_height * .75 + (bh*3) + int(bh/3)) and alt_dis == 2:
                    tduration = int(((mousex-((button_row - 9)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        tduration -=1
                        tduration = max(tduration,pmin)
                    else:
                        tduration +=1
                        tduration = min(tduration,pmax)
                td = timedelta(seconds=tduration)
                text(0,1,3,1,1,str(td),fv,12)
                draw_Vbar(0,1,lyelColor,'tduration',tduration)
                if tinterval > 0:
                    tshots = int(tduration / tinterval)
                    text(0,3,3,1,1,str(tshots),fv,12)
                else:
                    text(0,3,3,1,1," ",fv,12)
                draw_Vbar(0,3,lyelColor,'tshots',tshots)
                time.sleep(.25)

              elif button_row == 2:
                # TIMELAPSE INTERVAL
                for f in range(0,len(video_limits)-1,3):
                    if video_limits[f] == 'tinterval':
                        pmin = video_limits[f+1]
                        pmax = video_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    tinterval = round(((mousex-preview_width) / bw) * (pmax-pmin) + pmin, 2)
                elif (mousey > preview_height + (bh*3)  and mousey < preview_height + (bh*3) + int(bh/3)) and alt_dis == 1:
                    tinterval = round(((mousex-((button_row - 9)*bw)) / bw) * (pmax-pmin) + pmin, 2)
                elif (mousey > preview_height * .75 + (bh*3)  and mousey < preview_height * .75 + (bh*3) + int(bh/3)) and alt_dis == 2:
                    tinterval = round(((mousex-((button_row - 9)*bw)) / bw) * (pmax-pmin) + pmin, 2)
                else:
                    # Pas intelligent: 0.01s si < 1s, sinon 0.1s
                    step = 0.01 if tinterval < 1 else 0.1
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        tinterval = round(tinterval - step, 2)
                        tinterval = max(tinterval, pmin)
                    else:
                        tinterval = round(tinterval + step, 2)
                        tinterval = min(tinterval, pmax)
                td = timedelta(seconds=tinterval)
                text(0,2,3,1,1,str(td),fv,12)
                draw_Vbar(0,2,lyelColor,'tinterval',tinterval)
                if tinterval != 0:
                    tduration = tinterval * tshots
                    td = timedelta(seconds=tduration)
                    text(0,1,3,1,1,str(td),fv,12)
                    draw_Vbar(0,1,lyelColor,'tduration',tduration)
                if tinterval == 0:
                    text(0,3,3,1,1," ",fv,12)
                    if mode == 0:
                        speed = 15
                        shutter = shutters[speed]
                        if shutter < 0:
                            shutter = abs(1/shutter)
                        sspeed = int(shutter * 1000000)
                        if (shutter * 1000000) - int(shutter * 1000000) > 0.5:
                            sspeed +=1
                        restart = 1
                else:
                    text(0,3,3,1,1,str(tshots),fv,12)
                time.sleep(.25)
                
              elif button_row == 3 and tinterval > 0:
                # TIMELAPSE SHOTS
                for f in range(0,len(video_limits)-1,3):
                    if video_limits[f] == 'tshots':
                        pmin = video_limits[f+1]
                        pmax = video_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    tshots = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height + (bh*3)  and mousey < preview_height + (bh*3) + int(bh/3)) and alt_dis == 1:
                    tshots = int(((mousex-((button_row -9)*bw)) / bw) * (pmax+1-pmin))
                elif (mousey > preview_height * .75 + (bh*3)  and mousey < preview_height * .75 + (bh*3) + int(bh/3)) and alt_dis == 2:
                    tshots = int(((mousex-((button_row -9)*bw)) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        tshots -=1
                        tshots = max(tshots,pmin)
                    else:
                        tshots +=1
                        tshots = min(tshots,pmax)
                text(0,3,3,1,1,str(tshots),fv,12)
                draw_Vbar(0,3,lyelColor,'tshots',tshots)
                if tduration > 0:
                    tduration = tinterval * tshots
                if tduration == 0:
                    tduration = 1
                td = timedelta(seconds=tduration)
                text(0,1,3,1,1,str(td),fv,12)
                draw_Vbar(0,1,lyelColor,'tduration',tduration)
                time.sleep(.25)
              
              elif button_row == 8:
                   # SAVE CONFIG
                   text(0,8,3,0,1,"SAVE Config",fv,12)
                   config[0] = mode
                   config[1] = speed
                   config[2] = gain
                   config[3] = int(brightness)
                   config[4] = int(contrast)
                   config[5] = frame
                   config[6] = int(red)
                   config[7] = int(blue)
                   config[8] = ev
                   config[9] = vlen
                   config[10] = fps
                   config[11] = vformat
                   config[12] = codec
                   config[13] = tinterval
                   config[14] = tshots
                   config[15] = extn
                   config[16] = zx
                   config[17] = zy
                   config[18] = zoom
                   config[19] = int(saturation)
                   config[20] = meter
                   config[21] = awb
                   config[22] = sharpness
                   config[23] = int(denoise)
                   config[24] = quality
                   config[25] = profile
                   config[26] = level
                   config[27] = histogram
                   config[28] = histarea
                   config[29] = v3_f_speed
                   config[30] = v3_f_range
                   config[31] = rotate
                   config[32] = IRF
                   config[33] = str_cap
                   config[34] = v3_hdr
                   config[35] = timet
                   config[36] = vflip
                   config[37] = hflip
                   config[38] = stretch_p_low
                   config[39] = stretch_p_high
                   config[40] = stretch_factor
                   config[41] = stretch_preset
                   with open(config_file, 'w') as f:
                      for item in range(0,len(titles)):
                          f.write(titles[item] + " : " + str(config[item]) + "\n")
                   time.sleep(1)
                   text(0,8,2,0,1,"SAVE CONFIG",fv,12)

            # MENU 7 - STRETCH Settings
            elif menu == 7:
              if button_row == 9:
                  # Retour CAMERA Settings
                  menu = 1
                  Menu()

              elif button_row == 1:
                # STRETCH LOW PERCENTILE
                pmin = 1    # 0.1%
                pmax = 100  # 10%
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    stretch_p_low = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        stretch_p_low -=1
                        stretch_p_low = max(stretch_p_low,pmin)
                    else:
                        stretch_p_low +=1
                        stretch_p_low = min(stretch_p_low,pmax)
                # Note: modification manuelle des paramètres - le preset affiché reste inchangé
                text(0,1,3,1,1,str(stretch_p_low/10)[0:4],fv,7)
                draw_Vbar(0,1,greyColor,'stretch_p_low',stretch_p_low)
                text(0,4,3,1,1,stretch_presets[stretch_preset],fv,7)
                draw_Vbar(0,4,greyColor,'stretch_preset',stretch_preset)
                time.sleep(.05)

              elif button_row == 2:
                # STRETCH HIGH PERCENTILE
                pmin = 900  # 90%
                pmax = 1000 # 100%
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    stretch_p_high = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        stretch_p_high -=1
                        stretch_p_high = max(stretch_p_high,pmin)
                    else:
                        stretch_p_high +=1
                        stretch_p_high = min(stretch_p_high,pmax)
                # Note: modification manuelle des paramètres - le preset affiché reste inchangé
                text(0,2,3,1,1,str(stretch_p_high/10)[0:5],fv,7)
                draw_Vbar(0,2,greyColor,'stretch_p_high',stretch_p_high)
                text(0,4,3,1,1,stretch_presets[stretch_preset],fv,7)
                draw_Vbar(0,4,greyColor,'stretch_preset',stretch_preset)
                time.sleep(.05)

              elif button_row == 3:
                # STRETCH FACTOR
                pmin = 10   # 1.0
                pmax = 300  # 30.0
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    stretch_factor = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        stretch_factor -=5
                        stretch_factor = max(stretch_factor,pmin)
                    else:
                        stretch_factor +=5
                        stretch_factor = min(stretch_factor,pmax)
                # Note: modification manuelle des paramètres - le preset affiché reste inchangé
                text(0,3,3,1,1,str(stretch_factor/10)[0:4],fv,7)
                draw_Vbar(0,3,greyColor,'stretch_factor',stretch_factor)
                text(0,4,3,1,1,stretch_presets[stretch_preset],fv,7)
                draw_Vbar(0,4,greyColor,'stretch_preset',stretch_preset)
                time.sleep(.05)

              elif button_row == 4:
                # STRETCH PRESET
                pmin = 0
                pmax = 3
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    stretch_preset = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        stretch_preset -=1
                        stretch_preset = max(stretch_preset,pmin)
                    else:
                        stretch_preset +=1
                        stretch_preset = min(stretch_preset,pmax)

                # Charger les valeurs du preset
                if stretch_preset == 0:
                    # OFF - Pas de stretch, juste affichage full screen sans traitement
                    # stretch_mode reste à 1 pour garder le fullscreen, mais stretch_preset=0
                    # empêche l'application du traitement stretch dans le code de capture
                    pass
                elif stretch_preset == 1:
                    # AUTO - Détection automatique basée sur l'histogramme
                    # Analyse l'image pour adapter automatiquement les paramètres
                    stretch_p_low = 10
                    stretch_p_high = 995
                    stretch_factor = 100
                elif stretch_preset == 2:
                    # Nébuleuse - Moyennement agressif pour nébuleuses et amas
                    stretch_p_low = 10
                    stretch_p_high = 995
                    stretch_factor = 120
                elif stretch_preset == 3:
                    # Galaxie - Très agressif pour révéler les détails faibles
                    stretch_p_low = 5
                    stretch_p_high = 995
                    stretch_factor = 150

                # Mettre à jour l'affichage
                text(0,4,3,1,1,stretch_presets[stretch_preset],fv,7)
                draw_Vbar(0,4,greyColor,'stretch_preset',stretch_preset)
                if stretch_preset != 0:
                    # Mettre à jour l'affichage des paramètres si un preset a été sélectionné
                    text(0,1,3,1,1,str(stretch_p_low/10)[0:4],fv,7)
                    draw_Vbar(0,1,greyColor,'stretch_p_low',stretch_p_low)
                    text(0,2,3,1,1,str(stretch_p_high/10)[0:5],fv,7)
                    draw_Vbar(0,2,greyColor,'stretch_p_high',stretch_p_high)
                    text(0,3,3,1,1,str(stretch_factor/10)[0:4],fv,7)
                    draw_Vbar(0,3,greyColor,'stretch_factor',stretch_factor)
                time.sleep(.05)

              elif button_row == 8:
                   # SAVE CONFIG
                   text(0,8,3,0,1,"SAVE Config",fv,7)
                   config[0] = mode
                   config[1] = speed
                   config[2] = gain
                   config[3] = int(brightness)
                   config[4] = int(contrast)
                   config[5] = frame
                   config[6] = int(red)
                   config[7] = int(blue)
                   config[8] = ev
                   config[9] = vlen
                   config[10] = fps
                   config[11] = vformat
                   config[12] = codec
                   config[13] = tinterval
                   config[14] = tshots
                   config[15] = extn
                   config[16] = zx
                   config[17] = zy
                   config[18] = zoom
                   config[19] = int(saturation)
                   config[20] = meter
                   config[21] = awb
                   config[22] = sharpness
                   config[23] = int(denoise)
                   config[24] = quality
                   config[25] = profile
                   config[26] = level
                   config[27] = histogram
                   config[28] = histarea
                   config[29] = v3_f_speed
                   config[30] = v3_f_range
                   config[31] = rotate
                   config[32] = IRF
                   config[33] = str_cap
                   config[34] = v3_hdr
                   config[35] = timet
                   config[36] = vflip
                   config[37] = hflip
                   config[38] = stretch_p_low
                   config[39] = stretch_p_high
                   config[40] = stretch_factor
                   config[41] = stretch_preset
                   with open(config_file, 'w') as f:
                      for item in range(0,len(titles)):
                          f.write(titles[item] + " : " + str(config[item]) + "\n")
                   time.sleep(1)
                   text(0,8,2,0,1,"SAVE CONFIG",fv,7)

            elif menu == 8:
              # LIVE STACK Settings - Gestion des clics
              if button_row == 1:
                # PREVIEW REFRESH (3-10)
                for f in range(0,len(livestack_limits)-1,3):
                    if livestack_limits[f] == 'ls_preview_refresh':
                        pmin = livestack_limits[f+1]
                        pmax = livestack_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    ls_preview_refresh = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        ls_preview_refresh -=1
                        ls_preview_refresh = max(ls_preview_refresh,pmin)
                    else:
                        ls_preview_refresh +=1
                        ls_preview_refresh = min(ls_preview_refresh,pmax)
                text(0,1,3,1,1,str(ls_preview_refresh),fv,7)
                draw_Vbar(0,1,greyColor,'ls_preview_refresh',ls_preview_refresh)
                time.sleep(.05)

              elif button_row == 2:
                # ALIGNMENT MODE (0-2: translation/rotation/affine)
                for f in range(0,len(livestack_limits)-1,3):
                    if livestack_limits[f] == 'ls_alignment_mode':
                        pmin = livestack_limits[f+1]
                        pmax = livestack_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    ls_alignment_mode = int(((mousex-preview_width) / bw) * (pmax+1-pmin))
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        ls_alignment_mode -=1
                        ls_alignment_mode = max(ls_alignment_mode,pmin)
                    else:
                        ls_alignment_mode +=1
                        ls_alignment_mode = min(ls_alignment_mode,pmax)
                text(0,2,3,1,1,ls_alignment_modes[ls_alignment_mode],fv,7)
                draw_Vbar(0,2,greyColor,'ls_alignment_mode',ls_alignment_mode)
                time.sleep(.05)

              elif button_row == 3:
                # MAX FWHM (0=OFF, 100-250 affiché 10.0-25.0)
                for f in range(0,len(livestack_limits)-1,3):
                    if livestack_limits[f] == 'ls_max_fwhm':
                        pmin = livestack_limits[f+1]
                        pmax = livestack_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    ls_max_fwhm = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        ls_max_fwhm -=10
                        ls_max_fwhm = max(ls_max_fwhm,pmin)
                    else:
                        ls_max_fwhm +=10
                        ls_max_fwhm = min(ls_max_fwhm,pmax)
                if ls_max_fwhm == 0:
                    text(0,3,3,1,1,"OFF",fv,7)
                else:
                    text(0,3,3,1,1,str(ls_max_fwhm/10)[0:4],fv,7)
                draw_Vbar(0,3,greyColor,'ls_max_fwhm',ls_max_fwhm)
                time.sleep(.05)

              elif button_row == 4:
                # MIN SHARPNESS (0=OFF, 30-150 affiché 0.030-0.150)
                for f in range(0,len(livestack_limits)-1,3):
                    if livestack_limits[f] == 'ls_min_sharpness':
                        pmin = livestack_limits[f+1]
                        pmax = livestack_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    ls_min_sharpness = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        ls_min_sharpness -=5
                        ls_min_sharpness = max(ls_min_sharpness,pmin)
                    else:
                        ls_min_sharpness +=5
                        ls_min_sharpness = min(ls_min_sharpness,pmax)
                if ls_min_sharpness == 0:
                    text(0,4,3,1,1,"OFF",fv,7)
                else:
                    text(0,4,3,1,1,str(ls_min_sharpness/1000)[0:5],fv,7)
                draw_Vbar(0,4,greyColor,'ls_min_sharpness',ls_min_sharpness)
                time.sleep(.05)

              elif button_row == 5:
                # MAX DRIFT (0=OFF, 500-5000 pixels)
                for f in range(0,len(livestack_limits)-1,3):
                    if livestack_limits[f] == 'ls_max_drift':
                        pmin = livestack_limits[f+1]
                        pmax = livestack_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    ls_max_drift = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        ls_max_drift -=100
                        ls_max_drift = max(ls_max_drift,pmin)
                    else:
                        ls_max_drift +=100
                        ls_max_drift = min(ls_max_drift,pmax)
                if ls_max_drift == 0:
                    text(0,5,3,1,1,"OFF",fv,7)
                else:
                    text(0,5,3,1,1,str(ls_max_drift),fv,7)
                draw_Vbar(0,5,greyColor,'ls_max_drift',ls_max_drift)
                time.sleep(.05)

              elif button_row == 6:
                # MIN STARS (0=OFF, 1-20 étoiles)
                for f in range(0,len(livestack_limits)-1,3):
                    if livestack_limits[f] == 'ls_min_stars':
                        pmin = livestack_limits[f+1]
                        pmax = livestack_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    ls_min_stars = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                else:
                    if (alt_dis == 0 and mousex < preview_width + (bw/2)) or (alt_dis > 0 and button_pos == 0):
                        ls_min_stars -=1
                        ls_min_stars = max(ls_min_stars,pmin)
                    else:
                        ls_min_stars +=1
                        ls_min_stars = min(ls_min_stars,pmax)
                if ls_min_stars == 0:
                    text(0,6,3,1,1,"OFF",fv,7)
                else:
                    text(0,6,3,1,1,str(ls_min_stars),fv,7)
                draw_Vbar(0,6,greyColor,'ls_min_stars',ls_min_stars)
                time.sleep(.05)

              elif button_row == 7:
                # QUALITY CONTROL (0=OFF, 1=ON)
                for f in range(0,len(livestack_limits)-1,3):
                    if livestack_limits[f] == 'ls_enable_qc':
                        pmin = livestack_limits[f+1]
                        pmax = livestack_limits[f+2]
                if (mousex > preview_width and mousey < ((button_row)*bh) + int(bh/3)):
                    ls_enable_qc = int(((mousex-preview_width) / bw) * (pmax+1-pmin) + pmin)
                else:
                    # Toggle entre 0 et 1
                    ls_enable_qc = 1 - ls_enable_qc
                if ls_enable_qc == 0:
                    text(0,7,3,1,1,"OFF",fv,7)
                else:
                    text(0,7,3,1,1,"ON",fv,7)
                draw_Vbar(0,7,greyColor,'ls_enable_qc',ls_enable_qc)
                time.sleep(.05)

              elif button_row == 8:
                   # SAVE CONFIG
                   text(0,8,3,0,1,"SAVE Config",fv,7)
                   config[0] = mode
                   config[1] = speed
                   config[2] = gain
                   config[3] = int(brightness)
                   config[4] = int(contrast)
                   config[5] = frame
                   config[6] = int(red)
                   config[7] = int(blue)
                   config[8] = ev
                   config[9] = vlen
                   config[10] = fps
                   config[11] = vformat
                   config[12] = codec
                   config[13] = tinterval
                   config[14] = tshots
                   config[15] = extn
                   config[16] = zx
                   config[17] = zy
                   config[18] = zoom
                   config[19] = saturation
                   config[20] = meter
                   config[21] = awb
                   config[22] = sharpness
                   config[23] = denoise
                   config[24] = quality
                   config[25] = profile
                   config[26] = level
                   config[27] = histogram
                   config[28] = histarea
                   config[29] = v3_f_speed
                   config[30] = v3_f_range
                   config[31] = rotate
                   config[32] = IRF
                   config[33] = str_cap
                   config[34] = v3_hdr
                   config[35] = timet
                   config[36] = vflip
                   config[37] = hflip
                   config[38] = stretch_p_low
                   config[39] = stretch_p_high
                   config[40] = stretch_factor
                   config[41] = stretch_preset
                   config[42] = ls_preview_refresh
                   config[43] = ls_alignment_mode
                   config[44] = ls_enable_qc
                   config[45] = ls_max_fwhm
                   config[46] = ls_min_sharpness
                   config[47] = ls_max_drift
                   config[48] = ls_min_stars
                   with open(config_file, 'w') as f:
                       for item in range(0,len(titles)):
                           f.write(titles[item] + " : " + str(config[item]) + "\n")
                   time.sleep(1)
                   text(0,8,2,0,1,"SAVE CONFIG",fv,7)

              elif button_row == 9:
                  # Retour CAMERA Settings
                  menu = 1
                  Menu()

        # RESTART
        if restart > 0:
            kill_preview_process()
            text(0,0,6,2,1,"Waiting for preview ...",int(fv*1.7),1)
            time.sleep(1)
            preview()
