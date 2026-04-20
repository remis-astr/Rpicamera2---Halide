#!/usr/bin/env python3
"""
Interface interactive GHS avec Tkinter
Sliders en temps réel pour explorer TOUTES les combinaisons de paramètres
"""

import sys
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

from libastrostack.io import load_image


def ghs_stretch_universal(data, D, b, SP, LP=0.0, HP=1.0):
    """GHS 5 paramètres - Version optimisée"""
    epsilon = 1e-10

    # Normaliser en float (0-1)
    img_float = data.astype(np.float64)
    if img_float.max() > 1.0:
        img_float = img_float / img_float.max()

    img_float = np.clip(img_float, epsilon, 1.0 - epsilon)

    if abs(D) < epsilon:
        return img_float

    LP = max(0.0, min(LP, SP))
    HP = max(SP, min(HP, 1.0))

    def T_base(x, D, b):
        x = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(x)

        if abs(b - (-1.0)) < epsilon:
            result = np.log1p(D * x)
        elif b < 0 and abs(b - (-1.0)) >= epsilon:
            base = np.maximum(1.0 - b * D * x, epsilon)
            exponent = (b + 1.0) / b
            result = (1.0 - np.power(base, exponent)) / (D * (b + 1.0))
        elif abs(b) < epsilon:
            result = 1.0 - np.exp(-D * x)
        elif abs(b - 1.0) < epsilon:
            result = 1.0 - 1.0 / (1.0 + D * x)
        else:
            base = np.maximum(1.0 + b * D * x, epsilon)
            result = 1.0 - np.power(base, -1.0 / b)

        return result

    def T_prime(x, D, b):
        x = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(x)

        if abs(b - (-1.0)) < epsilon:
            result = D / (1.0 + D * x)
        elif b < 0 and abs(b - (-1.0)) >= epsilon:
            base = np.maximum(1.0 - b * D * x, epsilon)
            result = np.power(base, 1.0 / b)
        elif abs(b) < epsilon:
            result = D * np.exp(-D * x)
        elif abs(b - 1.0) < epsilon:
            result = D / np.power(1.0 + D * x, 2)
        else:
            base = np.maximum(1.0 + b * D * x, epsilon)
            result = D * np.power(base, -(1.0 / b + 1.0))

        return result

    T_1 = T_base(1.0, D, b)
    img_stretched = np.zeros_like(img_float)

    if LP > epsilon:
        mask_low = img_float < LP
        if np.any(mask_low):
            slope_SP = T_prime(SP, D, b) / T_1
            slope_LP = slope_SP * (LP / SP)
            img_stretched[mask_low] = slope_LP * img_float[mask_low]

    mask_mid_low = (img_float >= LP) & (img_float < SP)
    if np.any(mask_mid_low):
        x_norm = img_float[mask_mid_low] / SP
        T_x = T_base(x_norm, D, b) / T_1
        img_stretched[mask_mid_low] = SP * T_x

    mask_high = img_float >= SP
    if np.any(mask_high):
        if HP < 1.0 - epsilon:
            mask_mid_high = (img_float >= SP) & (img_float < HP)
            if np.any(mask_mid_high):
                x_mirror = 1.0 - img_float[mask_mid_high]
                x_norm_mirror = x_mirror / (1.0 - SP)
                T_mirror = T_base(x_norm_mirror, D, b) / T_1
                img_stretched[mask_mid_high] = 1.0 - (1.0 - SP) * T_mirror

            mask_very_high = img_float >= HP
            if np.any(mask_very_high):
                slope_SP_high = T_prime(SP, D, b) / T_1
                slope_HP = slope_SP_high * ((1.0 - HP) / (1.0 - SP))

                T_HP_val = T_base((1.0 - HP) / (1.0 - SP), D, b) / T_1
                y_HP = 1.0 - (1.0 - SP) * T_HP_val

                img_stretched[mask_very_high] = y_HP + slope_HP * (img_float[mask_very_high] - HP)
        else:
            x_mirror = 1.0 - img_float[mask_high]
            x_norm_mirror = x_mirror / (1.0 - SP)
            T_mirror = T_base(x_norm_mirror, D, b) / T_1
            img_stretched[mask_high] = 1.0 - (1.0 - SP) * T_mirror

    return np.clip(img_stretched, 0.0, 1.0)


class GHSApp:
    def __init__(self, root, image_path):
        self.root = root
        self.root.title("GHS Interactive - Explorateur de paramètres")

        # Charger l'image
        print(f"Chargement: {image_path}")
        data = load_image(image_path)

        if data is None:
            raise ValueError(f"Impossible de charger {image_path}")

        # Convertir en mono et normaliser
        if len(data.shape) == 3:
            data = np.mean(data, axis=2)

        if data.max() > 1.0:
            data = data / data.max()

        self.data = data
        print(f"Image prête: shape={data.shape}, range=[{data.min():.6f}, {data.max():.6f}]")

        # Redimensionner pour affichage (max 800x600)
        h, w = data.shape
        max_dim = 800
        if h > max_dim or w > max_dim:
            scale = min(max_dim / h, max_dim / w)
            new_h, new_w = int(h * scale), int(w * scale)
            from scipy.ndimage import zoom
            self.display_data = zoom(data, (new_h / h, new_w / w), order=1)
        else:
            self.display_data = data

        print(f"Taille affichage: {self.display_data.shape}")

        # Paramètres initiaux
        self.params = {
            'D': tk.DoubleVar(value=2.0),
            'b': tk.DoubleVar(value=6.0),
            'SP': tk.DoubleVar(value=0.15),
            'LP': tk.DoubleVar(value=0.0),
            'HP': tk.DoubleVar(value=0.85),
            'use_percentiles': tk.BooleanVar(value=False),
            'clip_low': tk.DoubleVar(value=1.0),
            'clip_high': tk.DoubleVar(value=99.5)
        }

        # Layout principal
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Cadre image (gauche)
        img_frame = ttk.LabelFrame(main_frame, text="Image GHS", padding="5")
        img_frame.grid(row=0, column=0, rowspan=6, padx=5, pady=5, sticky=(tk.N, tk.S))

        self.img_label = ttk.Label(img_frame)
        self.img_label.grid(row=0, column=0)

        self.stats_label = ttk.Label(img_frame, text="", font=('Arial', 10))
        self.stats_label.grid(row=1, column=0, pady=5)

        # Cadre sliders (droite)
        sliders_frame = ttk.LabelFrame(main_frame, text="Paramètres GHS", padding="10")
        sliders_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.N))

        # Créer les sliders GHS
        self.create_slider(sliders_frame, 'D', 0.0, 10.0, 0,
                          "D - Stretch factor (force étirement)")
        self.create_slider(sliders_frame, 'b', -5.0, 20.0, 1,
                          "b - Local intensity (concentration contraste)")
        self.create_slider(sliders_frame, 'SP', 0.0, 1.0, 2,
                          "SP - Symmetry point (point focal)")
        self.create_slider(sliders_frame, 'LP', 0.0, 1.0, 3,
                          "LP - Protect shadows")
        self.create_slider(sliders_frame, 'HP', 0.0, 1.0, 4,
                          "HP - Protect highlights")

        # Cadre normalisation percentiles
        percentiles_frame = ttk.LabelFrame(main_frame, text="Normalisation Percentiles", padding="10")
        percentiles_frame.grid(row=2, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N))

        # Checkbox pour activer/désactiver
        self.percentiles_check = ttk.Checkbutton(
            percentiles_frame,
            text="Activer normalisation par percentiles",
            variable=self.params['use_percentiles'],
            command=self.update_image
        )
        self.percentiles_check.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))

        # Sliders percentiles
        self.create_slider(percentiles_frame, 'clip_low', 0.01, 5.0, 1,
                          "Percentile bas (%) - coupe shadows", parent_offset=0)
        self.create_slider(percentiles_frame, 'clip_high', 95.0, 99.99, 2,
                          "Percentile haut (%) - coupe highlights", parent_offset=0)

        # Cadre presets
        presets_frame = ttk.LabelFrame(main_frame, text="Presets", padding="10")
        presets_frame.grid(row=1, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))

        ttk.Button(presets_frame, text="Galaxies (D=2, b=6, SP=0.15)",
                  command=self.preset_galaxies).grid(row=0, column=0, pady=2, sticky=(tk.W, tk.E))
        ttk.Button(presets_frame, text="Nébuleuses (D=2.5, b=4, SP=0.1)",
                  command=self.preset_nebulae).grid(row=1, column=0, pady=2, sticky=(tk.W, tk.E))
        ttk.Button(presets_frame, text="Initial (D=3.5, b=12, SP=0.08)",
                  command=self.preset_initial).grid(row=2, column=0, pady=2, sticky=(tk.W, tk.E))
        ttk.Button(presets_frame, text="Personnalisé fort (D=5, b=15, SP=0.5)",
                  command=self.preset_custom).grid(row=3, column=0, pady=2, sticky=(tk.W, tk.E))
        ttk.Button(presets_frame, text="+ Percentiles ARCSINH (1%, 99.5%)",
                  command=self.preset_arcsinh_percentiles).grid(row=4, column=0, pady=2, sticky=(tk.W, tk.E))
        ttk.Button(presets_frame, text="Reset (D=0)",
                  command=self.preset_reset).grid(row=5, column=0, pady=2, sticky=(tk.W, tk.E))

        # Affichage initial
        self.update_image()

    def create_slider(self, parent, name, min_val, max_val, row, label, parent_offset=0):
        """Créer un slider"""
        ttk.Label(parent, text=label).grid(row=row*2+parent_offset, column=0, sticky=tk.W, pady=(10 if row > 0 else 0, 2))

        frame = ttk.Frame(parent)
        frame.grid(row=row*2+1+parent_offset, column=0, sticky=(tk.W, tk.E))

        scale = ttk.Scale(frame, from_=min_val, to=max_val,
                         variable=self.params[name],
                         orient=tk.HORIZONTAL, length=300,
                         command=lambda v: self.update_image())
        scale.grid(row=0, column=0, sticky=(tk.W, tk.E))

        value_label = ttk.Label(frame, text=f"{self.params[name].get():.3f}", width=8)
        value_label.grid(row=0, column=1, padx=5)

        # Mettre à jour le label quand la valeur change
        def update_label(*args):
            value_label.config(text=f"{self.params[name].get():.3f}")

        self.params[name].trace_add('write', update_label)

    def update_image(self):
        """Mettre à jour l'affichage"""
        # Préparer les données
        data_to_process = self.display_data.copy()

        # Appliquer normalisation par percentiles si activée
        if self.params['use_percentiles'].get():
            clip_low = self.params['clip_low'].get()
            clip_high = self.params['clip_high'].get()

            vmin = np.percentile(data_to_process, clip_low)
            vmax = np.percentile(data_to_process, clip_high)

            if vmax > vmin:
                data_to_process = np.clip((data_to_process - vmin) / (vmax - vmin), 0, 1)

        # Appliquer GHS
        result = ghs_stretch_universal(
            data_to_process,
            self.params['D'].get(),
            self.params['b'].get(),
            self.params['SP'].get(),
            self.params['LP'].get(),
            self.params['HP'].get()
        )

        # Convertir en image PIL
        img_array = (result * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_array, mode='L')

        # Convertir en PhotoImage
        self.photo = ImageTk.PhotoImage(img_pil)
        self.img_label.config(image=self.photo)

        # Mettre à jour les stats
        mean_val = result.mean()
        std_val = result.std()
        pct_bright = (result > 0.5).sum() / result.size * 100

        # Indiquer si percentiles sont actifs
        percentiles_status = ""
        if self.params['use_percentiles'].get():
            percentiles_status = f" | Percentiles: ({self.params['clip_low'].get():.2f}%, {self.params['clip_high'].get():.2f}%)"

        stats_text = f"Mean: {mean_val:.4f} | Std: {std_val:.4f} | >0.5: {pct_bright:.2f}%{percentiles_status}"
        self.stats_label.config(text=stats_text)

    def preset_galaxies(self):
        self.params['D'].set(2.0)
        self.params['b'].set(6.0)
        self.params['SP'].set(0.15)
        self.params['LP'].set(0.0)
        self.params['HP'].set(0.85)

    def preset_nebulae(self):
        self.params['D'].set(2.5)
        self.params['b'].set(4.0)
        self.params['SP'].set(0.10)
        self.params['LP'].set(0.02)
        self.params['HP'].set(0.95)

    def preset_initial(self):
        self.params['D'].set(3.5)
        self.params['b'].set(12.0)
        self.params['SP'].set(0.08)
        self.params['LP'].set(0.0)
        self.params['HP'].set(1.0)

    def preset_custom(self):
        """Preset personnalisé avec valeurs plus élevées"""
        self.params['D'].set(5.0)
        self.params['b'].set(15.0)
        self.params['SP'].set(0.50)
        self.params['LP'].set(0.0)
        self.params['HP'].set(0.90)

    def preset_arcsinh_percentiles(self):
        """Activer les percentiles comme ARCSINH"""
        self.params['use_percentiles'].set(True)
        self.params['clip_low'].set(1.0)
        self.params['clip_high'].set(99.5)

    def preset_reset(self):
        self.params['D'].set(0.0)
        self.params['b'].set(6.0)
        self.params['SP'].set(0.15)
        self.params['LP'].set(0.0)
        self.params['HP'].set(0.85)
        self.params['use_percentiles'].set(False)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "/media/admin/THKAILAR/M81/250917224338_0.dng"

    print("="*80)
    print("INTERFACE INTERACTIVE GHS - Tkinter")
    print("="*80)
    print("Explorez TOUTES les combinaisons de paramètres avec les sliders!")
    print("="*80 + "\n")

    try:
        root = tk.Tk()
        app = GHSApp(root, image_path)
        root.mainloop()
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()
