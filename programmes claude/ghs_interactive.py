#!/usr/bin/env python3
"""
Interface interactive pour ajuster les paramètres GHS en temps réel
"""

import sys
import numpy as np

# Forcer l'utilisation d'un backend GUI
import matplotlib
matplotlib.use('TkAgg')  # ou 'Qt5Agg' si TkAgg n'est pas disponible

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from pathlib import Path

from libastrostack.io import load_image


def ghs_stretch_universal(data, D, b, SP, LP=0.0, HP=1.0):
    """GHS 5 paramètres - Version optimisée pour temps réel"""
    epsilon = 1e-10

    # Normaliser en float (0-1)
    if data.dtype == np.uint8:
        img_float = data.astype(np.float64) / 255.0
    elif data.dtype == np.uint16:
        data_max = data.max()
        if data_max <= 4095:
            img_float = data.astype(np.float64) / 4095.0
        else:
            img_float = data.astype(np.float64) / 65535.0
    elif data.dtype in [np.float32, np.float64]:
        img_float = data.astype(np.float64)
        if img_float.max() > 1.0:
            img_float = img_float / img_float.max()
    else:
        img_float = data.astype(np.float64)

    img_float = np.clip(img_float, epsilon, 1.0 - epsilon)

    if abs(D) < epsilon:
        return img_float

    # Contraintes
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

    # Zone 1: x < LP
    if LP > epsilon:
        mask_low = img_float < LP
        if np.any(mask_low):
            slope_SP = T_prime(SP, D, b) / T_1
            slope_LP = slope_SP * (LP / SP)
            img_stretched[mask_low] = slope_LP * img_float[mask_low]

    # Zone 2: LP <= x < SP
    mask_mid_low = (img_float >= LP) & (img_float < SP)
    if np.any(mask_mid_low):
        x_norm = img_float[mask_mid_low] / SP
        T_x = T_base(x_norm, D, b) / T_1
        img_stretched[mask_mid_low] = SP * T_x

    # Zone 3: x >= SP
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


class GHSInteractive:
    """Interface interactive pour GHS"""

    def __init__(self, image_path):
        # Charger l'image
        print(f"Chargement: {image_path}")
        self.data_original = load_image(image_path)

        if self.data_original is None:
            raise ValueError(f"Impossible de charger {image_path}")

        print(f"Image: shape={self.data_original.shape}, dtype={self.data_original.dtype}")

        # Convertir en mono si couleur
        if len(self.data_original.shape) == 3:
            self.data = np.mean(self.data_original, axis=2)
        else:
            self.data = self.data_original

        # Normaliser
        if self.data.max() > 1.0:
            self.data = self.data / self.data.max()

        print(f"Données prêtes: shape={self.data.shape}, range=[{self.data.min():.6f}, {self.data.max():.6f}]")

        # Créer l'interface
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.canvas.manager.set_window_title('GHS Interactive - Ajustement en temps réel')

        # Zone d'affichage de l'image
        self.ax_img = plt.subplot2grid((4, 3), (0, 0), colspan=2, rowspan=3)

        # Histogramme
        self.ax_hist = plt.subplot2grid((4, 3), (3, 0), colspan=2)

        # Zone des sliders
        self.ax_sliders = plt.subplot2grid((4, 3), (0, 2), rowspan=4)
        self.ax_sliders.axis('off')

        # Paramètres initiaux (preset Galaxies)
        self.params = {
            'D': 2.0,
            'b': 6.0,
            'SP': 0.15,
            'LP': 0.0,
            'HP': 0.85
        }

        # Créer les sliders
        self.sliders = {}
        slider_configs = [
            ('D', 0.0, 5.0, 'Stretch factor\n(force étirement)'),
            ('b', -5.0, 15.0, 'Local intensity\n(concentration contraste)'),
            ('SP', 0.0, 1.0, 'Symmetry point\n(point focal contraste)'),
            ('LP', 0.0, 1.0, 'Protect shadows\n(protection basses lumières)'),
            ('HP', 0.0, 1.0, 'Protect highlights\n(protection hautes lumières)')
        ]

        y_positions = [0.82, 0.68, 0.54, 0.40, 0.26]

        for idx, (name, vmin, vmax, label) in enumerate(slider_configs):
            ax_slider = plt.axes([0.72, y_positions[idx], 0.22, 0.03])
            slider = Slider(
                ax_slider,
                f'{name}\n{label}',
                vmin,
                vmax,
                valinit=self.params[name],
                valstep=(vmax - vmin) / 1000
            )
            slider.on_changed(lambda val, n=name: self.update_param(n, val))
            self.sliders[name] = slider

        # Boutons de presets
        ax_btn_galaxies = plt.axes([0.72, 0.16, 0.22, 0.04])
        self.btn_galaxies = Button(ax_btn_galaxies, 'Preset: Galaxies')
        self.btn_galaxies.on_clicked(self.preset_galaxies)

        ax_btn_nebulae = plt.axes([0.72, 0.11, 0.22, 0.04])
        self.btn_nebulae = Button(ax_btn_nebulae, 'Preset: Nébuleuses')
        self.btn_nebulae.on_clicked(self.preset_nebulae)

        ax_btn_initial = plt.axes([0.72, 0.06, 0.22, 0.04])
        self.btn_initial = Button(ax_btn_initial, 'Preset: Étirement initial')
        self.btn_initial.on_clicked(self.preset_initial)

        ax_btn_reset = plt.axes([0.72, 0.01, 0.22, 0.04])
        self.btn_reset = Button(ax_btn_reset, 'Reset (D=0)')
        self.btn_reset.on_clicked(self.preset_reset)

        # Affichage initial
        self.img_display = None
        self.hist_bars = None
        self.update_display()

        plt.tight_layout()

    def update_param(self, name, value):
        """Mise à jour d'un paramètre"""
        self.params[name] = value

        # Contraintes automatiques
        if name == 'SP':
            # LP ne peut pas dépasser SP
            if self.params['LP'] > value:
                self.params['LP'] = value
                self.sliders['LP'].set_val(value)
            # HP ne peut pas être inférieur à SP
            if self.params['HP'] < value:
                self.params['HP'] = value
                self.sliders['HP'].set_val(value)

        self.update_display()

    def update_display(self):
        """Mise à jour de l'affichage"""
        # Appliquer GHS
        result = ghs_stretch_universal(
            self.data,
            self.params['D'],
            self.params['b'],
            self.params['SP'],
            self.params['LP'],
            self.params['HP']
        )

        # Afficher l'image
        if self.img_display is None:
            self.img_display = self.ax_img.imshow(result, cmap='gray', vmin=0, vmax=1)
            self.ax_img.set_title('Image GHS', fontsize=14, fontweight='bold')
            self.ax_img.axis('off')
        else:
            self.img_display.set_data(result)

        # Mettre à jour le titre avec les stats
        mean_val = result.mean()
        std_val = result.std()
        pct_bright = (result > 0.5).sum() / result.size * 100
        self.ax_img.set_title(
            f'GHS - Mean: {mean_val:.4f}, Std: {std_val:.4f}, >0.5: {pct_bright:.2f}%',
            fontsize=12, fontweight='bold'
        )

        # Afficher l'histogramme
        self.ax_hist.clear()
        self.ax_hist.hist(result.flatten(), bins=100, alpha=0.7, color='blue')
        self.ax_hist.set_xlabel('Valeur pixel')
        self.ax_hist.set_ylabel('Fréquence')
        self.ax_hist.set_title('Histogramme', fontsize=10)
        self.ax_hist.set_xlim(0, 1)
        self.ax_hist.grid(True, alpha=0.3)

        self.fig.canvas.draw_idle()

    def preset_galaxies(self, event):
        """Preset Galaxies"""
        self.set_params(D=2.0, b=6.0, SP=0.15, LP=0.0, HP=0.85)

    def preset_nebulae(self, event):
        """Preset Nébuleuses"""
        self.set_params(D=2.5, b=4.0, SP=0.10, LP=0.02, HP=0.95)

    def preset_initial(self, event):
        """Preset Étirement initial"""
        self.set_params(D=3.5, b=12.0, SP=0.08, LP=0.0, HP=1.0)

    def preset_reset(self, event):
        """Reset (pas de stretch)"""
        self.set_params(D=0.0, b=6.0, SP=0.15, LP=0.0, HP=0.85)

    def set_params(self, **kwargs):
        """Définir plusieurs paramètres"""
        for name, value in kwargs.items():
            if name in self.params:
                self.params[name] = value
                self.sliders[name].set_val(value)

    def show(self):
        """Afficher l'interface"""
        print("\n" + "="*80)
        print("INTERFACE INTERACTIVE GHS")
        print("="*80)
        print("\nUtilisez les sliders pour ajuster les paramètres en temps réel:")
        print("  D  : Stretch factor (0-5) - force de l'étirement")
        print("  b  : Local intensity (-5 à 15) - concentration du contraste")
        print("  SP : Symmetry point (0-1) - point focal du contraste")
        print("  LP : Protect shadows (0-SP) - protection des basses lumières")
        print("  HP : Protect highlights (SP-1) - protection des hautes lumières")
        print("\nPresets disponibles:")
        print("  - Galaxies: D=2.0, b=6.0, SP=0.15, LP=0.0, HP=0.85")
        print("  - Nébuleuses: D=2.5, b=4.0, SP=0.10, LP=0.02, HP=0.95")
        print("  - Étirement initial: D=3.5, b=12.0, SP=0.08, LP=0.0, HP=1.0")
        print("\nFermez la fenêtre pour terminer.")
        print("="*80 + "\n")

        plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "/media/admin/THKAILAR/M81/250917224338_0.dng"

    try:
        app = GHSInteractive(image_path)
        app.show()
    except Exception as e:
        print(f"Erreur: {e}")
        import traceback
        traceback.print_exc()
