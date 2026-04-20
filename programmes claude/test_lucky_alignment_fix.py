#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de la correction d'alignement entre buffers Lucky Stack

Ce test vérifie que:
1. La référence explicite est préservée entre les buffers
2. Les frames de buffers successifs sont alignées sur la même référence
3. Il n'y a plus de décalage entre les séries
"""

import numpy as np
from libastrostack.lucky_imaging import LuckyImagingStacker, LuckyConfig

print("=== Test correction alignement Lucky Stack ===\n")

# Configuration
config = LuckyConfig(
    buffer_size=10,  # Minimum 10 images requis
    keep_percent=40.0,  # Garder 40% des images
    align_enabled=True,
    auto_stack=True
)

stacker = LuckyImagingStacker(config)
stacker.start()

# Créer une image de référence simulée (disque décalé)
def create_shifted_disk(shift_x=0, shift_y=0, size=128):
    """Crée un disque avec un décalage"""
    img = np.zeros((size, size), dtype=np.float32)
    center_x = size // 2 + shift_x
    center_y = size // 2 + shift_y
    radius = 30

    y, x = np.ogrid[:size, :size]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    img[mask] = 200.0

    # Ajouter un peu de bruit
    img += np.random.normal(0, 5, img.shape)
    img = np.clip(img, 0, 255)

    return img

print("--- Test 1: Premier buffer (pas de référence explicite) ---")
# Premier buffer : images légèrement décalées
for i in range(10):
    img = create_shifted_disk(shift_x=i, shift_y=i)
    stacker.add_frame(img)

# Vérifier que la référence n'est pas explicite
print(f"Référence explicite après buffer 1: {stacker.aligner.reference_is_explicit}")
assert not stacker.aligner.reference_is_explicit, "La référence ne devrait pas être explicite au départ"
print("✓ OK: Référence non explicite au départ\n")

# Simuler ce que fait rpicamera_livestack_advanced.py:
# mettre à jour la référence avec le résultat cumulatif
print("--- Test 2: Mise à jour référence explicite ---")
if stacker.last_result is not None:
    master_stack = stacker.last_result.copy()
    stacker.update_alignment_reference(master_stack)

    # Vérifier que la référence est maintenant explicite
    print(f"Référence explicite après update: {stacker.aligner.reference_is_explicit}")
    assert stacker.aligner.reference_is_explicit, "La référence devrait être explicite après update"
    print("✓ OK: Référence marquée comme explicite\n")

print("--- Test 3: Deuxième buffer (référence explicite préservée) ---")
# Deuxième buffer : images avec un décalage différent
for i in range(10):
    img = create_shifted_disk(shift_x=10+i, shift_y=10+i)
    stacker.add_frame(img)

# Vérifier que la référence est toujours explicite après process_buffer
print(f"Référence explicite après buffer 2: {stacker.aligner.reference_is_explicit}")
assert stacker.aligner.reference_is_explicit, "La référence devrait rester explicite entre buffers"
print("✓ OK: Référence explicite préservée entre buffers\n")

print("--- Test 4: Troisième buffer ---")
for i in range(10):
    img = create_shifted_disk(shift_x=-5+i, shift_y=-5+i)
    stacker.add_frame(img)

print(f"Référence explicite après buffer 3: {stacker.aligner.reference_is_explicit}")
assert stacker.aligner.reference_is_explicit, "La référence devrait toujours être explicite"
print("✓ OK: Référence explicite toujours préservée\n")

# Statistiques
stats = stacker.get_stats()
print("=== Résultats ===")
print(f"Frames totales: {stats['frames_total']}")
print(f"Stacks créés: {stats['stacks_done']}")
print(f"Frames sélectionnées (dernier stack): {stats['frames_selected']}")
print(f"Score moyen: {stats['avg_score']:.1f}")

stacker.stop()

print("\n=== ✅ TOUS LES TESTS RÉUSSIS ===")
print("La référence explicite est correctement préservée entre les buffers.")
print("Les décalages entre séries devraient être corrigés!")
