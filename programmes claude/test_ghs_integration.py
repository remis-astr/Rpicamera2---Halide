#!/usr/bin/env python3
"""
Test d'intégration GHS 5 paramètres dans libastrostack
Vérifie que les nouveaux paramètres fonctionnent correctement
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from libastrostack.io import load_image
from libastrostack.stretch import stretch_ghs, stretch_asinh, apply_stretch
from libastrostack.config import StretchMethod, StackingConfig


def test_stretch_ghs_direct():
    """Test 1: Fonction stretch_ghs() directement avec nouveaux paramètres"""
    print("="*80)
    print("TEST 1: stretch_ghs() - Appel direct")
    print("="*80)

    image_path = "/media/admin/THKAILAR/M81/250917224338_0.dng"
    data = load_image(image_path)

    if len(data.shape) == 3:
        data = np.mean(data, axis=2)
    if data.max() > 1.0:
        data = data / data.max()

    # Test avec paramètres optimisés (trouvés via sliders)
    result = stretch_ghs(
        data,
        D=3.06,
        b=0.131,
        SP=0.194,
        LP=0.0,
        HP=0.0,
        clip_low=2.0,
        clip_high=98.0
    )

    print(f"  • Image shape: {result.shape}")
    print(f"  • Range: [{result.min():.3f}, {result.max():.3f}]")
    print(f"  • Mean: {result.mean():.6f}")
    print(f"  • Pixels >0.5: {(result > 0.5).sum() / result.size * 100:.2f}%")

    if result.mean() > 0.5:
        print("  ✅ Résultat correct (Mean > 0.5)")
        return result, True
    else:
        print("  ❌ Résultat trop sombre (Mean < 0.5)")
        return result, False


def test_apply_stretch_with_ghs():
    """Test 2: apply_stretch() avec méthode GHS"""
    print("\n" + "="*80)
    print("TEST 2: apply_stretch() - Via méthode GHS")
    print("="*80)

    image_path = "/media/admin/THKAILAR/M81/250917224338_0.dng"
    data = load_image(image_path)

    if len(data.shape) == 3:
        data = np.mean(data, axis=2)
    if data.max() > 1.0:
        data = data / data.max()

    # Test avec nouveaux paramètres via apply_stretch
    result = apply_stretch(
        data,
        method=StretchMethod.GHS,
        ghs_D=3.06,
        ghs_b=0.131,
        ghs_SP=0.194,
        ghs_LP=0.0,
        ghs_HP=0.0,
        clip_low=2.0,
        clip_high=98.0
    )

    print(f"  • Image shape: {result.shape}")
    print(f"  • Range: [{result.min():.3f}, {result.max():.3f}]")
    print(f"  • Mean: {result.mean():.6f}")
    print(f"  • Pixels >0.5: {(result > 0.5).sum() / result.size * 100:.2f}%")

    if result.mean() > 0.5:
        print("  ✅ Résultat correct")
        return result, True
    else:
        print("  ❌ Résultat trop sombre")
        return result, False


def test_backward_compatibility():
    """Test 3: Rétro-compatibilité avec ancien paramètre ghs_B"""
    print("\n" + "="*80)
    print("TEST 3: Rétro-compatibilité - Paramètre ghs_B")
    print("="*80)

    image_path = "/media/admin/THKAILAR/M81/250917224338_0.dng"
    data = load_image(image_path)

    if len(data.shape) == 3:
        data = np.mean(data, axis=2)
    if data.max() > 1.0:
        data = data / data.max()

    # Test avec ancien nom de paramètre ghs_B
    result = apply_stretch(
        data,
        method=StretchMethod.GHS,
        ghs_D=3.06,
        ghs_B=0.131,  # Ancien nom
        ghs_SP=0.194,
        clip_low=2.0,
        clip_high=98.0
    )

    print(f"  • Test avec ghs_B (ancien nom)")
    print(f"  • Mean: {result.mean():.6f}")

    # Comparer avec nouveau nom
    result_new = apply_stretch(
        data,
        method=StretchMethod.GHS,
        ghs_D=3.06,
        ghs_b=0.131,  # Nouveau nom
        ghs_SP=0.194,
        clip_low=2.0,
        clip_high=98.0
    )

    print(f"  • Test avec ghs_b (nouveau nom)")
    print(f"  • Mean: {result_new.mean():.6f}")

    # Vérifier identité
    if np.allclose(result, result_new):
        print("  ✅ Rétro-compatibilité OK (ghs_B = ghs_b)")
        return True
    else:
        print("  ❌ Différence détectée entre ghs_B et ghs_b")
        return False


def test_config_defaults():
    """Test 4: Valeurs par défaut de StackingConfig"""
    print("\n" + "="*80)
    print("TEST 4: Valeurs par défaut StackingConfig")
    print("="*80)

    config = StackingConfig()

    print(f"  • ghs_D:  {config.ghs_D} (attendu: 3.0)")
    print(f"  • ghs_b:  {config.ghs_b} (attendu: 0.13)")
    print(f"  • ghs_SP: {config.ghs_SP} (attendu: 0.2)")
    print(f"  • ghs_LP: {config.ghs_LP} (attendu: 0.0)")
    print(f"  • ghs_HP: {config.ghs_HP} (attendu: 0.0)")

    checks = [
        (config.ghs_D, 3.0, "ghs_D"),
        (config.ghs_b, 0.13, "ghs_b"),
        (config.ghs_SP, 0.2, "ghs_SP"),
        (config.ghs_LP, 0.0, "ghs_LP"),
        (config.ghs_HP, 0.0, "ghs_HP")
    ]

    all_ok = True
    for actual, expected, name in checks:
        if abs(actual - expected) < 0.001:
            print(f"  ✅ {name} correct")
        else:
            print(f"  ❌ {name} incorrect: {actual} != {expected}")
            all_ok = False

    return all_ok


def test_comparison_arcsinh_vs_ghs():
    """Test 5: Comparaison ARCSINH vs GHS optimisé"""
    print("\n" + "="*80)
    print("TEST 5: Comparaison ARCSINH vs GHS")
    print("="*80)

    image_path = "/media/admin/THKAILAR/M81/250917224338_0.dng"
    data = load_image(image_path)

    if len(data.shape) == 3:
        data = np.mean(data, axis=2)
    if data.max() > 1.0:
        data = data / data.max()

    # ARCSINH (référence)
    result_arcsinh = stretch_asinh(data, factor=10.0, clip_low=1.0, clip_high=99.5)

    # GHS optimisé
    result_ghs = stretch_ghs(data, D=3.06, b=0.131, SP=0.194, LP=0.0, HP=0.0,
                            clip_low=2.0, clip_high=98.0)

    print(f"  • ARCSINH Mean: {result_arcsinh.mean():.6f}")
    print(f"  • GHS Mean:     {result_ghs.mean():.6f}")

    diff_pct = (result_ghs.mean() - result_arcsinh.mean()) / result_arcsinh.mean() * 100
    print(f"  • Différence:   {diff_pct:+.2f}%")

    # Créer visualisation
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Validation Intégration GHS 5 Paramètres', fontsize=14, fontweight='bold')

    # Original
    axes[0].imshow(data, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'Original\nMean: {data.mean():.3f}', fontsize=12)
    axes[0].axis('off')

    # ARCSINH
    axes[1].imshow(result_arcsinh, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'ARCSINH (référence)\nMean: {result_arcsinh.mean():.3f}',
                     fontsize=12, fontweight='bold', color='blue')
    axes[1].axis('off')

    # GHS
    axes[2].imshow(result_ghs, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'GHS 5 paramètres\n(D=3.06, b=0.131, SP=0.194)\nMean: {result_ghs.mean():.3f}',
                     fontsize=12, fontweight='bold', color='green')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('test_ghs_integration.png', dpi=150, bbox_inches='tight')
    print(f"\n  💾 Sauvegardé: test_ghs_integration.png")

    if result_ghs.mean() >= result_arcsinh.mean() * 0.95:
        print(f"  ✅ GHS comparable ou meilleur qu'ARCSINH")
        return True
    else:
        print(f"  ❌ GHS significativement moins bon qu'ARCSINH")
        return False


if __name__ == "__main__":
    print("="*80)
    print("TESTS D'INTÉGRATION GHS 5 PARAMÈTRES")
    print("="*80)
    print()

    results = {}

    # Test 1: stretch_ghs() direct
    _, results['test1'] = test_stretch_ghs_direct()

    # Test 2: apply_stretch()
    _, results['test2'] = test_apply_stretch_with_ghs()

    # Test 3: Rétro-compatibilité
    results['test3'] = test_backward_compatibility()

    # Test 4: Config defaults
    results['test4'] = test_config_defaults()

    # Test 5: Comparaison ARCSINH vs GHS
    results['test5'] = test_comparison_arcsinh_vs_ghs()

    # Résumé
    print("\n" + "="*80)
    print("RÉSUMÉ DES TESTS")
    print("="*80)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSÉ" if passed else "❌ ÉCHOUÉ"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*80)
    if all_passed:
        print("✅ TOUS LES TESTS SONT PASSÉS!")
        print("L'intégration GHS 5 paramètres est fonctionnelle.")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        print("Vérifier les erreurs ci-dessus.")
    print("="*80)

    sys.exit(0 if all_passed else 1)
