#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Empilement d'images pour libastrostack
"""

import numpy as np


class ImageStacker:
    """Empile progressivement les images alignées"""
    
    def __init__(self, config):
        """
        Args:
            config: StackingConfig instance
        """
        self.config = config
        self.stacked_image = None
        self.weight_map = None
    
    def stack(self, image):
        """
        Empile une nouvelle image
        
        Args:
            image: Image alignée (float array)
        
        Returns:
            Image empilée courante (copie)
        """
        if self.stacked_image is None:
            # Première image : initialisation
            self.stacked_image = image.astype(np.float64)
            
            if len(image.shape) == 3:
                # RGB : weight_map 2D
                self.weight_map = np.ones(
                    (image.shape[0], image.shape[1]),
                    dtype=np.float64
                )
            else:
                # MONO
                self.weight_map = np.ones_like(image, dtype=np.float64)
            
            self.config.num_stacked = 1
        
        else:
            # Images suivantes : moyenne glissante
            self.config.num_stacked += 1
            
            if len(image.shape) == 3:
                # RGB : traiter chaque canal
                for i in range(3):
                    valid_mask = image[:, :, i] > 0
                    
                    self.stacked_image[:, :, i][valid_mask] = (
                        self.stacked_image[:, :, i][valid_mask] * (self.config.num_stacked - 1) +
                        image[:, :, i][valid_mask]
                    ) / self.config.num_stacked
                    
                    self.weight_map[valid_mask] += 1
            
            else:
                # MONO
                valid_mask = image > 0
                
                self.stacked_image[valid_mask] = (
                    self.stacked_image[valid_mask] * (self.config.num_stacked - 1) +
                    image[valid_mask]
                ) / self.config.num_stacked
                
                self.weight_map[valid_mask] += 1
        
        # Mettre à jour bruit estimé
        self.config.noise_level = np.std(self.stacked_image)
        
        return self.stacked_image.copy()
    
    def get_result(self):
        """
        Retourne l'image empilée finale
        
        Returns:
            Image empilée (copie), ou None si vide
        """
        if self.stacked_image is None:
            return None
        
        return self.stacked_image.copy()
    
    def get_snr_improvement(self):
        """
        Calcule le gain SNR théorique
        
        Returns:
            Facteur de gain SNR (ex: 7.07 pour 50 images)
        """
        if self.config.num_stacked == 0:
            return 1.0
        
        return np.sqrt(self.config.num_stacked)
    
    def reset(self):
        """Réinitialise le stacker"""
        self.stacked_image = None
        self.weight_map = None
        self.config.num_stacked = 0
        self.config.noise_level = 0.0
