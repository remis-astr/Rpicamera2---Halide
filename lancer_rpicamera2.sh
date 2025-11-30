#!/bin/bash
# Script de lancement RPiCamera2 avec LiveStack
# Lance le programme depuis le bon répertoire

cd "$(dirname "$0")"
python3 RPiCamera2.py
