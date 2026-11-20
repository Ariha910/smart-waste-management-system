#!/bin/bash
set -e

echo "============================================================"
echo " Pushing Smart Waste Classification to GitHub"
echo " Remote: https://github.com/Ariha910/smart-waste-management-system.git"
echo " Branch: main"
echo "============================================================"

# Ensure remote is correctly set
git remote set-url origin https://github.com/Ariha910/smart-waste-management-system.git 2>/dev/null || git remote add origin https://github.com/Ariha910/smart-waste-management-system.git

# Push to GitHub
git push -u origin main

echo ""
echo " Successfully pushed project to GitHub!"
echo " View your repository at: https://github.com/Ariha910/smart-waste-management-system"
