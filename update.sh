#!/bin/bash

echo "========================================"
echo " LocalStoryChat - GitHub Update"
echo "========================================"
echo

cd "$(dirname "$0")" || exit 1

echo "Checking repository..."
git status
echo

read -p "Commit message: " MESSAGE

if [ -z "$MESSAGE" ]; then
    echo
    echo "No commit message entered. Update cancelled."
    exit 1
fi

echo
echo "Adding changed files..."
git add .

echo
echo "Committing..."
git commit -m "$MESSAGE"

if [ $? -ne 0 ]; then
    echo
    echo "Commit failed. Nothing was pushed."
    exit 1
fi

echo
echo "Pushing to GitHub..."
git push

if [ $? -ne 0 ]; then
    echo
    echo "Push failed."
    exit 1
fi

echo
echo "========================================"
echo " Update complete!"
echo "========================================"
echo
git status
echo
read -p "Press Enter to close..."
