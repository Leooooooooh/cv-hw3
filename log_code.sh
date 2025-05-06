echo "==== dataset.py ====" > clipboard.txt
cat src/dataset.py >> clipboard.txt
echo "\n==== train.py ====" >> clipboard.txt
cat src/train.py >> clipboard.txt
echo "\n==== transforms.py ====" >> clipboard.txt
cat src/transforms.py >> clipboard.txt
echo "\n==== model.py ====" >> clipboard.txt
cat src/model.py >> clipboard.txt

# Copy to clipboard (macOS)
pbcopy < clipboard.txt

# Or on Linux with xclip
# xclip -selection clipboard < clipboard.txt