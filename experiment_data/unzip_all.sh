find . -type f -name "*.zip" -exec sh -c '
for f do
  unzip -o "$f" -d "$(dirname "$f")"
done
' sh {} +