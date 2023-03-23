import os

# Set the root directory
chunks_dir = '/home/student/Music/1/FYP/data/train/chunks'

# Initialize a dictionary to store the count of tracks per subgenre
subgenre_counts = {}

# Walk through the directory tree and count the tracks per subgenre
for subgenre_dir, _, files in os.walk(chunks_dir):
    subgenre = os.path.basename(os.path.dirname(subgenre_dir))
    for file in files:
        if file.endswith('.mp3'):  # Change file extension to match your audio files
            subgenre_counts[subgenre] = subgenre_counts.get(subgenre, 0) + 1

# Print the subgenre counts
for subgenre, count in subgenre_counts.items():
    print(subgenre, count)