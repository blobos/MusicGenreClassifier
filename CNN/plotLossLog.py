import matplotlib.pyplot as plt

# Read the data from the text file
with open("training_log_copy.txt", "r") as f:
    data = f.read()

# Split the data into lines
lines = data.strip().split("\n")

# Extract the epoch, training loss, validation loss, and validation accuracy from each line
epochs = []
training_loss = []
validation_loss = []
validation_accuracy = []
for line in lines:
    parts = line.split(", ")
    epoch = int(parts[0].split(" ")[1])
    epochs.append(epoch)
    training_loss.append(float(parts[1].split(": ")[1]))
    validation_loss.append(float(parts[2].split(": ")[1]))
    validation_accuracy.append(float(parts[3].split(": ")[1]))

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot the training loss and validation loss in the first subplot
ax1.plot(epochs, training_loss, label="Training Loss")
ax1.plot(epochs, validation_loss, label="Validation Loss")
ax1.set_xlabel("Epoch")
ax1.legend()

# Plot the validation accuracy in the second subplot
ax2.plot(epochs, validation_accuracy, label="Validation Accuracy")
ax2.set_xlabel("Epoch")
ax2.legend()

# Display the picture
plt.tight_layout()
plt.show()

