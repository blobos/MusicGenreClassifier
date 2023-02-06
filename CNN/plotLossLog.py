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

# Find the index of the point with the highest validation accuracy
max_index = validation_accuracy.index(max(validation_accuracy))

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

# Add an annotation to the plot marking the point with the highest validation accuracy
ax2.annotate(f"Max validation accuracy: {validation_accuracy[max_index]:.2f}\nat epoch: {epochs[max_index]}",
             xy=(epochs[max_index], validation_accuracy[max_index]),
             xytext=(30, 0), textcoords="offset points",
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))

# Display the picture
plt.tight_layout()
plt.show()