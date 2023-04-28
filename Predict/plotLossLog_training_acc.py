import matplotlib.pyplot as plt
import pandas as pd
model = "/home/student/Music/1/FYP/MusicGenreClassifier/CRNN/checkpoints/training_log.txt"
# Read the data from the text file
with open(model, "r") as f:
    data = f.read()

# Split the data into lines
lines = data.strip().split("\n")

# Extract the epoch, training loss, validation loss, and validation accuracy from each line
epochs = []
training_loss = []
validation_loss = []
validation_accuracy = []
training_accuracy = []
for line in lines:
    parts = line.split(", ")
    print(parts)
    epochs.append(int(parts[0].split(" ")[-1]))
    training_loss.append(float(parts[1].split(": ")[1]))
    training_accuracy.append(float(parts[2].split(": ")[1]))
    validation_loss.append(float(parts[3].split(": ")[1]))
    validation_accuracy.append(float(parts[4].split(": ")[1]))

df = pd.DataFrame(list(zip(epochs, training_loss, training_accuracy, validation_loss, validation_accuracy)),
               columns =['Epochs', 'Training loss', 'Training accuracy', 'Validation loss', 'Validation accuracy'])

# Find the index of the point with the highest validation accuracy
# max_index_acc = validation_accuracy.index(max(validation_accuracy))

# Find the index of the point with the lowest validation loss
# min_index_loss = validation_loss.index(min(validation_loss))
# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot the training loss and validation loss in the first subplot
ax1.plot(df['Epochs'], df['Training loss'], label="Training Loss")
ax1.plot(df['Epochs'], df['Validation loss'], label="Validation Loss")
ax1.set_xlabel("Epoch")
ax1.legend()

min_val_loss_row = df[df['Validation loss'] == df['Validation loss'].min()]
# print(min_val_loss_row)
min_val_loss = min_val_loss_row['Validation loss'].item()
# print(min_val_loss)
min_val_loss_epoch = min_val_loss_row['Epochs'].item()
# print(min_val_loss_epoch)
ax1.annotate(f"Min validation loss: {min_val_loss:.2f}\n at epoch: {min_val_loss_epoch}",
             xy=(min_val_loss_epoch, min_val_loss),
             xytext=(30, 0), textcoords="offset points",
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))

#              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))
# ax1.annotate(f"Min validation loss: {validation_loss[min_index_loss]:.2f}\nat epoch: {epochs[min_index_loss]}",
#              xy=(epochs[min_index_loss], validation_loss[min_index_loss]),
#              xytext=(30, 0), textcoords="offset points",
#              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))

# Plot the validation accuracy in the second subplot
ax2.plot(epochs, training_accuracy, label="Training Accuracy")
ax2.plot(epochs, validation_accuracy, label="Validation Accuracy")
ax2.set_xlabel("Epoch")
ax2.legend()

max_val_acc_row = df[df['Validation accuracy'] == df['Validation accuracy'].max()]
max_val_acc_row = max_val_acc_row.sort_values('Validation loss', ascending=False)
print(max_val_acc_row)
max_val_acc = max_val_acc_row.iloc[0]['Validation accuracy']
print(max_val_acc)
max_val_acc_epoch = max_val_acc_row['Epochs'].iloc[0].item()
print(max_val_acc_epoch)
ax2.annotate(f"Max validation accuracy: {max_val_acc:.2f}\nat epoch: {max_val_acc_epoch}",
             xy=(max_val_acc_epoch, max_val_acc),
             xytext=(30, 0), textcoords="offset points",
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))
# Add an annotation to the plot marking the point with the highest validation accuracy
# ax2.annotate(f"Max validation accuracy: {validation_accuracy[max_index_acc]:.2f}\nat epoch: {epochs[max_index_acc]}",
#              xy=(epochs[max_index_acc], validation_accuracy[max_index_acc]),
#              xytext=(30, 0), textcoords="offset points",
#              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))



# Display the picture
plt.tight_layout()
plt.savefig(model + "_Loss.png")
plt.show()

#Remove validation loss spikes/Remove Outliers
#
# # Q1 = df['Validation loss'].quantile(0.25)
# # Q3 = df['Validation loss'].quantile(0.75)
# # IQR = Q3 - Q1
# # upper_bound = Q3 + 1.5 * IQR
# # print(upper_bound)
# df = df[(df['Validation loss'] <= 3)]
#
#
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
#
# # Plot the training loss and validation loss in the first subplot
# ax1.plot(df['Epochs'], df['Training loss'], label="Training Loss")
# ax1.plot(df['Epochs'], df['Validation loss'], label="Validation Loss")
# ax1.set_xlabel("Epoch")
# ax1.legend()
#
# min_val_loss_row = df[df['Validation loss'] == df['Validation loss'].min()]
# # print(min_val_loss_row)
# min_val_loss = min_val_loss_row['Validation loss'].item()
# # print(min_val_loss)
# min_val_loss_epoch = min_val_loss_row['Epochs'].item()
# # print(min_val_loss_epoch)
# ax1.annotate(f"Min validation loss: {min_val_loss:.2f}\n at epoch: {min_val_loss_epoch}",
#              xy=(min_val_loss_epoch, min_val_loss),
#              xytext=(30, 0), textcoords="offset points",
#              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))
#
# #              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))
# # ax1.annotate(f"Min validation loss: {validation_loss[min_index_loss]:.2f}\nat epoch: {epochs[min_index_loss]}",
# #              xy=(epochs[min_index_loss], validation_loss[min_index_loss]),
# #              xytext=(30, 0), textcoords="offset points",
# #              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))
#
# # Plot the validation accuracy in the second subplot
# ax2.plot(epochs, validation_accuracy, label="Validation Accuracy")
# ax2.set_xlabel("Epoch")
# ax2.legend()
#
# max_val_acc_row = df[df['Validation accuracy'] == df['Validation accuracy'].max()]
# max_val_acc_row = min_val_loss_row.sort_values('Validation loss', ascending=False)
# # print(max_val_acc_row)
# max_val_acc = max_val_acc_row.iloc[0]['Validation accuracy']
# # print(max_val_acc)
# max_val_acc_epoch = max_val_acc_row['Epochs'].iloc[0].item()
# # print(max_val_acc_epoch)
# ax2.annotate(f"Max validation accuracy: {max_val_acc:.2f}\nat epoch: {max_val_acc_epoch}",
#              xy=(max_val_acc_epoch, max_val_acc),
#              xytext=(30, 0), textcoords="offset points",
#              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))
# # Add an annotation to the plot marking the point with the highest validation accuracy
# # ax2.annotate(f"Max validation accuracy: {validation_accuracy[max_index_acc]:.2f}\nat epoch: {epochs[max_index_acc]}",
# #              xy=(epochs[max_index_acc], validation_accuracy[max_index_acc]),
# #              xytext=(30, 0), textcoords="offset points",
# #              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))
#
#
#
# # Display the picture
# plt.tight_layout()
# plt.savefig(model + "_Loss_val_loss_outliers_removed.png")
# plt.show()