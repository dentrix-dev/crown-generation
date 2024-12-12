import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Function to create the plots
def plot_training_data(train_miou=None, test_miou=None, train_acc=None, test_acc=None, train_accuracy=None, test_accuracy=None, train_loss=None, test_loss=None, save_dir='plots'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if train_miou!=None:
        # Plot training and test accuracy
        plt.figure()
        plt.plot(train_miou, label='Train mIOU')
        plt.plot(test_miou, label='Test mIOU')
        plt.xlabel('Epochs')
        plt.ylabel('mIOU')
        plt.title('Training vs Test mIOU')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'mIOU_plot.png'))
        plt.close()

    if train_acc!=None:
        # Plot training and test accuracy
        plt.figure()
        plt.plot(train_acc, label='Train Accuracy per class')
        plt.plot(test_acc, label='Test Accuracy per class')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy per class')
        plt.title('Training vs Test Accuracy per class')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'accuracy_per_class_plot.png'))
        plt.close()

    if train_accuracy!=None:
        # Plot training and test accuracy
        plt.figure()
        plt.plot(train_accuracy, label='Train Accuracy')
        plt.plot(test_accuracy, label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training vs Test Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'accuracy_plot.png'))
        plt.close()

    if train_loss!=None:
        # Plot training and test loss
        plt.figure()
        plt.plot(train_loss, label='Train Loss')
        plt.plot(test_loss, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training vs Test Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
        plt.close()