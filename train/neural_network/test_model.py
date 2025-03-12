import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import os

def load_test_data(base_path):
    """
    Load test data from the specified directory
    
    Args:
        base_path (str): Base path to the image directory
    
    Returns:
        Test dataset and class names
    """
    data_dir = pathlib.Path(base_path)
    
    if not data_dir.exists():
        raise ValueError(f"Directory does not exist: {data_dir}")
    
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        str(data_dir / 'test'),
        color_mode='rgb',
        seed=123,
        image_size=(224, 224),
        batch_size=32,
        label_mode='binary'
    )
    
    # Get class names (**directory name**)
    class_names = test_dataset.class_names
    # print(f"Class names: {class_names}")
    
    # Normalize images
    test_dataset = test_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    
    return test_dataset, class_names

def evaluate_model(model, test_dataset):
    """
    Evaluate model performance with medical-specific metrics
    
    Args:
        model: Trained TensorFlow model
        test_dataset: Test dataset
    
    Returns:
        Evaluation results
    """
    print("Evaluating model...")
    results = model.evaluate(test_dataset)
    
    # Compile evaluation conclusion 
    metrics = {name: value for name, value in zip(model.metrics_names, results)}
    print("\nTest Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

def plot_confusion_matrix(model, test_dataset, class_names):
    """
    Generate and plot confusion matrix
    """
    # Collect all predictions and true labels
    y_true = []
    y_pred = []
    
    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)
        pred_classes = (predictions > 0.5).astype(int)
        
        y_true.extend(labels.numpy())
        y_pred.extend(pred_classes.flatten())
    
    # Init confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png') # Save as png
    plt.close()
    
    # Print-runtime classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return cm

def plot_roc_curve(model, test_dataset):
    """
    Generate and plot ROC curve
    """
    # Collect all predictions and true labels
    y_true = []
    y_pred_proba = []
    
    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)
        
        y_true.extend(labels.numpy())
        y_pred_proba.extend(predictions.flatten())
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png') # Save as png
    plt.close()
    
    return roc_auc

# Samples visualize prediction image
def visualize_predictions(model, test_dataset, class_names, num_samples=5):
    """
    Visualize model predictions on sample images
    """
    images, labels = next(iter(test_dataset))
    
    # Make predictions
    predictions = model.predict(images[:num_samples])
    pred_classes = (predictions > 0.5).astype(int)
    
    # Plot images with predictions
    plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        
        # Convert tensor
        img = images[i].numpy()
        
        # Show image
        plt.imshow(img)
        
        # Get actual and predicted class names
        true_class = class_names[int(labels[i])]
        pred_class = class_names[int(pred_classes[i][0])]
        confidence = predictions[i][0]
        
        # Set title color based on correct/incorrect prediction
        title_color = 'green' if true_class == pred_class else 'red'
        plt.title(f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}", 
                 color=title_color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.close()

def main():
    """
    Main function to load model and test data
    """
    # Define paths
    base_path = '../data/neural/chest_xray'
    model_path = 'pneumonia_model.h5'
    
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        # Load model
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        model.summary()
        
        # Load test
        test_dataset, class_names = load_test_data(base_path)
        
        # Evaluate model
        metrics = evaluate_model(model, test_dataset)
        
        # Visualize results
        cm = plot_confusion_matrix(model, test_dataset, class_names)
        roc_auc = plot_roc_curve(model, test_dataset)
        visualize_predictions(model, test_dataset, class_names)
        
        print("\nTesting completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()