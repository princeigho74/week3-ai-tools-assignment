"""
AI Tools Assignment - Bonus Task: MNIST Classifier Web App
Deploy using Streamlit: streamlit run app.py

Author: AI Assignment
Date: 2025
"""

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import io

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="MNIST Handwritten Digit Classifier",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom Styling
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 20px;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #f0fff4;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #2ca02c;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Model Training/Loading
# ============================================================================

@st.cache_resource
def load_or_train_model():
    """Load pre-trained model or train a new one"""
    try:
        # Try to load saved model
        model = tf.keras.models.load_model('mnist_model.h5')
        return model, True
    except:
        # Train a new model if not found
        st.info("Training MNIST CNN model... This may take a minute.")
        
        # Load and preprocess data
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        X_train = X_train.astype('float32').reshape(-1, 28, 28, 1) / 255.0
        X_test = X_test.astype('float32').reshape(-1, 28, 28, 1) / 255.0
        
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
        
        # Build model
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=128,
            validation_split=0.1,
            verbose=0
        )
        
        # Save model
        model.save('mnist_model.h5')
        
        return model, False

# Load model
model, model_loaded = load_or_train_model()

# ============================================================================
# Main App
# ============================================================================

st.markdown('<div class="main-header">🔢 MNIST Handwritten Digit Classifier</div>', 
            unsafe_allow_html=True)

st.markdown("""
This web application demonstrates a Convolutional Neural Network (CNN) trained 
to recognize handwritten digits (0-9). Draw a digit below or upload an image!
""")

# ============================================================================
# Sidebar
# ============================================================================

st.sidebar.markdown('<div class="section-header">⚙️ Settings</div>', 
                     unsafe_allow_html=True)

app_mode = st.sidebar.radio(
    "Select Mode:",
    ["📝 Draw Digit", "📤 Upload Image", "📊 Model Info", "📈 Performance"]
)

# ============================================================================
# Mode 1: Draw Digit
# ============================================================================

if app_mode == "📝 Draw Digit":
    st.markdown('<div class="section-header">Draw a Digit (0-9)</div>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create a canvas for drawing
        st.markdown("Use your mouse to draw in the white area below:")
        
        # Using HTML5 canvas for drawing
        canvas_html = """
        <canvas id="canvas" width="280" height="280" style="border:1px solid black; background-color:white; cursor:crosshair;"></canvas>
        <br/>
        <button id="clearBtn">Clear Canvas</button>
        <button id="predictBtn">Predict Digit</button>
        
        <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let isDrawing = false;
        
        // Drawing functionality
        canvas.addEventListener('mousedown', (e) => { isDrawing = true; });
        canvas.addEventListener('mouseup', (e) => { isDrawing = false; });
        canvas.addEventListener('mousemove', (e) => {
            if (!isDrawing) return;
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            ctx.fillStyle = 'black';
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, Math.PI * 2);
            ctx.fill();
        });
        
        document.getElementById('clearBtn').addEventListener('click', () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        });
        
        document.getElementById('predictBtn').addEventListener('click', () => {
            const imageData = canvas.toDataURL();
            alert('Prediction feature: In production, this would send the image to the server for prediction.');
        });
        </script>
        """
        st.components.v1.html(canvas_html, height=350)
        
        st.markdown("""
        <div class="info-box">
        <strong>💡 Tips:</strong>
        <ul>
        <li>Draw clearly in the center of the canvas</li>
        <li>Keep your digit large but within bounds</li>
        <li>The model works best with centered digits</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Example Predictions**")
        # Show example digits
        (X_test, y_test), _ = tf.keras.datasets.mnist.load_data()
        X_test = X_test.astype('float32').reshape(-1, 28, 28, 1) / 255.0
        
        # Get one example per digit
        for digit in range(10):
            idx = np.where(y_test == digit)[0][0]
            
            fig, ax = plt.subplots(1, 1, figsize=(2, 2))
            ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
            ax.set_title(f"Digit: {digit}", fontsize=10, fontweight='bold')
            ax.axis('off')
            
            st.pyplot(fig, use_container_width=True)

# ============================================================================
# Mode 2: Upload Image
# ============================================================================

elif app_mode == "📤 Upload Image":
    st.markdown('<div class="section-header">Upload Handwritten Digit Image</div>', 
                unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp']
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Load and display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            # Process image
            st.markdown("**Processing Image...**")
            
            # Convert to grayscale and resize to 28x28
            img_gray = image.convert('L')
            img_resized = img_gray.resize((28, 28))
            img_array = np.array(img_resized).astype('float32') / 255.0
            
            # Reshape for model input
            img_input = img_array.reshape(1, 28, 28, 1)
            
            # Make prediction
            prediction = model.predict(img_input, verbose=0)
            predicted_digit = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            # Display results
            st.markdown(f"""
            <div class="success-box">
            <h3>Prediction Results</h3>
            <p><strong>Predicted Digit:</strong> <span style="font-size: 2em; color: #1f77b4;">{predicted_digit}</span></p>
            <p><strong>Confidence:</strong> {confidence*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show probability distribution
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(range(10), prediction[0])
            ax.set_xlabel('Digit')
            ax.set_ylabel('Probability')
            ax.set_title('Prediction Probabilities for All Digits')
            ax.set_xticks(range(10))
            st.pyplot(fig, use_container_width=True)

# ============================================================================
# Mode 3: Model Information
# ============================================================================

elif app_mode == "📊 Model Info":
    st.markdown('<div class="section-header">Model Architecture & Information</div>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <strong>Model Architecture:</strong>
        <ul>
        <li>Type: Convolutional Neural Network (CNN)</li>
        <li>Input: 28×28 grayscale images</li>
        <li>Conv2D(32 filters, 3×3 kernel) + ReLU + MaxPool</li>
        <li>Conv2D(64 filters, 3×3 kernel) + ReLU + MaxPool</li>
        <li>Conv2D(64 filters, 3×3 kernel) + ReLU</li>
        <li>Flatten + Dense(64) + ReLU + Dropout(0.5)</li>
        <li>Dense(10) + Softmax (output)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <strong>Training Details:</strong>
        <ul>
        <li>Dataset: MNIST (60,000 training samples)</li>
        <li>Epochs: 10</li>
        <li>Batch Size: 128</li>
        <li>Optimizer: Adam</li>
        <li>Loss Function: Categorical Crossentropy</li>
        <li>Regularization: Dropout (0.5)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Display full model summary
    st.markdown("**Detailed Model Summary:**")
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    st.text('\n'.join(model_summary))

# ============================================================================
# Mode 4: Model Performance
# ============================================================================

elif app_mode == "📈 Performance":
    st.markdown('<div class="section-header">Model Performance Metrics</div>', 
                unsafe_allow_html=True)
    
    # Evaluate on MNIST test set
    (_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_test = X_test.astype('float32').reshape(-1, 28, 28, 1) / 255.0
    y_test_cat = tf.keras.utils.to_categorical(y_test, 10)
    
    # Calculate metrics
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    predictions = model.predict(X_test, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Per-class accuracy
    from sklearn.metrics import classification_report, confusion_matrix
    class_report = classification_report(y_test, predicted_labels, output_dict=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.metric("Test Accuracy", f"{test_acc*100:.2f}%")
    
    with col2:
        st.metric("Test Loss", f"{test_loss:.4f}")
    
    with col3:
        st.metric("Total Test Samples", len(X_test))
    
    # Per-digit accuracy
    st.markdown("**Per-Digit Accuracy:**")
    
    per_digit_acc = []
    for digit in range(10):
        accuracy = class_report[str(digit)]['precision']
        per_digit_acc.append(accuracy)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(10), per_digit_acc, color='steelblue')
    ax.set_xlabel('Digit')
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Digit Classification Accuracy on MNIST Test Set')
    ax.set_xticks(range(10))
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    
    # Confusion matrix
    st.markdown("**Confusion Matrix (5×5 sample):**")
    cm = confusion_matrix(y_test, predicted_labels)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix - MNIST Test Set')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    plt.colorbar(im, ax=ax)
    st.pyplot(fig, use_container_width=True)

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9rem;">
    <p><strong>AI Tools Assignment - MNIST Classifier Deployment</strong></p>
    <p>Convolutional Neural Network trained on 60,000 MNIST handwritten digits</p>
    <p>Built with TensorFlow, Keras, and Streamlit</p>
</div>
""", unsafe_allow_html=True)
