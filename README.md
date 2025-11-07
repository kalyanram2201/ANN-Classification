# 🧠 Deep Learning Customer Churn Prediction (Streamlit + TensorFlow)

A **Deep Learning** web app built with **Streamlit** that predicts whether a customer is likely to churn based on demographic and account details.  
The prediction model is a neural network implemented and trained using **TensorFlow / Keras**. Pre-trained encoders and scalers are used for consistent input preprocessing and scaling.

---

## 🚀 Features

- Interactive **Streamlit UI** for user input  
- **Real-time churn prediction** using a pre-trained **deep neural network (Keras/TensorFlow)**  
- Input preprocessing with **Label Encoding**, **One-Hot Encoding**, and **Standard Scaling**  
- Clean and simple deployment-ready design  

---

## 🧩 Tech Stack

- **Python 3.x**  
- **TensorFlow / Keras** (Deep Learning)  
- **Streamlit**  
- **NumPy**, **Pandas**, **scikit-learn**  
- **Pickle** (for model and encoder loading)

---

## 📦 Project Structure

```
├── model.h5                       # Trained Keras deep learning model (saved with model.save)
├── onehot_encoder_geo.pkl         # One-hot encoder for geography
├── label_encoder_gender.pkl       # Label encoder for gender
├── scaler.pkl                     # StandardScaler for feature scaling
├── app.py                         # Streamlit application file
└── README.md                      # Project documentation
```

---

## ⚙️ Setup & Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *(If you don’t have a `requirements.txt`, run this to generate one)*  
   ```bash
   pip freeze > requirements.txt
   ```

4. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

---

## 🧠 Model Details (Deep Learning)

- **Model type:** Keras Sequential / Functional neural network (saved as `model.h5`).  
- **Task:** Binary classification (predict whether a customer will churn).  
- **Input features:** CreditScore, Gender (encoded), Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography (one-hot encoded).  
- **Output:** Churn probability (single sigmoid output).

> ⚠️ If you want the exact architecture (layers, neurons, activation functions, optimizer, training epochs, loss, metrics), add the training script or share the model summary and I’ll insert it here precisely.

---

## 🧠 Model Inputs

| Feature | Description |
|----------|--------------|
| Geography | Country or region of the customer |
| Gender | Male/Female (label-encoded) |
| Age | Customer age |
| CreditScore | Customer’s credit score |
| Tenure | Years with the bank |
| Balance | Account balance |
| NumOfProducts | Number of products owned |
| HasCrCard | Whether customer owns a credit card (0/1) |
| IsActiveMember | Whether the customer is active (0/1) |
| EstimatedSalary | Annual income |

---

## 🎯 Output

- **Churn Probability** → A float between 0 and 1 (sigmoid output)  
- **Prediction Message** →  
  - If `> 0.5`: “The customer is likely to churn.”  
  - Else: “The customer is not likely to churn.”

---

## 🔁 Model Training (optional)

Add your training script or brief notes here. Example template:

```
# Example training outline
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)
model.save('model.h5')
```

---

## 💡 Example Screenshot (optional)

*(Add a screenshot of your Streamlit app here)*

```
![App Screenshot](screenshot.png)
```

---

## 🧾 License

This project is open-source and available under the [MIT License](LICENSE).
