import json
import os

notebook_path = r'c:\Mern Stack\Mediconnection\backend\prediction_model.ipynb'

new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Proof of Validity & Robustness Checks\n",
            "\n",
            "Given the high accuracy, we need to verify if the model is truly learning or just memorizing.\n",
            "We will perform:\n",
            "1.  **Feature Importance Visualization**: To see if the model relies on medically relevant symptoms.\n",
            "2.  **Noise Injection Test**: To test if the model can handle imperfect data.\n",
            "3.  **Model Comparison**: To check if other models (like SVM) perform similarly."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 7.1 Feature Importance (Top Symptoms per Disease)\n",
            "# We examine the Logistic Regression coefficients.\n",
            "\n",
            "diseases_to_check = ['Fungal infection', 'Allergy', 'Drug Reaction', 'Heart attack']\n",
            "try:\n",
            "    # Map disease name to index if present in classes\n",
            "    valid_diseases = [d for d in diseases_to_check if d in model.classes_]\n",
            "    disease_indices = [np.where(model.classes_ == d)[0][0] for d in valid_diseases]\n",
            "\n",
            "    if disease_indices:\n",
            "        plt.figure(figsize=(20, 10))\n",
            "        for i, disease_idx in enumerate(disease_indices):\n",
            "            disease_name = model.classes_[disease_idx]\n",
            "            coefs = model.coef_[disease_idx]\n",
            "\n",
            "            # Get top 10 positive coefficients (symptoms strongly indicative of the disease)\n",
            "            top_indices = np.argsort(coefs)[-10:]\n",
            "            top_symptoms = [unique_symptoms[j] for j in top_indices]\n",
            "            top_values = coefs[top_indices]\n",
            "\n",
            "            plt.subplot(2, 2, i + 1)\n",
            "            sns.barplot(x=top_values, y=top_symptoms, palette='viridis')\n",
            "            plt.title(f'Top Symptoms for {disease_name}')\n",
            "            plt.xlabel('Coefficient Value')\n",
            "        plt.tight_layout()\n",
            "        plt.show()\n",
            "    else:\n",
            "        print(\"None of the sample diseases found in the model classes.\")\n",
            "except Exception as e:\n",
            "    print(f\"Could not plot feature importance: {e}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 7.2 Robustness Test: Noise Injection\n",
            "# We inject random noise into the test set (flipping 0s to 1s and vice versa).\n",
            "# A robust model should maintain decent accuracy even with some noise.\n",
            "\n",
            "def add_noise(X, noise_level):\n",
            "    X_noisy = X.copy()\n",
            "    # Ensure X is discrete (0/1) for bit flipping, though random addition works for continuous too.\n",
            "    # Here filtering to flat array for ease\n",
            "    if hasattr(X_noisy, 'values'):\n",
            "        flat_X = X_noisy.values.flatten()\n",
            "    else:\n",
            "        flat_X = X_noisy.flatten()\n",
            "    \n",
            "    # Number of features to flip\n",
            "    n_flip = int(noise_level * flat_X.size)\n",
            "    if n_flip > 0:\n",
            "        indices = np.random.choice(flat_X.size, n_flip, replace=False)\n",
            "        flat_X[indices] = 1 - flat_X[indices] # Flip bits\n",
            "    \n",
            "    if hasattr(X_noisy, 'values'):\n",
            "        return pd.DataFrame(flat_X.reshape(X_noisy.shape), columns=X_noisy.columns)\n",
            "    else:\n",
            "        return flat_X.reshape(X_noisy.shape)\n",
            "\n",
            "print(\"\\n--- Robustness Test (Noise Injection) ---\")\n",
            "noise_levels = [0.01, 0.05, 0.10, 0.20]\n",
            "for level in noise_levels:\n",
            "    X_test_noisy = add_noise(X_test, level)\n",
            "    y_pred_noisy = model.predict(X_test_noisy)\n",
            "    acc = accuracy_score(y_test, y_pred_noisy)\n",
            "    print(f\"Accuracy with {int(level*100)}% random noise: {acc:.4f}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 7.3 Model Comparison (SVM)\n",
            "# If SVM also gets 100%, the dataset is likely linearly separable.\n",
            "\n",
            "from sklearn.svm import SVC\n",
            "\n",
            "print(\"\\n--- Model Comparison ---\")\n",
            "svm_model = SVC(kernel='linear')\n",
            "svm_model.fit(X_train, y_train)\n",
            "svm_acc = svm_model.score(X_test, y_test)\n",
            "print(f\"Support Vector Machine (Linear Kernel) Accuracy: {svm_acc:.4f}\")"
        ]
    }
]

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Find insertion point
    insert_idx = -1
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            source = "".join(cell['source'])
            if "7. Saving the Model" in source:
                insert_idx = i
                # Update the title
                cell['source'] = ["## 8. Saving the Model & Inference Function"]
                break
    
    if insert_idx != -1:
        # Insert new cells
        for cell in reversed(new_cells):
            nb['cells'].insert(insert_idx, cell)
        
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=4)
        print("Notebook updated successfully.")
    else:
        print("Error: Could not find insertion point '## 7. Saving the Model'.")

except Exception as e:
    print(f"Error updating notebook: {e}")
