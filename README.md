# 🧠 Deep Learning Segmentation Pipeline for the Spires-Jones Lab

This pipeline provides a deep learning-based alternative to the traditional thresholding approach used in the Spires-Jones lab.  
It supports both the **Cellpose CNN model** and the **SAM2 Transformer model** for image segmentation.

The system allows users to either retrain models or load pretrained ones via a command-line interface.  
This design enables flexible model usage without modifying the core Python code.  
Segmentation behavior can be adjusted through mode selection and parameter tuning, depending on the object type.



---

## ⚙️ Key Features
- Compatible with Cellpose and SAM2 segmentation models  
- Flexible run modes: `tune`, `train`, `evaluate`  
- Command-line interface for full pipeline control  
- Output includes logs, performance metrics, visualizations, and trained models  

---

## 📜 Example Command
```bash
python "cellpose_retraining.py" \
  -mn PSD \
  -tts 0.9 \
  -mode train_test_mixed_batch \
  -input1 "<path_to_batch_B_input>" \
  -gt1 "<path_to_batch_B_gt>" \
  -input2 "<path_to_batch_A_input>" \
  -gt2 "<path_to_batch_A_gt>" \
  -ts 0.3 \
  -tunep 1 \
  -trainp 1 \
  -testp 1 \
  -numt 7 \
  -nume 599 \
  -run_mode tune
```

## 🚀 Run Modes
The pipeline supports three main execution modes:

- **`tune`**: Runs hyperparameter tuning using the specified tuning split.  
- **`train`**: Trains the model using the given training data and number of epochs.  
- **`eval`**: Evaluates a trained model on the test dataset. The test batch can differ from the training batch depending on the selected training mode.  

---

## 📂 Output Files
Each run produces five key outputs to support training transparency and post-run analysis:

- **Logs** — Show current processing steps, training loss, and test loss. Helpful for identifying model convergence.  
- **Parameter Configurations (JSON)** — Saves hyperparameters used in the run. Useful for auditing and comparing models.  
- **Results CSVs** — Contain performance metrics for individual slices and for overlapping masks (via post-processing).  
- **Visual Comparisons** — Side-by-side plots showing original images, predicted masks, and ground truth labels for both high- and low-performing predictions.  
- **Saved Models** — Trained models are saved and can be reused for future evaluations without retraining.  

---

## 🎛️ Parameter Overview
| Parameter | Description |
|-----------|-------------|
| `-mn` | Model name (e.g., PSD). Also defines save folder structure. |
| `-tts` | Train-test split ratio (e.g., 0.8). |
| `-mode` | Segmentation mode (e.g., `train_test_mixed_batch`). |
| `-input1`, `-gt1`, `-input2`, `-gt2` | Paths to image and ground truth data. |
| `-ts` | Tuning data split ratio (e.g., 0.3). |
| `-tunep` | Proportion of tuning data used in tuning (0–1). |
| `-trainp` | Proportion of training data used in training (0–1). |
| `-testp` | Proportion of test data used in evaluation (0–1). |
| `-numt` | Number of tuning trials (e.g., 7). |
| `-nume` | Number of epochs (e.g., 599). |
| `-run_mode` | Mode to execute (`tune`, `train`, `eval`). |

---

## 🧪 Notes
- Model training and evaluation depend on the balance and quality of input data across batches.  
- Use the `train_test_mixed_batch` mode when combining batches with differing ground truths.
- Models, Image and result CSV files not uploaded to Git to large size but will be made available directly to the lab

