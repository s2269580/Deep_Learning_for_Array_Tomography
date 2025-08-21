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
