# 🌿 Plant Disease Classifier

An image classification web app that detects plant diseases from leaf photos using transfer learning. Built to help identify crop diseases early, which can support timely treatment decisions.

## Problem
Manually identifying plant diseases requires expert knowledge that isn't always accessible, especially for smallholder farmers. This tool takes a photo of a leaf and predicts the disease (or confirms the plant is healthy) in seconds.

## Approach
- **Model:** ResNet18 (pretrained on ImageNet), fine-tuned via transfer learning on the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Technique:** Froze base convolutional layers, retrained only the final classification layer — reduces training time and overfitting risk on a moderately-sized dataset
- **Data augmentation:** Random crops, flips, and rotations applied during training to improve generalization
- **Evaluation:** Tracked train/val accuracy and loss per epoch; [add your final confusion matrix / precision / recall / F1 results here]

## Results
- Achieved **80% validation accuracy** across 10 tomato disease classes after 4 epochs of transfer learning
- Trained on a class-balanced subset of the PlantVillage dataset (250 images/class train, 60 images/class validation)
- Model shows steadily improving accuracy each epoch (38% → 66% → 77% → 80%), indicating good convergence without overfitting
- Classes covered: Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites (Two-spotted), Target Spot, Tomato Yellow Leaf Curl Virus, Tomato mosaic virus, and Healthy

## Tech Stack
- **Model training:** PyTorch, torchvision
- **Interface:** Streamlit
- **Deployment:** Hugging Face Spaces / Streamlit Community Cloud

## Project Structure
```
plant-disease-classifier/
├── data/               # train/val image folders (not included — download from Kaggle)
├── split_dataset.py    # splits raw dataset into train/val folders
├── train.py            # training script
├── app.py              # Streamlit inference app
├── requirements.txt
└── README.md
```

## How to Run

1. Download the [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) and extract it
2. Update `SOURCE_DIR` in `split_dataset.py` to point to the extracted `color` folder, then run:
   ```bash
   python split_dataset.py
   ```
   This creates `data/train/` and `data/val/` folders with an 80/20 split across the selected tomato disease classes.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Train the model:
   ```bash
   python train.py
   ```
5. Run the app locally:
   ```bash
   streamlit run app.py
   ```

## Live Demo
[Add your Hugging Face Spaces / Streamlit Cloud link here once deployed]

## Future Improvements
- Expand to multiple crop types beyond the initial scope
- Add Grad-CAM visualization to show which part of the leaf influenced the prediction
- Fine-tune more layers (not just the final one) for potentially higher accuracy
- Add a confidence threshold and "uncertain" category to avoid overconfident wrong predictions

---
*Educational project — not a substitute for professional agricultural diagnosis.*
