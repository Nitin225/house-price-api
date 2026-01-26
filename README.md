# Housing Price Prediction (ML)

A small ML project that trains a **RandomForestRegressor** to predict `median_house_value` using the California housing dataset, with a saved preprocessing pipeline for easy inference.

## Contents
- `main.py` : Training + inference script  
- `housing.csv` : Dataset (required)  
- `model.pkl` : Saved trained model (generated after training)  
- `pipeline.pkl` : Saved preprocessing pipeline (generated after training)  
- `input.csv` : Test input file for inference (auto-generated)  
- `output.csv` : Final predictions output (generated after inference)

## Build Requirements
- Python 3.x  
- numpy, pandas, scikit-learn, joblib  

Install dependencies:
```bash
pip install numpy pandas scikit-learn joblib
