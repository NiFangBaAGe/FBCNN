# FBCNN
This page shows the original code of the paper 'Fine-grained Breast Cancer Classification with Bilinear Convolutional Neural Networks(BCNNs)'.

# Running
python main.py

## Model Selection
You can config it in main.py.
bcnn_cfg: 0: base model,  1: fast bcnn,  2: bcnn.   attention_module: se_block: 'se_block',  None: None.
They have been implemented in the load_model.py.
