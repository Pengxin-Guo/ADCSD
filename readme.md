# Online Test-Time Adaptation of Spatial-Temporal Forecasting

This is a PyTorch implementation of  Adaptive Double Correction by Series Decomposition (**ADCSD**) for Online Test-Time Adaptation of Spatial-Temporal Forecasting.

## Requirements

Our code is based on Python version 3.10 and PyTorch version 2.0.1. Please make sure you have installed Python and PyTorch correctly. Then you can install all the dependencies with the following command by pip:

```shell
pip install -r requirements.txt
```

## Data

You can get the PeMS07,  NYCTaxi and T-Drive datasets from the [LibCity](https://github.com/LibCity/Bigscity-LibCity) repository and the BayArea dataset will be released soon.

All datasets used in LibCity needs to be processed into the [atomic files](https://bigscity-libcity-docs.readthedocs.io/en/latest/user_guide/data/atomic_files.html) format.

## Train

First, you need to train a model on the historical data. For example, train a PDFormer model on the PeMS07 or NYCTaxi dataset:

```shell
python run_model.py --task traffic_state_pred --model PDFormer --dataset PeMS07 --config_file PeMS07
python run_model.py --task traffic_state_pred --model PDFormer --dataset NYCTaxi --config_file NYCTaxi --evaluator TrafficStateGridEvaluator
```

## Fine-tune & Test

Then, fine-tune and test the trained model with the proposed ADCSD on the future data. For example, fine-tune and test the PDFormer model trained above on the PeMS07 or NYCTaxi dataset, assuming the experiment ID during training is $ID:

```shell
python run_model.py --task traffic_state_pred --model PDFormer --dataset PeMS07 --config_file PeMS07 --train false --exp_id $ID --batch_size 1 --method ADCSD --learning_rate 0.0001
python run_model.py --task traffic_state_pred --model PDFormer --dataset NYCTaxi --config_file NYCTaxi --evaluator TrafficStateGridEvaluator --train false --exp_id $ID --batch_size 1 --method ADCSD --learning_rate 0.0001
```


