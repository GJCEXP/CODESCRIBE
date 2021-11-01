<<<<<<< HEAD
Runtime Environment in our experiments:  
4 NVIDIA 2080 Ti GPUs  
Ubuntu 16.04  
CUDA 10.2 (with CuDNN of the corresponding version)  
Python 3.9  
Python 2.7  
PyTorch 1.9 for Python 3.9  
PyTorch Geometric 1.7 for Python 3.9  
Specifically, you should install our package with "pip install my-lib-0.0.6.tar.gz" for both Python 3.9 and Python 2.7. The package can be downloaded at https://drive.google.com/file/d/1yAczhoUP_xYl8N_9vmPNu8RhtKVch7dc/view?usp=sharing  

For Python dataset:  
We provide 100 samples for train/valid/test dataset in the directory "data/python/raw_data/".  
With the full Python dataset, step into the directory "src_code/python/" to run the experiment:  
    1. Run "python s1_preprocessor.py" with Python 3.9 for preprocessing.  
    2. Run "python s2_preprocessor_py27.py" with Python 2.7 for preprocessing.  
    3. Run "python s3_preprocessor.py" with Python 3.9 for preprocessing.  
    4. Run "python s4_model.py" with Python 3.9 to train and test the model.  
After running, the performance will be printed in the console, and the predicted results of test data and will be saved in the path "data/python/result/result.json", with ground truth and code involved for comparison.  
We have provided the results of test dataset, you can run "python s5_eval_res.py" with Python 3.9 to get the evaluation results.  
Note that: all the parameters are set in "src_code/python/config.py" and "src_code/python/config_py27.py".  
If the model has been trained, you can set the parameter "train_mode" in line 83 in config.py to "False". Then you can predict the test data directly by using the model saved.  

For Java dataset:  
We provide 100 samples for train/valid/test dataset in the directory "data/java/raw_data/".  
With the full Python dataset, step into the directory "src_code/java/" to run the experiment:  
    1. Run "python s1_preprocessor.py" with Python 3.9 for preprocessing.  
    2. Run "python s2_model.py" with Python 3.9 to train and test the model.  
After running, the performance will be printed in the console, and the predicted results of test data and will be saved in the path "data/java/result/result.json", with ground truth and code involved for comparison.  
We have provided the results of test dataset, you can run "python s3_eval_res.py" with Python 3.9 to get the evaluation results.  
Note that: all the parameters are set in "src_code/python/config.py".  
If the model has been trained, you can set the parameter "train_mode" in line 113 in config.py to "False". Then you can predict the test data directly by using the model saved.  

The whole datasets of Python and Java can be downloaded at https://drive.google.com/drive/folders/1Mx0xEPZfQzb5h0z753XV-JgoWUuxiuKZ
=======
# CODESCRIBE
>>>>>>> f51b58b31295c94ffa58dab3261ad0187e67ca8b
