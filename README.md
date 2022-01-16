# CODESCRIBE
In this work, we propose CODESCRIBE to model the hierarchical syntax structure of code by introducing a **novel triplet position** for code summarization. Specifically, CODESCRIBE leverages the **graph neural network** and **Transformer** to preserve the structural and sequential information of code, respectively. In addition, we propose a **pointer-generator network** that pays attention to both the structure and sequential tokens of code for a better summary generation. Experiments on two real-world datasets in Java and Python demonstrate the effectiveness of our proposed approach when compared with several state-of-the-art baselines.
# Runtime Environment
- 4 NVIDIA 2080 Ti GPUs 
- Ubuntu 16.04  
- CUDA 10.2 (with CuDNN of the corresponding version)
- Anaconda
    * Python 3.9 (base environment)
    * Python 2.7 (virtual environment named as python27)
- PyTorch 1.9 for Python 3.9  
- PyTorch Geometric 1.7 for Python 3.9  
- Specifically, install our package with "pip install my-lib-0.0.6.tar.gz" for both Python 3.9 and Python 2.7. The package can be downloaded from [Google Drive](https://drive.google.com/file/d/1yAczhoUP_xYl8N_9vmPNu8RhtKVch7dc/view?usp=sharing)  
# Dataset
The whole datasets of Python and Java can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1kh3TEeoChoEqBJKdLf0HLVZXwzn2R0Xq?usp=sharing).

**Note that:** We provide 100 samples for train/valid/test datasets in the directory `data/python/raw_data/` and `data/java/raw_data/`. To run on the whole dataset, replace these samples with the data files downloaded.

# Experiment on the Python Dataset
1. Step into the directory `src_code/python/`:
    ```angular2html
    cd src_code/python
    ```
2. Proprecess the train/valid/test data:
   ```angular2html
   python s1_preprocessor.py
   conda activate python27
   python s2_preprocessor_py27.py
   conda deactivate
   python s3_preprocessor.py
    ```
3. Run the model for training and testing:
   python s4_model.py
  
After running, the performance will be printed in the console, and the predicted results of test data and will be saved in the path `data/python/result/result.json`, with ground truth and code involved for comparison.  

We have provided the results of test dataset, you can get the evaluation results directly by running 
```angular2html
python s5_eval_res.py"
```

**Note that:** 
- all the parameters are set in `src_code/python/config.py` and `src_code/python/config_py27.py`.
- If the model has been trained, you can set the parameter "train_mode" in line 83 in `config.py` to "False". Then you can predict the test data directly by using the model that has been saved in `data/python/model/`.  

# Experiment on the Java Dataset
1. Step into the directory `src_code/java/`:
    ```angular2html
    cd src_code/java
    ```
2. Proprecess the train/valid/test data:
   ```angular2html
   python s1_preprocessor.py
    ```
3. Run the model for training and testing:
   ```angular2html
   python s2_model.py
   ```
   
  
After running, the performance will be printed in the console, and the predicted results of test data and will be saved in the path `data/java/result/result.json`, with ground truth and code involved for comparison.  

We have provided the results of test dataset, you can get the evaluation results directly by running 
```angular2html
python s3_eval_res.py"
```

**Note that:** 
- all the parameters are set in `src_code/java/config.py`.
- If the model has been trained, you can set the parameter "train_mode" in line 113 in `config.py` to "False". Then you can predict the test data directly by using the model that has been saved in `data/python/model/`.

# More Implementation Details.

- The main parameter settings for CODESCRIBE are show as:

<img src="https://github.com/anonymousrepoxxx/IMG/blob/main/CODESCRIBE_IMG/IMG453.png" width="50%" height="50%" alt="params">

[comment]: <> (![params]&#40;https://github.com/anonymousrepoxxx/IMG/blob/main/CODESCRIBE_IMG/IMG453.png&#41;)

- The time used per epoch and the memory usage are provided as:

<img src="https://github.com/anonymousrepoxxx/IMG/blob/main/CODESCRIBE_IMG/IMG454.png" width="50%" height="50%" alt="usage">

[comment]: <> (![usage]&#40;https://github.com/anonymousrepoxxx/IMG/blob/main/CODESCRIBE_IMG/IMG454.png&#41;)

# Experimental Result:
We provide part of our experiment result as below.
- Comparison with state-of-the-arts. We add two pre-trained models as additional baselines for comparison.

![Comparison with the baselines](https://github.com/anonymousrepoxxx/IMG/blob/main/CODESCRIBE_IMG/IMG450.png)

- We addintionally provide human evluation.

![Human evluation](https://github.com/anonymousrepoxxx/IMG/blob/main/CODESCRIBE_IMG/IMG451.png)

- Qualitative examples.

![Examples](https://github.com/anonymousrepoxxx/IMG/blob/main/CODESCRIBE_IMG/IMG452.png)

***This paper is still under review, please do not distribute it.***

    

