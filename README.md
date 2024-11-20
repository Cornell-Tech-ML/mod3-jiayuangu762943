# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

- Docs: <https://minitorch.github.io/>

- Overview: <https://minitorch.github.io/module3.html>

You will need to modify `tensor_functions.py` slightly in this assignment.

- Tests:

```
python run_tests.py
```

- Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

# 3.4

Comparison of fast versus GPU implementation
![task 3.4](images/3.4.png)

# 3.5

## Simple dataset

- CPU
  HIDDEN == 100 <br><br>
  RATE == 0.05 <br><br>

  Epoch 0 | Loss: 5.8553 | Correct: 47 | Time: 18.90 sec<br>
  Epoch 10 | Loss: 1.2436 | Correct: 49 | Time: 0.19 sec<br>
  Epoch 20 | Loss: 1.4908 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 30 | Loss: 1.0256 | Correct: 50 | Time: 0.15 sec<br>
  Epoch 40 | Loss: 0.4500 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 50 | Loss: 0.4163 | Correct: 49 | Time: 0.14 sec<br>
  Epoch 60 | Loss: 0.3871 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 70 | Loss: 0.2856 | Correct: 50 | Time: 0.15 sec<br>
  Epoch 80 | Loss: 0.7351 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 90 | Loss: 0.1316 | Correct: 50 | Time: 0.34 sec<br>
  Epoch 100 | Loss: 0.7973 | Correct: 50 | Time: 0.14 sec<br>
  Epoch 110 | Loss: 0.8426 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 120 | Loss: 0.1419 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 130 | Loss: 0.0783 | Correct: 50 | Time: 0.14 sec<br>
  Epoch 140 | Loss: 0.7813 | Correct: 50 | Time: 0.14 sec<br>
  Epoch 150 | Loss: 0.3118 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 160 | Loss: 0.6614 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 170 | Loss: 0.2142 | Correct: 50 | Time: 0.15 sec<br>
  Epoch 180 | Loss: 0.2699 | Correct: 50 | Time: 0.27 sec<br>
  Epoch 190 | Loss: 0.1091 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 200 | Loss: 0.2515 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 210 | Loss: 0.1938 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 220 | Loss: 0.0473 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 230 | Loss: 0.1314 | Correct: 50 | Time: 0.14 sec<br>
  Epoch 240 | Loss: 0.0239 | Correct: 50 | Time: 0.15 sec<br>
  Epoch 250 | Loss: 0.2966 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 260 | Loss: 0.2730 | Correct: 50 | Time: 0.20 sec<br>
  Epoch 270 | Loss: 0.0150 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 280 | Loss: 0.0173 | Correct: 50 | Time: 0.14 sec<br>
  Epoch 290 | Loss: 0.1972 | Correct: 50 | Time: 0.14 sec<br>
  Epoch 300 | Loss: 0.1963 | Correct: 50 | Time: 0.14 sec<br>
  Epoch 310 | Loss: 0.0004 | Correct: 50 | Time: 0.14 sec<br>
  Epoch 320 | Loss: 0.5164 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 330 | Loss: 0.2547 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 340 | Loss: 0.0342 | Correct: 50 | Time: 0.14 sec<br>
  Epoch 350 | Loss: 0.0371 | Correct: 50 | Time: 0.24 sec<br>
  Epoch 360 | Loss: 0.1412 | Correct: 50 | Time: 0.14 sec<br>
  Epoch 370 | Loss: 0.0015 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 380 | Loss: 0.0974 | Correct: 50 | Time: 0.15 sec<br>
  Epoch 390 | Loss: 0.1217 | Correct: 50 | Time: 0.14 sec<br>
  Epoch 400 | Loss: 0.3376 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 410 | Loss: 0.4263 | Correct: 50 | Time: 0.14 sec<br>
  Epoch 420 | Loss: 0.3769 | Correct: 50 | Time: 0.14 sec<br>
  Epoch 430 | Loss: 0.0377 | Correct: 50 | Time: 0.21 sec<br>
  Epoch 440 | Loss: 0.1790 | Correct: 50 | Time: 0.14 sec<br>
  Epoch 450 | Loss: 0.0125 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 460 | Loss: 0.0017 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 470 | Loss: 0.3521 | Correct: 50 | Time: 0.14 sec<br>
  Epoch 480 | Loss: 0.0124 | Correct: 50 | Time: 0.14 sec<br>
  Epoch 490 | Loss: 0.1626 | Correct: 50 | Time: 0.14 sec<br>

- GPU
  HIDDEN == 100 <br><br>
  RATE == 0.05 <br><br>

  Epoch 0 | Loss: 5.3618 | Correct: 41 | Time: 5.33 sec<br>
  Epoch 10 | Loss: 2.1399 | Correct: 48 | Time: 1.44 sec<br>
  Epoch 20 | Loss: 1.3034 | Correct: 50 | Time: 1.49 sec<br>
  Epoch 30 | Loss: 1.0336 | Correct: 48 | Time: 1.45 sec<br>
  Epoch 40 | Loss: 0.9926 | Correct: 50 | Time: 1.97 sec<br>
  Epoch 50 | Loss: 0.7501 | Correct: 50 | Time: 1.46 sec<br>
  Epoch 60 | Loss: 0.3036 | Correct: 49 | Time: 1.44 sec<br>
  Epoch 70 | Loss: 0.5973 | Correct: 50 | Time: 1.51 sec<br>
  Epoch 80 | Loss: 0.6145 | Correct: 50 | Time: 2.19 sec<br>
  Epoch 90 | Loss: 1.0825 | Correct: 49 | Time: 1.52 sec<br>
  Epoch 100 | Loss: 0.6957 | Correct: 50 | Time: 1.45 sec<br>
  Epoch 110 | Loss: 0.4526 | Correct: 50 | Time: 1.52 sec<br>
  Epoch 120 | Loss: 0.1331 | Correct: 50 | Time: 1.83 sec<br>
  Epoch 130 | Loss: 0.0357 | Correct: 50 | Time: 1.43 sec<br>
  Epoch 140 | Loss: 0.6444 | Correct: 50 | Time: 1.42 sec<br>
  Epoch 150 | Loss: 0.2371 | Correct: 50 | Time: 1.45 sec<br>
  Epoch 160 | Loss: 0.8665 | Correct: 50 | Time: 1.45 sec<br>
  Epoch 170 | Loss: 0.7532 | Correct: 50 | Time: 1.48 sec<br>
  Epoch 180 | Loss: 0.4887 | Correct: 50 | Time: 1.44 sec<br>
  Epoch 190 | Loss: 0.2996 | Correct: 50 | Time: 1.42 sec<br>
  Epoch 200 | Loss: 0.0512 | Correct: 50 | Time: 1.55 sec<br>
  Epoch 210 | Loss: 0.0596 | Correct: 50 | Time: 1.85 sec<br>
  Epoch 220 | Loss: 0.5766 | Correct: 50 | Time: 1.46 sec<br>
  Epoch 230 | Loss: 0.2920 | Correct: 50 | Time: 1.44 sec<br>
  Epoch 240 | Loss: 0.5421 | Correct: 50 | Time: 1.48 sec<br>
  Epoch 250 | Loss: 0.3622 | Correct: 50 | Time: 2.18 sec<br>
  Epoch 260 | Loss: 0.0048 | Correct: 50 | Time: 1.48 sec<br>
  Epoch 270 | Loss: 0.0840 | Correct: 50 | Time: 1.43 sec<br>
  Epoch 280 | Loss: 0.4952 | Correct: 50 | Time: 1.51 sec<br>
  Epoch 290 | Loss: 0.5882 | Correct: 50 | Time: 2.09 sec<br>
  Epoch 300 | Loss: 0.0634 | Correct: 50 | Time: 1.43 sec<br>
  Epoch 310 | Loss: 0.2894 | Correct: 50 | Time: 1.45 sec<br>
  Epoch 320 | Loss: 0.3949 | Correct: 50 | Time: 1.43 sec<br>
  Epoch 330 | Loss: 0.0407 | Correct: 50 | Time: 1.76 sec<br>
  Epoch 340 | Loss: 0.0322 | Correct: 50 | Time: 1.43 sec<br>
  Epoch 350 | Loss: 0.2414 | Correct: 50 | Time: 1.45 sec<br>
  Epoch 360 | Loss: 0.2898 | Correct: 50 | Time: 1.49 sec<br>
  Epoch 370 | Loss: 0.0620 | Correct: 50 | Time: 1.73 sec<br>
  Epoch 380 | Loss: 0.4597 | Correct: 50 | Time: 1.49 sec<br>
  Epoch 390 | Loss: 0.0240 | Correct: 50 | Time: 1.46 sec<br>
  Epoch 400 | Loss: 0.0225 | Correct: 50 | Time: 1.51 sec<br>
  Epoch 410 | Loss: 0.1953 | Correct: 50 | Time: 1.42 sec<br>
  Epoch 420 | Loss: 0.2277 | Correct: 50 | Time: 1.60 sec<br>
  Epoch 430 | Loss: 0.1573 | Correct: 50 | Time: 1.41 sec<br>
  Epoch 440 | Loss: 0.0389 | Correct: 50 | Time: 1.47 sec<br>
  Epoch 450 | Loss: 0.3076 | Correct: 50 | Time: 1.41 sec<br>
  Epoch 460 | Loss: 0.0939 | Correct: 50 | Time: 1.91 sec<br>
  Epoch 470 | Loss: 0.3078 | Correct: 50 | Time: 1.51 sec<br>
  Epoch 480 | Loss: 0.0087 | Correct: 50 | Time: 1.40 sec<br>
  Epoch 490 | Loss: 0.1523 | Correct: 50 | Time: 1.42 sec<br>

## Split dataset

- CPU

HIDDEN == 100 <br><br>
RATE == 0.05 <br><br>

Epoch 0 | Loss: 6.8016 | Correct: 35 | Time: 19.71 sec<br>
Epoch 10 | Loss: 5.2064 | Correct: 38 | Time: 0.13 sec<br>
Epoch 20 | Loss: 5.0360 | Correct: 44 | Time: 0.14 sec<br>
Epoch 30 | Loss: 3.7920 | Correct: 46 | Time: 0.14 sec<br>
Epoch 40 | Loss: 3.9430 | Correct: 48 | Time: 0.27 sec<br>
Epoch 50 | Loss: 2.9389 | Correct: 50 | Time: 0.13 sec<br>
Epoch 60 | Loss: 2.6814 | Correct: 49 | Time: 0.13 sec<br>
Epoch 70 | Loss: 1.9525 | Correct: 50 | Time: 0.13 sec<br>
Epoch 80 | Loss: 1.6827 | Correct: 50 | Time: 0.13 sec<br>
Epoch 90 | Loss: 1.3845 | Correct: 50 | Time: 0.13 sec<br>
Epoch 100 | Loss: 1.3409 | Correct: 50 | Time: 0.13 sec<br>
Epoch 110 | Loss: 0.9326 | Correct: 50 | Time: 0.13 sec<br>
Epoch 120 | Loss: 0.4066 | Correct: 50 | Time: 0.13 sec<br>
Epoch 130 | Loss: 0.6358 | Correct: 50 | Time: 0.28 sec<br>
Epoch 140 | Loss: 0.6801 | Correct: 50 | Time: 0.14 sec<br>
Epoch 150 | Loss: 0.7065 | Correct: 50 | Time: 0.13 sec<br>
Epoch 160 | Loss: 0.7532 | Correct: 50 | Time: 0.14 sec<br>
Epoch 170 | Loss: 0.7671 | Correct: 50 | Time: 0.13 sec<br>
Epoch 180 | Loss: 0.3284 | Correct: 50 | Time: 0.13 sec<br>
Epoch 190 | Loss: 0.4138 | Correct: 50 | Time: 0.13 sec<br>
Epoch 200 | Loss: 0.6483 | Correct: 50 | Time: 0.13 sec<br>
Epoch 210 | Loss: 0.4617 | Correct: 50 | Time: 0.17 sec<br>
Epoch 220 | Loss: 0.2393 | Correct: 50 | Time: 0.23 sec<br>
Epoch 230 | Loss: 0.1704 | Correct: 50 | Time: 0.13 sec<br>
Epoch 240 | Loss: 0.1646 | Correct: 50 | Time: 0.14 sec<br>
Epoch 250 | Loss: 0.3689 | Correct: 50 | Time: 0.13 sec<br>
Epoch 260 | Loss: 0.4578 | Correct: 50 | Time: 0.14 sec<br>
Epoch 270 | Loss: 0.3907 | Correct: 50 | Time: 0.15 sec<br>
Epoch 280 | Loss: 0.1592 | Correct: 50 | Time: 0.13 sec<br>
Epoch 290 | Loss: 0.2722 | Correct: 50 | Time: 0.13 sec<br>
Epoch 300 | Loss: 0.1544 | Correct: 50 | Time: 0.22 sec<br>
Epoch 310 | Loss: 0.2945 | Correct: 50 | Time: 0.29 sec<br>
Epoch 320 | Loss: 0.2140 | Correct: 50 | Time: 0.13 sec<br>
Epoch 330 | Loss: 0.3505 | Correct: 50 | Time: 0.13 sec<br>
Epoch 340 | Loss: 0.1493 | Correct: 50 | Time: 0.14 sec<br>
Epoch 350 | Loss: 0.1814 | Correct: 50 | Time: 0.13 sec<br>
Epoch 360 | Loss: 0.1183 | Correct: 50 | Time: 0.13 sec<br>
Epoch 370 | Loss: 0.0267 | Correct: 50 | Time: 0.13 sec<br>
Epoch 380 | Loss: 0.3471 | Correct: 50 | Time: 0.13 sec<br>
Epoch 390 | Loss: 0.1655 | Correct: 50 | Time: 0.30 sec<br>
Epoch 400 | Loss: 0.1912 | Correct: 50 | Time: 0.13 sec<br>
Epoch 410 | Loss: 0.1385 | Correct: 50 | Time: 0.13 sec<br>
Epoch 420 | Loss: 0.2628 | Correct: 50 | Time: 0.14 sec<br>
Epoch 430 | Loss: 0.0951 | Correct: 50 | Time: 0.13 sec<br>
Epoch 440 | Loss: 0.1060 | Correct: 50 | Time: 0.13 sec<br>
Epoch 450 | Loss: 0.1136 | Correct: 50 | Time: 0.14 sec<br>
Epoch 460 | Loss: 0.1005 | Correct: 50 | Time: 0.13 sec<br>
Epoch 470 | Loss: 0.1705 | Correct: 50 | Time: 0.13 sec<br>
Epoch 480 | Loss: 0.1153 | Correct: 50 | Time: 0.23 sec<br>
Epoch 490 | Loss: 0.1102 | Correct: 50 | Time: 0.13 sec<br>

- GPU
  HIDDEN == 100 <br><br>
  RATE == 0.05 <br><br>

Epoch 0 | Loss: 10.8887 | Correct: 20 | Time: 4.45 sec<br>
Epoch 10 | Loss: 4.1624 | Correct: 39 | Time: 1.43 sec<br>
Epoch 20 | Loss: 6.4828 | Correct: 43 | Time: 1.96 sec<br>
Epoch 30 | Loss: 3.0198 | Correct: 48 | Time: 1.45 sec<br>
Epoch 40 | Loss: 2.6152 | Correct: 48 | Time: 1.45 sec<br>
Epoch 50 | Loss: 2.3295 | Correct: 49 | Time: 1.53 sec<br>
Epoch 60 | Loss: 2.1601 | Correct: 48 | Time: 2.17 sec<br>
Epoch 70 | Loss: 1.1260 | Correct: 49 | Time: 1.54 sec<br>
Epoch 80 | Loss: 0.7908 | Correct: 49 | Time: 1.43 sec<br>
Epoch 90 | Loss: 1.4306 | Correct: 49 | Time: 1.52 sec<br>
Epoch 100 | Loss: 2.8631 | Correct: 49 | Time: 1.70 sec<br>
Epoch 110 | Loss: 1.2839 | Correct: 48 | Time: 1.52 sec<br>
Epoch 120 | Loss: 0.8533 | Correct: 49 | Time: 1.44 sec<br>
Epoch 130 | Loss: 0.6947 | Correct: 49 | Time: 1.47 sec<br>
Epoch 140 | Loss: 1.1758 | Correct: 49 | Time: 1.45 sec<br>
Epoch 150 | Loss: 1.1748 | Correct: 49 | Time: 1.64 sec<br>
Epoch 160 | Loss: 0.6005 | Correct: 49 | Time: 1.46 sec<br>
Epoch 170 | Loss: 0.3681 | Correct: 49 | Time: 1.46 sec<br>
Epoch 180 | Loss: 1.2455 | Correct: 50 | Time: 1.46 sec<br>
Epoch 190 | Loss: 0.8355 | Correct: 49 | Time: 2.04 sec<br>
Epoch 200 | Loss: 0.4454 | Correct: 49 | Time: 1.56 sec<br>
Epoch 210 | Loss: 1.5734 | Correct: 49 | Time: 1.44 sec<br>
Epoch 220 | Loss: 0.3727 | Correct: 49 | Time: 1.44 sec<br>
Epoch 230 | Loss: 0.9912 | Correct: 49 | Time: 2.07 sec<br>
Epoch 240 | Loss: 0.4816 | Correct: 49 | Time: 1.44 sec<br>
Epoch 250 | Loss: 0.4841 | Correct: 50 | Time: 1.45 sec<br>
Epoch 260 | Loss: 1.9721 | Correct: 50 | Time: 1.44 sec<br>
Epoch 270 | Loss: 0.1944 | Correct: 50 | Time: 1.58 sec<br>
Epoch 280 | Loss: 0.9244 | Correct: 49 | Time: 1.52 sec<br>
Epoch 290 | Loss: 0.1379 | Correct: 49 | Time: 1.50 sec<br>
Epoch 300 | Loss: 0.3325 | Correct: 49 | Time: 1.46 sec<br>
Epoch 310 | Loss: 0.3971 | Correct: 49 | Time: 1.43 sec<br>
Epoch 320 | Loss: 0.5060 | Correct: 50 | Time: 2.09 sec<br>
Epoch 330 | Loss: 0.1879 | Correct: 49 | Time: 1.43 sec<br>
Epoch 340 | Loss: 1.3429 | Correct: 50 | Time: 1.46 sec<br>
Epoch 350 | Loss: 1.4333 | Correct: 50 | Time: 1.44 sec<br>
Epoch 360 | Loss: 0.0717 | Correct: 49 | Time: 1.78 sec<br>
Epoch 370 | Loss: 0.1294 | Correct: 49 | Time: 1.44 sec<br>
Epoch 380 | Loss: 0.5242 | Correct: 49 | Time: 1.52 sec<br>
Epoch 390 | Loss: 0.7630 | Correct: 50 | Time: 1.46 sec<br>
Epoch 400 | Loss: 0.2821 | Correct: 49 | Time: 1.47 sec<br>
Epoch 410 | Loss: 1.7085 | Correct: 50 | Time: 1.95 sec<br>
Epoch 420 | Loss: 0.8861 | Correct: 49 | Time: 1.48 sec<br>
Epoch 430 | Loss: 0.2607 | Correct: 49 | Time: 1.42 sec<br>
Epoch 440 | Loss: 0.4070 | Correct: 49 | Time: 1.49 sec<br>
Epoch 450 | Loss: 1.1194 | Correct: 50 | Time: 2.10 sec<br>
Epoch 460 | Loss: 0.0665 | Correct: 49 | Time: 1.46 sec<br>
Epoch 470 | Loss: 1.1530 | Correct: 50 | Time: 1.49 sec<br>
Epoch 480 | Loss: 0.1312 | Correct: 50 | Time: 1.45 sec<br>
Epoch 490 | Loss: 0.3074 | Correct: 49 | Time: 1.91 sec<br>

## XOR Dataset

- GPU
  Epoch 0 | Loss: 8.5838 | Correct: 25 | Time: 5.85 sec<br>
  Epoch 10 | Loss: 7.1221 | Correct: 41 | Time: 1.48 sec<br>
  Epoch 20 | Loss: 6.2308 | Correct: 45 | Time: 1.46 sec<br>
  Epoch 30 | Loss: 4.7234 | Correct: 43 | Time: 1.48 sec<br>
  Epoch 40 | Loss: 4.4958 | Correct: 45 | Time: 1.78 sec<br>
  Epoch 50 | Loss: 4.1151 | Correct: 45 | Time: 1.45 sec<br>
  Epoch 60 | Loss: 1.3272 | Correct: 45 | Time: 1.48 sec<br>
  Epoch 70 | Loss: 1.5607 | Correct: 45 | Time: 1.51 sec<br>
  Epoch 80 | Loss: 1.0135 | Correct: 46 | Time: 1.49 sec<br>
  Epoch 90 | Loss: 1.0689 | Correct: 45 | Time: 1.81 sec<br>
  Epoch 100 | Loss: 1.9355 | Correct: 46 | Time: 1.48 sec<br>
  Epoch 110 | Loss: 2.5595 | Correct: 45 | Time: 1.53 sec<br>
  Epoch 120 | Loss: 2.6738 | Correct: 46 | Time: 1.46 sec<br>
  Epoch 130 | Loss: 3.0706 | Correct: 47 | Time: 2.02 sec<br>
  Epoch 140 | Loss: 0.5645 | Correct: 46 | Time: 1.46 sec<br>
  Epoch 150 | Loss: 1.4982 | Correct: 47 | Time: 1.47 sec<br>
  Epoch 160 | Loss: 0.8734 | Correct: 46 | Time: 1.46 sec<br>
  Epoch 170 | Loss: 2.3607 | Correct: 47 | Time: 2.16 sec<br>
  Epoch 180 | Loss: 1.4214 | Correct: 48 | Time: 1.45 sec<br>
  Epoch 190 | Loss: 0.4451 | Correct: 48 | Time: 1.56 sec<br>
  Epoch 200 | Loss: 4.3863 | Correct: 48 | Time: 1.52 sec<br>
  Epoch 210 | Loss: 1.7514 | Correct: 47 | Time: 2.21 sec<br>
  Epoch 220 | Loss: 0.7269 | Correct: 49 | Time: 1.45 sec<br>
  Epoch 230 | Loss: 0.5026 | Correct: 47 | Time: 1.45 sec<br>
  Epoch 240 | Loss: 0.7092 | Correct: 48 | Time: 1.46 sec<br>
  Epoch 250 | Loss: 2.2023 | Correct: 48 | Time: 1.90 sec<br>
  Epoch 260 | Loss: 0.2580 | Correct: 49 | Time: 1.50 sec<br>
  Epoch 270 | Loss: 1.6708 | Correct: 49 | Time: 1.44 sec<br>
  Epoch 280 | Loss: 0.4718 | Correct: 48 | Time: 1.47 sec<br>
  Epoch 290 | Loss: 2.5725 | Correct: 48 | Time: 1.71 sec<br>
  Epoch 300 | Loss: 1.1565 | Correct: 48 | Time: 1.47 sec<br>
  Epoch 310 | Loss: 0.9243 | Correct: 48 | Time: 1.49 sec<br>
  Epoch 320 | Loss: 0.2484 | Correct: 49 | Time: 1.46 sec<br>
  Epoch 330 | Loss: 0.7053 | Correct: 50 | Time: 1.47 sec<br>
  Epoch 340 | Loss: 3.3578 | Correct: 47 | Time: 1.87 sec<br>
  Epoch 350 | Loss: 1.3912 | Correct: 49 | Time: 1.47 sec<br>
  Epoch 360 | Loss: 2.6955 | Correct: 48 | Time: 1.52 sec<br>
  Epoch 370 | Loss: 1.3114 | Correct: 49 | Time: 1.46 sec<br>
  Epoch 380 | Loss: 0.4629 | Correct: 49 | Time: 2.28 sec<br>
  Epoch 390 | Loss: 2.4257 | Correct: 50 | Time: 1.53 sec<br>
  Epoch 400 | Loss: 0.9487 | Correct: 49 | Time: 1.52 sec<br>
  Epoch 410 | Loss: 2.0202 | Correct: 50 | Time: 1.45 sec<br>
  Epoch 420 | Loss: 1.1797 | Correct: 49 | Time: 2.21 sec<br>
  Epoch 430 | Loss: 0.1227 | Correct: 49 | Time: 1.47 sec<br>
  Epoch 440 | Loss: 1.4504 | Correct: 50 | Time: 1.47 sec<br>
  Epoch 450 | Loss: 0.1671 | Correct: 50 | Time: 1.44 sec<br>
  Epoch 460 | Loss: 1.7854 | Correct: 50 | Time: 1.99 sec<br>
  Epoch 470 | Loss: 0.4012 | Correct: 49 | Time: 1.51 sec<br>
  Epoch 480 | Loss: 0.2509 | Correct: 50 | Time: 1.45 sec<br>
  Epoch 490 | Loss: 1.8845 | Correct: 49 | Time: 1.47 sec<br>

- CPU
  Epoch 0 | Loss: 7.3402 | Correct: 25 | Time: 18.91 sec<br>
  Epoch 10 | Loss: 5.6086 | Correct: 37 | Time: 0.29 sec<br>
  Epoch 20 | Loss: 5.4300 | Correct: 41 | Time: 0.13 sec<br>
  Epoch 30 | Loss: 2.7351 | Correct: 42 | Time: 0.13 sec<br>
  Epoch 40 | Loss: 4.2882 | Correct: 42 | Time: 0.13 sec<br>
  Epoch 50 | Loss: 2.8625 | Correct: 42 | Time: 0.15 sec<br>
  Epoch 60 | Loss: 4.0159 | Correct: 47 | Time: 0.13 sec<br>
  Epoch 70 | Loss: 3.7413 | Correct: 45 | Time: 0.13 sec<br>
  Epoch 80 | Loss: 3.9130 | Correct: 46 | Time: 0.13 sec<br>
  Epoch 90 | Loss: 2.2878 | Correct: 46 | Time: 0.16 sec<br>
  Epoch 100 | Loss: 3.8385 | Correct: 48 | Time: 0.32 sec<br>
  Epoch 110 | Loss: 2.3681 | Correct: 49 | Time: 0.14 sec<br>
  Epoch 120 | Loss: 2.0572 | Correct: 49 | Time: 0.14 sec<br>
  Epoch 130 | Loss: 2.8192 | Correct: 49 | Time: 0.13 sec<br>
  Epoch 140 | Loss: 2.1809 | Correct: 48 | Time: 0.13 sec<br>
  Epoch 150 | Loss: 2.4874 | Correct: 49 | Time: 0.15 sec<br>
  Epoch 160 | Loss: 1.7389 | Correct: 49 | Time: 0.13 sec<br>
  Epoch 170 | Loss: 1.2300 | Correct: 47 | Time: 0.13 sec<br>
  Epoch 180 | Loss: 1.7704 | Correct: 48 | Time: 0.34 sec<br>
  Epoch 190 | Loss: 1.2743 | Correct: 49 | Time: 0.14 sec<br>
  Epoch 200 | Loss: 0.6792 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 210 | Loss: 0.5333 | Correct: 50 | Time: 0.15 sec<br>
  Epoch 220 | Loss: 1.1398 | Correct: 50 | Time: 0.14 sec<br>
  Epoch 230 | Loss: 1.5905 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 240 | Loss: 0.7053 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 250 | Loss: 1.1877 | Correct: 50 | Time: 0.14 sec<br>
  Epoch 260 | Loss: 1.5629 | Correct: 49 | Time: 0.14 sec<br>
  Epoch 270 | Loss: 0.5012 | Correct: 49 | Time: 0.27 sec<br>
  Epoch 280 | Loss: 0.1489 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 290 | Loss: 1.3987 | Correct: 49 | Time: 0.14 sec<br>
  Epoch 300 | Loss: 1.8815 | Correct: 47 | Time: 0.13 sec<br>
  Epoch 310 | Loss: 0.9705 | Correct: 50 | Time: 0.14 sec<br>
  Epoch 320 | Loss: 1.4999 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 330 | Loss: 0.4168 | Correct: 50 | Time: 0.14 sec<br>
  Epoch 340 | Loss: 1.4449 | Correct: 48 | Time: 0.13 sec<br>
  Epoch 350 | Loss: 0.1865 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 360 | Loss: 0.3227 | Correct: 50 | Time: 0.24 sec<br>
  Epoch 370 | Loss: 0.2509 | Correct: 49 | Time: 0.16 sec<br>
  Epoch 380 | Loss: 0.9820 | Correct: 49 | Time: 0.13 sec<br>
  Epoch 390 | Loss: 0.9039 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 400 | Loss: 1.0794 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 410 | Loss: 0.8407 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 420 | Loss: 0.6558 | Correct: 50 | Time: 0.14 sec<br>
  Epoch 430 | Loss: 1.2168 | Correct: 50 | Time: 0.14 sec<br>
  Epoch 440 | Loss: 0.5782 | Correct: 50 | Time: 0.29 sec<br>
  Epoch 450 | Loss: 0.8412 | Correct: 50 | Time: 0.13 sec<br>
  Epoch 460 | Loss: 0.2165 | Correct: 49 | Time: 0.13 sec<br>
  Epoch 470 | Loss: 0.4062 | Correct: 50 | Time: 0.14 sec<br>
  Epoch 480 | Loss: 0.2840 | Correct: 50 | Time: 0.14 sec<br>
  Epoch 490 | Loss: 0.9179 | Correct: 50 | Time: 0.13 sec<br>

# Bigger Model for Split dataset

- CPU
  HIDDEN == 200 <br><br>
  RATE == 0.05 <br><br>
  Epoch 0 | Loss: 24.1892 | Correct: 21 | Time: 19.61 sec<br>
  Epoch 10 | Loss: 6.1946 | Correct: 40 | Time: 0.34 sec<br>
  Epoch 20 | Loss: 2.7527 | Correct: 44 | Time: 0.34 sec<br>
  Epoch 30 | Loss: 1.5211 | Correct: 48 | Time: 0.33 sec<br>
  Epoch 40 | Loss: 2.0418 | Correct: 48 | Time: 0.32 sec<br>
  Epoch 50 | Loss: 1.1405 | Correct: 48 | Time: 0.33 sec<br>
  Epoch 60 | Loss: 0.7138 | Correct: 50 | Time: 0.33 sec<br>
  Epoch 70 | Loss: 1.0284 | Correct: 50 | Time: 0.69 sec<br>
  Epoch 80 | Loss: 1.1244 | Correct: 50 | Time: 0.32 sec<br>
  Epoch 90 | Loss: 1.7448 | Correct: 49 | Time: 0.33 sec<br>
  Epoch 100 | Loss: 0.5788 | Correct: 50 | Time: 0.34 sec<br>
  Epoch 110 | Loss: 0.8391 | Correct: 49 | Time: 0.33 sec<br>
  Epoch 120 | Loss: 0.3793 | Correct: 50 | Time: 0.33 sec<br>
  Epoch 130 | Loss: 1.0211 | Correct: 50 | Time: 0.33 sec<br>
  Epoch 140 | Loss: 0.3913 | Correct: 50 | Time: 0.71 sec<br>
  Epoch 150 | Loss: 0.7549 | Correct: 50 | Time: 0.32 sec<br>
  Epoch 160 | Loss: 0.2933 | Correct: 50 | Time: 0.33 sec<br>
  Epoch 170 | Loss: 0.3597 | Correct: 50 | Time: 0.34 sec<br>
  Epoch 180 | Loss: 0.4580 | Correct: 50 | Time: 0.33 sec<br>
  Epoch 190 | Loss: 0.0964 | Correct: 50 | Time: 0.33 sec<br>
  Epoch 200 | Loss: 0.2084 | Correct: 50 | Time: 0.33 sec<br>
  Epoch 210 | Loss: 0.2239 | Correct: 50 | Time: 0.70 sec<br>
  Epoch 220 | Loss: 0.4423 | Correct: 50 | Time: 0.33 sec<br>
  Epoch 230 | Loss: 0.1942 | Correct: 50 | Time: 0.33 sec<br>
  Epoch 240 | Loss: 0.3863 | Correct: 50 | Time: 0.34 sec<br>
  Epoch 250 | Loss: 0.0182 | Correct: 50 | Time: 0.36 sec<br>
  Epoch 260 | Loss: 0.0809 | Correct: 50 | Time: 0.34 sec<br>
  Epoch 270 | Loss: 0.1195 | Correct: 50 | Time: 0.33 sec<br>
  Epoch 280 | Loss: 0.2938 | Correct: 50 | Time: 0.72 sec<br>
  Epoch 290 | Loss: 0.0351 | Correct: 50 | Time: 0.34 sec<br>
  Epoch 300 | Loss: 0.1565 | Correct: 50 | Time: 0.34 sec<br>
  Epoch 310 | Loss: 0.0375 | Correct: 50 | Time: 0.35 sec<br>
  Epoch 320 | Loss: 0.0427 | Correct: 50 | Time: 0.32 sec<br>
  Epoch 330 | Loss: 0.0960 | Correct: 50 | Time: 0.33 sec<br>
  Epoch 340 | Loss: 0.3373 | Correct: 50 | Time: 0.33 sec<br>
  Epoch 350 | Loss: 0.0777 | Correct: 50 | Time: 0.77 sec<br>
  Epoch 360 | Loss: 0.3055 | Correct: 50 | Time: 0.32 sec<br>
  Epoch 370 | Loss: 0.1224 | Correct: 50 | Time: 0.34 sec<br>
  Epoch 380 | Loss: 0.0193 | Correct: 50 | Time: 0.34 sec<br>
  Epoch 390 | Loss: 0.0956 | Correct: 50 | Time: 0.34 sec<br>
  Epoch 400 | Loss: 0.1828 | Correct: 50 | Time: 0.32 sec<br>
  Epoch 410 | Loss: 0.3179 | Correct: 50 | Time: 0.32 sec<br>
  Epoch 420 | Loss: 0.0441 | Correct: 50 | Time: 0.75 sec<br>
  Epoch 430 | Loss: 0.1316 | Correct: 50 | Time: 0.33 sec<br>
  Epoch 440 | Loss: 0.1404 | Correct: 50 | Time: 0.33 sec<br>
  Epoch 450 | Loss: 0.2632 | Correct: 50 | Time: 0.34 sec<br>
  Epoch 460 | Loss: 0.2942 | Correct: 50 | Time: 0.33 sec<br>
  Epoch 470 | Loss: 0.1096 | Correct: 50 | Time: 0.32 sec<br>
  Epoch 480 | Loss: 0.1357 | Correct: 50 | Time: 0.32 sec<br>
  Epoch 490 | Loss: 0.1226 | Correct: 50 | Time: 0.70 sec<br>

- GPU
  HIDDEN == 200 <br><br>
  RATE == 0.05 <br><br>
  Epoch 0 | Loss: 12.7801 | Correct: 22 | Time: 4.28 sec<br>
  Epoch 10 | Loss: 3.3068 | Correct: 49 | Time: 1.57 sec<br>
  Epoch 20 | Loss: 2.6589 | Correct: 45 | Time: 2.32 sec<br>
  Epoch 30 | Loss: 1.0211 | Correct: 47 | Time: 1.53 sec<br>
  Epoch 40 | Loss: 1.6052 | Correct: 50 | Time: 1.58 sec<br>
  Epoch 50 | Loss: 1.7395 | Correct: 50 | Time: 2.17 sec<br>
  Epoch 60 | Loss: 0.4578 | Correct: 50 | Time: 1.57 sec<br>
  Epoch 70 | Loss: 0.4728 | Correct: 50 | Time: 1.61 sec<br>
  Epoch 80 | Loss: 0.6145 | Correct: 50 | Time: 2.17 sec<br>
  Epoch 90 | Loss: 0.8745 | Correct: 50 | Time: 1.62 sec<br>
  Epoch 100 | Loss: 0.5269 | Correct: 50 | Time: 1.53 sec<br>
  Epoch 110 | Loss: 0.6607 | Correct: 50 | Time: 2.16 sec<br>
  Epoch 120 | Loss: 0.7763 | Correct: 49 | Time: 1.55 sec<br>
  Epoch 130 | Loss: 1.0885 | Correct: 49 | Time: 1.57 sec<br>
  Epoch 140 | Loss: 0.1877 | Correct: 50 | Time: 2.17 sec<br>
  Epoch 150 | Loss: 0.2080 | Correct: 50 | Time: 1.55 sec<br>
  Epoch 160 | Loss: 0.4236 | Correct: 50 | Time: 1.54 sec<br>
  Epoch 170 | Loss: 0.7901 | Correct: 50 | Time: 2.21 sec<br>
  Epoch 180 | Loss: 0.0936 | Correct: 50 | Time: 1.54 sec<br>
  Epoch 190 | Loss: 0.2425 | Correct: 50 | Time: 1.54 sec<br>
  Epoch 200 | Loss: 0.4564 | Correct: 50 | Time: 2.37 sec<br>
  Epoch 210 | Loss: 0.2029 | Correct: 50 | Time: 1.56 sec<br>
  Epoch 220 | Loss: 0.1738 | Correct: 50 | Time: 1.57 sec<br>
  Epoch 230 | Loss: 0.1653 | Correct: 50 | Time: 2.33 sec<br>
  Epoch 240 | Loss: 0.4167 | Correct: 50 | Time: 1.55 sec<br>
  Epoch 250 | Loss: 0.1068 | Correct: 50 | Time: 1.60 sec<br>
  Epoch 260 | Loss: 0.0700 | Correct: 50 | Time: 2.18 sec<br>
  Epoch 270 | Loss: 0.3297 | Correct: 50 | Time: 1.55 sec<br>
  Epoch 280 | Loss: 0.0456 | Correct: 50 | Time: 1.74 sec<br>
  Epoch 290 | Loss: 0.7376 | Correct: 50 | Time: 2.08 sec<br>
  Epoch 300 | Loss: 0.1007 | Correct: 50 | Time: 1.56 sec<br>
  Epoch 310 | Loss: 0.0272 | Correct: 50 | Time: 1.58 sec<br>
  Epoch 320 | Loss: 0.3432 | Correct: 50 | Time: 1.97 sec<br>
  Epoch 330 | Loss: 0.0504 | Correct: 50 | Time: 1.54 sec<br>
  Epoch 340 | Loss: 0.0677 | Correct: 50 | Time: 1.55 sec<br>
  Epoch 350 | Loss: 0.0460 | Correct: 50 | Time: 2.08 sec<br>
  Epoch 360 | Loss: 0.2471 | Correct: 50 | Time: 1.63 sec<br>
  Epoch 370 | Loss: 0.3215 | Correct: 50 | Time: 1.55 sec<br>
  Epoch 380 | Loss: 0.0830 | Correct: 50 | Time: 2.25 sec<br>
  Epoch 390 | Loss: 0.0341 | Correct: 50 | Time: 1.56 sec<br>
  Epoch 400 | Loss: 0.0714 | Correct: 50 | Time: 1.57 sec<br>
  Epoch 410 | Loss: 0.0481 | Correct: 50 | Time: 2.18 sec<br>
  Epoch 420 | Loss: 0.0502 | Correct: 50 | Time: 1.56 sec<br>
  Epoch 430 | Loss: 0.0510 | Correct: 50 | Time: 1.55 sec<br>
  Epoch 440 | Loss: 0.0662 | Correct: 50 | Time: 2.22 sec<br>
  Epoch 450 | Loss: 0.0506 | Correct: 50 | Time: 1.55 sec<br>
  Epoch 460 | Loss: 0.1670 | Correct: 50 | Time: 1.56 sec<br>
  Epoch 470 | Loss: 0.0401 | Correct: 50 | Time: 2.12 sec<br>
  Epoch 480 | Loss: 0.1236 | Correct: 50 | Time: 1.54 sec<br>
  Epoch 490 | Loss: 0.1936 | Correct: 50 | Time: 1.58 sec<br>
