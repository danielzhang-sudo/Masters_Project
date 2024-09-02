# Masters_Project
This is the repository for the Masters Project for UCL 2023/24

To run the code, first create the following empty folders inside the 'Masters_Project' folder:

```
Masters_Project/
|- CREDIT
|  |- credit
|  |  |- ours
|  |  |- ours_100
|  |  |- ours_500
|  |  |- ours_20
|  |  |- deng_100
|  |  |- deng_500
|  |  |- deng_20
|  |- aus
|  |  |- aus_deng
|  |  |- aus_ours
|  |- german
|  |  |- german_deng
|  |  |- german_ours
```

Then execute:

```python deng_prep.py```
or
```python ours_prep.py```
or
```python german_aus.py```

to create the image datasets.

To run the model:

```python main.py --dataset credit```

To finetune the model:

```python main.py --finetune --dataset credit --encoder both --position all --dropout_rate 0.25```

You can adjust the arguments given to change the dataset or the model configurations.