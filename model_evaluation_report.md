# Model Evaluation and Note Distribution Report

This report documents the dataset's note distribution and the validation performance metrics (frame-level F1-scores, precision, recall, confusion matrices) for several trained multi-task checkpoints.

---

## 1. Dataset Note Distributions

The dataset consists of aligned beatmaps and audio spectrograms. The distribution of notes can be analyzed from two perspectives: the **raw hit objects** (as defined in the beatmaps) and the **frame-level labels** (which are temporally thickened with an `onset_width` of 3 frames for model training).

### Raw Beatmap Hit Object counts
Calculated from `.osu` files directly (physical objects placed in the map):

| Hit Object Type | Total Count | Percentage |
|---|---|---|
| **Circles** | 164,995 | 59.30% |
| **Sliders** | 112,587 | 40.47% |
| **Spinners** | 641 | 0.23% |
| **Total** | 278,223 | 100.00% |

* **Circle-to-Slider Ratio**: **1.47 : 1** (roughly 3 circles for every 2 sliders).

### Frame-Level Label Distribution (5-Class Preprocessing)
Calculated across a representative sample of 100 validation maps. Note onsets are thickened by 3 frames, creating a near 1:1 balance between "None" and "Note" frames:

| Class Label | Frame Count | Overall % | Note Frame % |
|---|---|---|---|
| **0: None** (Silence/Sustain) | 25,884 | 50.55% | — |
| **1: Circle** | 11,301 | 22.07% | 44.64% |
| **2: Slider Start** | 10,991 | 21.47% | 43.42% |
| **3: Stream** | 2,964 | 5.79% | 11.71% |
| **4: Spinner** | 60 | 0.12% | 0.24% |
| **Total Note Frames** | 25,316 | 49.45% | 100.00% |

---

## 2. Multi-Task Model Evaluations

Evaluation was performed on the validation split (**228 songs / 430,592 frames**) using the new **177-dimensional feature vectors** (which incorporate 8-dimensional difficulty and style metadata). 

We evaluated three different multi-task checkpoints corresponding to different class configurations.

### A. Checkpoint: `rhythm_model_focal.pth` (5-Class Mode)
* **Architecture**: CNN-LSTM (`cnn_channels=256`, `lstm_hidden=256`)
* **Classes**: 5 (`None`, `Circle`, `Slider Start`, `Stream`, `Spinner`)
* **Onset Class Accuracy**: **9.43%**
* **Overall Accuracy**: **58%**

#### Classification Report:

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **None** | 0.60 | 0.97 | 0.74 | 238,181 |
| **Circle** | 0.33 | 0.08 | 0.13 | 86,591 |
| **Slider Start** | 0.55 | 0.12 | 0.19 | 84,231 |
| **Stream** | 0.39 | 0.04 | 0.08 | 21,198 |
| **Spinner** | 0.00 | 0.00 | 0.00 | 391 |
| **Macro Average** | 0.37 | 0.24 | 0.23 | 430,592 |
| **Weighted Average** | 0.52 | 0.58 | 0.48 | 430,592 |

#### Confusion Matrix:
```
Predicted \ Actual    None    Circle    SliderStart    Stream    Spinner
None                 231338     3095           3383       365          0
Circle                74507     7274           4246       564          0
Slider Start          63821     9943           9939       528          0
Stream                17709     1992            568       929          0
Spinner                 325       10             52         4          0
```

---

### B. Checkpoint: `rhythm_model_focal_4.pth` (6-Class Mode)
* **Architecture**: CNN-LSTM (`cnn_channels=128`, `lstm_hidden=128`)
* **Classes**: 6 (`None`, `Circle`, `Slider Start`, `Slider End`, `Spinner`, `Stream`)
* **Onset Class Accuracy**: **10.46%**
* **Overall Accuracy**: **43%**

#### Classification Report:

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **None** | 0.45 | 0.96 | 0.61 | 164,980 |
| **Circle** | 0.32 | 0.18 | 0.23 | 86,591 |
| **Slider Start** | 0.51 | 0.14 | 0.22 | 80,954 |
| **Slider End** | 0.28 | 0.00 | 0.01 | 76,478 |
| **Spinner** | 0.00 | 0.00 | 0.00 | 391 |
| **Stream** | 0.35 | 0.05 | 0.09 | 21,198 |
| **Macro Average** | 0.32 | 0.22 | 0.19 | 430,592 |
| **Weighted Average** | 0.40 | 0.43 | 0.33 | 430,592 |

#### Confusion Matrix:
```
Predicted \ Actual    None    Circle    SliderStart    SliderEnd    Spinner    Stream
None                 159072     3575           2025          125          0       183
Circle                64767    15269           5524          155          0       876
Slider Start          48672    20124          11178          215          0       765
Slider End            67524     5964           2551          201          0       238
Spinner                 306       29             50            0          0         6
Stream                16445     2850            750           19          0      1134
```

---

### C. Checkpoint: `rhythm_model_focal_3.pth` (8-Class Mode)
* **Architecture**: CNN-LSTM (`cnn_channels=128`, `lstm_hidden=128`)
* **Classes**: 8 (`None`, `Circle`, `Slider Start`, `Slider End`, `Spinner`, `B-Stream`, `I-Stream`, `Triplet`)
* **Onset Class Accuracy**: **6.68%**
* **Overall Accuracy**: **42%**

#### Classification Report:

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **None** | 0.42 | 0.98 | 0.59 | 164,980 |
| **Circle** | 0.38 | 0.11 | 0.18 | 107,789 |
| **Slider Start** | 0.54 | 0.07 | 0.12 | 80,954 |
| **Slider End** | 0.27 | 0.00 | 0.00 | 76,478 |
| **Spinner** | 0.00 | 0.00 | 0.00 | 391 |
| **B-Stream** | 0.00 | 0.00 | 0.00 | 0 |
| **I-Stream** | 0.00 | 0.00 | 0.00 | 0 |
| **Triplet** | 0.00 | 0.00 | 0.00 | 0 |
| **Macro Average** | 0.20 | 0.15 | 0.11 | 430,592 |
| **Weighted Average** | 0.40 | 0.42 | 0.29 | 430,592 |

#### Confusion Matrix:
```
Predicted \ Actual    None    Circle    SliderStart    SliderEnd    Spinner    B-Stream
None                 162007     1978            912            3          0          80
Circle                92419    12356           2581            6          0         427
Slider Start          61218    14185           5374            2          0         175
Slider End            71704     3613           1080            4          0          77
Spinner                 339       13             35            0          0           4
```

---

## 3. Analysis Insights & Recommendations

1. **Classification Challenges**:
   * While **None** is classified with high recall (96-98%), active note types have much lower recall (8-18%). The model frequently misses the exact onset frames or misclassifies them as `None`.
   * **Slider End** is rarely predicted correctly (0.00 - 0.01 F1 score). This is likely because predicting the precise release/end of a slider is temporally ambiguous from a spectrogram compared to the attack (start).
2. **Impact of Class Count**:
   * The **5-Class model** (`rhythm_model_focal.pth`) achieves the highest overall validation accuracy (**58%**) and weighted F1-score (**0.48**), making it the most robust baseline.
   * As the number of classes increases (6-Class and 8-Class models), performance drops due to the higher granularity of predictions and class sparsity (e.g. `Stream` subdivisions).
3. **Recommendation**:
   * Using the **5-Class configuration** is recommended for general beatmap generation. Since slider endpoints can be calculated mathematically based on beat snapping and spacing (as implemented in `generate_test_map.py`), explicitly predicting `Slider End` as a separate class is unnecessary.
