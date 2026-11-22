### Cross-event test-region results (mean ± std over 3 seeds; threshold fixed on the source event's val split; `ndvi_diff` is deterministic)

| Model | Train on | Test on | F1 | IoU | Precision | Recall | AP | Oracle F1 |
|---|---|---|---|---|---|---|---|---|
| ndvi_diff | kodagu | kodagu | 0.672 ± 0.000 | 0.506 ± 0.000 | 0.664 ± 0.000 | 0.679 ± 0.000 | 0.680 ± 0.000 | 0.673 ± 0.000 |
| ndvi_diff | kodagu | wayanad *(zero-shot)* | 0.835 ± 0.000 | 0.716 ± 0.000 | 0.963 ± 0.000 | 0.737 ± 0.000 | 0.951 ± 0.000 | 0.862 ± 0.000 |
| ndvi_diff | wayanad | kodagu *(zero-shot)* | 0.651 ± 0.000 | 0.483 ± 0.000 | 0.571 ± 0.000 | 0.759 ± 0.000 | 0.680 ± 0.000 | 0.673 ± 0.000 |
| ndvi_diff | wayanad | wayanad | 0.857 ± 0.000 | 0.750 ± 0.000 | 0.881 ± 0.000 | 0.835 ± 0.000 | 0.951 ± 0.000 | 0.862 ± 0.000 |
| ndvi_diff | both | kodagu | 0.672 ± 0.000 | 0.506 ± 0.000 | 0.664 ± 0.000 | 0.679 ± 0.000 | 0.680 ± 0.000 | 0.673 ± 0.000 |
| ndvi_diff | both | wayanad | 0.835 ± 0.000 | 0.716 ± 0.000 | 0.963 ± 0.000 | 0.737 ± 0.000 | 0.951 ± 0.000 | 0.862 ± 0.000 |
| siamese | kodagu | kodagu | 0.686 ± 0.009 | 0.522 ± 0.011 | 0.678 ± 0.013 | 0.695 ± 0.033 | 0.697 ± 0.014 | 0.689 ± 0.006 |
| siamese | kodagu | wayanad *(zero-shot)* | 0.894 ± 0.012 | 0.809 ± 0.019 | 0.904 ± 0.002 | 0.885 ± 0.022 | 0.921 ± 0.016 | 0.896 ± 0.012 |
| siamese | wayanad | kodagu *(zero-shot)* | 0.514 ± 0.069 | 0.348 ± 0.063 | 0.665 ± 0.041 | 0.428 ± 0.101 | 0.462 ± 0.108 | 0.557 ± 0.046 |
| siamese | wayanad | wayanad | 0.929 ± 0.004 | 0.867 ± 0.006 | 0.934 ± 0.006 | 0.924 ± 0.012 | 0.944 ± 0.036 | 0.932 ± 0.002 |
| siamese | both | kodagu | 0.694 ± 0.008 | 0.531 ± 0.009 | 0.682 ± 0.017 | 0.707 ± 0.007 | 0.709 ± 0.022 | 0.695 ± 0.008 |
| siamese | both | wayanad | 0.914 ± 0.003 | 0.842 ± 0.005 | 0.939 ± 0.014 | 0.891 ± 0.018 | 0.947 ± 0.018 | 0.921 ± 0.002 |
| transformer | kodagu | kodagu | 0.683 ± 0.046 | 0.519 ± 0.052 | 0.680 ± 0.075 | 0.687 ± 0.021 | 0.732 ± 0.065 | 0.685 ± 0.044 |
| transformer | kodagu | wayanad *(zero-shot)* | 0.868 ± 0.019 | 0.767 ± 0.029 | 0.944 ± 0.012 | 0.803 ± 0.033 | 0.940 ± 0.021 | 0.892 ± 0.004 |
| transformer | wayanad | kodagu *(zero-shot)* | 0.413 ± 0.133 | 0.266 ± 0.112 | 0.600 ± 0.129 | 0.326 ± 0.143 | 0.341 ± 0.159 | 0.431 ± 0.127 |
| transformer | wayanad | wayanad | 0.937 ± 0.008 | 0.882 ± 0.014 | 0.937 ± 0.002 | 0.937 ± 0.018 | 0.979 ± 0.005 | 0.939 ± 0.008 |
| transformer | both | kodagu | 0.689 ± 0.004 | 0.526 ± 0.005 | 0.701 ± 0.033 | 0.680 ± 0.031 | 0.736 ± 0.013 | 0.694 ± 0.006 |
| transformer | both | wayanad | 0.903 ± 0.016 | 0.824 ± 0.027 | 0.916 ± 0.014 | 0.891 ± 0.018 | 0.960 ± 0.009 | 0.907 ± 0.015 |

### Zero-shot transfer scored on the target's whole scene

| Model | Train on | Test on | F1 | IoU | Precision | Recall | AP |
|---|---|---|---|---|---|---|---|
| ndvi_diff | kodagu | wayanad | 0.774 ± 0.000 | 0.631 ± 0.000 | 0.831 ± 0.000 | 0.725 ± 0.000 | 0.871 ± 0.000 |
| ndvi_diff | wayanad | kodagu | 0.563 ± 0.000 | 0.392 ± 0.000 | 0.477 ± 0.000 | 0.687 ± 0.000 | 0.574 ± 0.000 |
| siamese | kodagu | wayanad | 0.766 ± 0.042 | 0.622 ± 0.055 | 0.801 ± 0.020 | 0.736 ± 0.074 | 0.793 ± 0.048 |
| siamese | wayanad | kodagu | 0.454 ± 0.035 | 0.294 ± 0.030 | 0.550 ± 0.068 | 0.397 ± 0.079 | 0.384 ± 0.093 |
| transformer | kodagu | wayanad | 0.694 ± 0.050 | 0.533 ± 0.058 | 0.857 ± 0.121 | 0.604 ± 0.129 | 0.756 ± 0.075 |
| transformer | wayanad | kodagu | 0.359 ± 0.074 | 0.221 ± 0.056 | 0.382 ± 0.051 | 0.342 ± 0.099 | 0.242 ± 0.082 |

*Oracle F1* = best F1 over all thresholds on the test region itself (an upper bound that is **not** a valid deployment number; it isolates ranking quality from threshold miscalibration).
