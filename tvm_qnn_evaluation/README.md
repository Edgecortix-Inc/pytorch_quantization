To run evalaution on 1k subset of imagenet data,
```python imagenet_test.py```

The output should be something like:

```
Model name: resnet18, per channel quantization
PyTorch accuracy: Top1 = 69.70, Top5 = 90.20
TVM accuracy: Top1 = 70.00, Top5 = 90.10
PyTorch top5 label: [101 386 385 340 351]
TVM top5 label: [101 386 385 354 340]
PyTorch top5 raw output: [17.519054 17.206213 14.703492  8.133846  8.133846]
TVM top5 raw output: [17.519054 17.206213 14.703492  8.759527  8.446687]
max abs diff: 0.6256809
mean abs_diff: 0.10198592
675 in 1000 raw outputs correct.

Model name: resnet50, per channel quantization
PyTorch accuracy: Top1 = 78.10, Top5 = 94.10
TVM accuracy: Top1 = 77.70, Top5 = 93.70
PyTorch top5 label: [101 386 385  51 354]
TVM top5 label: [386 101 385  51 354]
PyTorch top5 raw output: [16.671005  16.671005  11.497245   8.622933   6.6109157]
TVM top5 raw output: [16.096142  16.096142  10.922382   8.335503   6.3234844]
max abs diff: 0.57486343
mean abs_diff: 0.10175061
653 in 1000 raw outputs correct.

Model name: mobilenet_v2, per channel quantization
PyTorch accuracy: Top1 = 69.90, Top5 = 89.40
TVM accuracy: Top1 = 71.00, Top5 = 89.40
PyTorch top5 label: [101 386 385  51  69]
TVM top5 label: [101 386 385  51  69]
PyTorch top5 raw output: [19.986526 18.400295 16.17957  15.545077 11.420873]
TVM top5 raw output: [19.034788  17.448555  14.593337  13.641598   9.8346405]
max abs diff: 3.4897113
mean abs_diff: 1.0450097
85 in 1000 raw outputs correct.

Model name: googlenet, per channel quantization
PyTorch accuracy: Top1 = 71.80, Top5 = 90.50
TVM accuracy: Top1 = 71.60, Top5 = 90.60
PyTorch top5 label: [101 386 385  51  69]
TVM top5 label: [101 386 385  51  69]
PyTorch top5 raw output: [11.26868   10.932301  10.427733   5.045677   3.3637848]
TVM top5 raw output: [10.932301  10.595922  10.091354   4.877488   3.3637848]
max abs diff: 0.33637905
mean abs_diff: 0.07602153
572 in 1000 raw outputs correct.

Model name: mobilenet_v3, per channel quantization
PyTorch accuracy: Top1 = 63.50, Top5 = 83.60
TVM accuracy: Top1 = 62.70, Top5 = 83.20
PyTorch top5 label: [386 101 385 354  51]
TVM top5 label: [101 386 385 354  51]
PyTorch top5 raw output: [19.691273 19.012262 17.654243 12.561673 12.222169]
TVM top5 raw output: [20.370281 20.030777 18.333254 12.561673 11.882664]
max abs diff: 3.7345514
mean abs_diff: 0.7947805
129 in 1000 raw outputs correct.
```
