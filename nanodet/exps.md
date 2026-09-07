优化记录:
- **multiscale** ×
- optime **TAl** √
- **dfl down** √：0.25为步长，0.5处最优
- boxloss down ？：实验不充分

### official
~~~bash
YOLOv8 summary (fused): 73 layers, 3,006,428 parameters, 0 gradients, 8.1 GFLOPs
    Class     Box(P          R      mAP50  mAP50-95): 
    all       0.843      0.873      0.906      0.651
    fish2     0.688      0.879      0.869      0.506
    zl        0.799      0.718      0.819      0.482
    yq        0.891      0.894      0.943       0.64
    hdy       0.995          1      0.995      0.976
Speed: 0.1ms preprocess, 0.5ms inference, 0.0ms loss, 0.8ms postprocess per image
~~~
### self
- Classes: 4 | Params: 3,130,644 | Gradients: 3,130,644 | GFLOPs@640: 8.22

TAL_old-soft_label-55_25_05
~~~bash
Per-class metrics for best.pt (99/100):
           class        P        R    mAP50   mAP50-95    valid
             all   0.8937   0.8969   0.9267     0.6765     True
           fish2   0.8263   0.8229   0.8857     0.5243     True
              zl   0.8521   0.8233   0.8636     0.5178     True
              yq   0.8973   0.9412   0.9626     0.6778     True
             hdy   0.9992   1.0000   0.9950     0.9863     True
~~~

正確：bs48---TAL_old-soft_label-55_20_05
~~~bash
           class        P        R    mAP50   mAP50-95    valid
             all   0.8959   0.8983   0.9328     0.6885     True
           fish2   0.8273   0.8183   0.8897     0.5343     True
              zl   0.8581   0.8414   0.8848     0.5403     True
              yq   0.8989   0.9336   0.9619     0.6899     True
             hdy   0.9992   1.0000   0.9950     0.9894     True
~~~

錯誤：bs48---optime TAl---boxloss down(45)---dfl（0.5） down（可）

正確：bs48---optime TAl---boxloss down(可)---dfl（0.5） down（可）
~~~bash
Per-class metrics for best.pt (100/100):
           class        P        R    mAP50   mAP50-95    valid
             all   0.9061   0.8902   0.9263     0.6990     True
           fish2   0.8457   0.7994   0.8881     0.5522     True
              zl   0.8494   0.8269   0.8556     0.5341     True
              yq   0.9301   0.9346   0.9664     0.7172     True
             hdy   0.9992   1.0000   0.9950     0.9923     True
saved curves to runs\yolov8n_parammatch_cy_optim-tal_boxloss-down_dfl-down3
~~~

bs48---optime TAl---boxloss down(可)---dfl（0.75） down（可）

bs48---optime TAl---boxloss down(可)---dfl（1.0） down（可）

bs48---optime TAl---boxloss down(可)

bs48---optime TAl(可)

bs48---optime TAl(可)---boxloss down(可)---loss*bs



