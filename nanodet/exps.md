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
錯誤：bs48---optime TAl---boxloss down(45)---dfl（0.5） down（可）
~~~bash
           class        P        R    mAP50   mAP50-95    valid
             all   0.9128   0.8750   0.9246     0.6977     True
           fish2   0.8579   0.7695   0.8867     0.5419     True
              zl   0.8656   0.8008   0.8504     0.5447     True
              yq   0.9284   0.9296   0.9662     0.7125     Tru      
~~~
錯誤：bs48---optime TAl---boxloss down(可)---dfl（0.25） down（不可）
~~~bash
Per-class metrics for best.pt (100/100):
           class        P        R    mAP50   mAP50-95    valid
             all   0.9221   0.8747   0.9265     0.7048     True
           fish2   0.8687   0.7739   0.8912     0.5551     True
              zl   0.8818   0.7855   0.8539     0.5594     True
              yq   0.9385   0.9395   0.9661     0.7117     True
             hdy   0.9992   1.0000   0.9950     0.9931     True
saved curves to runs\yolov8n_parammatch_cy_optim-tal_boxloss-down_dfl-down3
~~~

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
~~~bash
Per-class metrics for best.pt (100/100):
           class        P        R    mAP50   mAP50-95    valid
             all   0.8999   0.8864   0.9258     0.7031     True
           fish2   0.8177   0.7842   0.8715     0.5372     True
              zl   0.8588   0.8230   0.8702     0.5547     True
              yq   0.9239   0.9383   0.9666     0.7294     True
             hdy   0.9992   1.0000   0.9950     0.9912     True
saved curves to runs\yolov8n_parammatch_cy_optim-tal_boxloss-down_dfl-down2
~~~

bs48---optime TAl---boxloss down(可)---dfl（1.0） down（可）
~~~bash
close_aug active: disabled random perspective / HSV / multi_scale at epoch 100/100
Per-class metrics for best.pt (100/100):
    class        P        R    mAP50   mAP50-95    valid
    all       0.8831   0.8625   0.9095     0.6931     True
    fish2     0.8099   0.7700   0.8607     0.5282     True
    zl        0.8145   0.7444   0.8204     0.5275     True
    yq        0.9089   0.9354   0.9618     0.7247     True
    hdy       0.9992   1.0000   0.9950     0.9919     True
saved curves to runs\yolov8n_parammatch_cy_optim-tal_boxloss-down_dfl-down
~~~

bs48---optime TAl---boxloss down(可)
~~~bash
close_aug active: disabled random perspective / HSV / multi_scale at epoch 100/100
Per-class metrics for best.pt (100/100):
           class        P        R    mAP50   mAP50-95    valid
             all   0.8806   0.8491   0.9036     0.6868     True
           fish2   0.8010   0.7575   0.8480     0.5127     True
              zl   0.8205   0.7217   0.8135     0.5222     True
              yq   0.9017   0.9173   0.9580     0.7208     True
             hdy   0.9992   1.0000   0.9950     0.9914     True
saved curves to runs\yolov8n_parammatch_cy_optim-tal_boxloss-down

Per-class metrics for best.pt (100/100):
           class        P        R    mAP50   mAP50-95    valid
             all   0.8771   0.8435   0.8956     0.6822     True
           fish2   0.8000   0.7540   0.8492     0.5150     True
              zl   0.8068   0.6955   0.7778     0.4921     True
              yq   0.9022   0.9243   0.9602     0.7301     True
             hdy   0.9992   1.0000   0.9950     0.9917     True
saved curves to runs\yolov8n_parammatch_cy_optim-tal_boxloss-downv2
~~~


bs48---optime TAl(可)
~~~bash
close_aug active: disabled random perspective / HSV / multi_scale at epoch 100/100
Per-class metrics for best.pt (100/100):
    class        P        R    mAP50   mAP50-95    valid
        all   0.8990   0.8376   0.9005     0.6928     True
    fish2     0.8296   0.7265   0.8528     0.5262     True
        zl    0.8304   0.7180   0.7983     0.5191     True
        yq    0.9367   0.9057   0.9559     0.7339     True
        hdy   0.9993   1.0000   0.9950     0.9921     True
saved curves to runs\yolov8n_parammatch_cy_optim-tal
~~~

錯誤策略
---

bs48---optime TAl(可)---boxloss down(可)---loss*bs
~~~bash
close_aug active: disabled random perspective / HSV / multi_scale at epoch 100/100
Per-class metrics for best.pt (100/100):
           class        P        R    mAP50   mAP50-95    valid
             all   0.8664   0.8527   0.8964     0.6801     True
           fish2   0.8039   0.7666   0.8501     0.5150     True
              zl   0.7558   0.7215   0.7822     0.4856     True
              yq   0.9065   0.9226   0.9582     0.7290     True
             hdy   0.9992   1.0000   0.9950     0.9908     True
saved curves to runs\yolov8n_parammatch_cy_optim-tal_boxloss-down_lossbs
~~~



bs16
~~~bash
#复现：训练16s/epcoh、验证10s/epoch
Classes: 4 | Params: 3,130,644 | Gradients: 3,130,644 | GFLOPs@640: 8.22
Per-class metrics for best.pt (100/100):
      class        P         R        mAP50    mAP50-95
        all    0.8752     0.8385    <<0.8968     0.6840>> 
      cate1    0.8022     0.7444      0.8490     0.5204 
      cate2    0.7963     0.6917      0.7859     0.5016
      cate3    0.9030     0.9179      0.9574     0.7213
      cate4    0.9992     1.0000      0.9950     0.9927
~~~