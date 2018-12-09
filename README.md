# Deeper and Wider Siamese Networks for Real-Time Visual Tracking
This repo provides the source code and models of our deeper and wider siamese trackers. For CVPR2019 blind review.

## Introduction
[SiamFC]() formulates the task of visual tracking as classification between background and target. [SiamRPN]() improve SiamFC through introducing the robust region proposal estimation. However, the backbone networks used in these trackers are relatively shallow, both [AlexNet](), which does not fully take advantage of the capability of modern deep neural networks. 
  
Our proposals improve the performances of fully convolutional siamese trackers by,
1) introducing CIR and CIR-D units to unveil the power of deeper and wider networks like [ResNet]() and [Inceptipon](), 
2) designing a reasonable backbone guilded by the analysis of how internal network factors (eg. receptive field, stride, output  feature size) influence tracking performance.

## Result snapshots
<!-- <div align=center> <iframe width="560" height="315" src="https://www.youtube.com/embed/hp0idRAF0m0" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div> -->

<div align=center>
<img src="https://i.postimg.cc/6QrkG5sj/results.png" width="75%" height="75%" />
</div>
<!-- [![results.png](https://i.postimg.cc/6QrkG5sj/results.png)](https://postimg.cc/zLftsNGK) -->

## Requirements
Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz GPU: NVIDIA GTX1080
- python3.6
- pytorch == 0.3.1
- numpy == 1.12.1
- opencv == 3.1.0


**Important!!** Some opencv versions are relatively slow for some reason, so we suggest you install packages that are recommend above.

## Tracking  
We provide pretrained model of CIResNet-22 based SiamFC+  for reproducing.
- **Tracking on a specific video**
  - **step1:** Place video frames and bounding box file in `dataset` directory. We provide `David3` video in this demo.
  - **step2:** run,
```bash
python run_tracker.py --arch SiamFC_plus --resume ./pretrain/CIResNet22.pth --video David3 --vis True
```
`--arch` `--resume` `video` `vis` indicate model name, pretrained model path, video name and visualization flag respectively.

- **Testing on VOT-2017 benchmark**
   - **step1:** place VOT2017 benchmark files in `dataset` directory. The `dataset` directory should be organized as (same for OTB benchmark), <br/>
   |-- dataset  <br/>
   &emsp; |-- VOT2017 <br/>
   &emsp; &emsp; |-- ants1  <br/>
    &emsp; &emsp; |-- ... <br/>

  

   - **step2:** run,

  ```bash
    sh run_benchmark.sh
  ```


    Screen snapshot shows like this (about 70 fps), and the results will be saved in `test` directory.
    
    <div style="align: center"> <img src="https://i.postimg.cc/2yfTdzCy/screen-snaps2.png"/> 

  - **step3 (optional):** The results in `test` directory can be evaluated in vot-toolkit directly. Please refer to offical [vot-toolkit](http://votchallenge.net/howto/integration.html) for more information.

- **Testing on OTB-2013 benchmark**
   - **step1:** Place OTB2013 benchmark files in `dataset` directory.
   - **step2:** Modify `--dataset VOT2017` in `run_benchmark.sh` to `--dataset OTB2013`
   - **setp3:** run,
  ```bash
    sh run_benchmark.sh
  ```
  - **step4 (optional):** The results in `test` directory can be evaluated in otb-toolkit.

- **Testing on other benchmarks**  

  If you want to test on other benchmarks, modify the code to your needs. Object tracking algorithms are sensitive to hyperparameters, so careful fine-tuneing on different benchmarks is necessary.


## License
Licensed under an MIT license.




