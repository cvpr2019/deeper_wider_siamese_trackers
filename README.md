# Deeper and Wider Siamese Networks for Real-Time Visual Tracking
This repo provides the source code and models of our deeper and wider siamese trackers for CVPR2019 blind review.

## Introduction
[SiamFC](https://arxiv.org/abs/1606.09549) formulates the task of visual tracking as classification between background and target. [SiamRPN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf) improves SiamFC by introducing the robust region proposal estimation branch. However, the backbone network utilized in these trackers is still the classical AlexNet, which does not fully take advantage of the capability of modern deep neural networks. 
  
Our proposals improve the performances of fully convolutional siamese trackers by,
1) introducing CIR and CIR-D units to unveil the power of deeper and wider networks like [ResNet](https://arxiv.org/abs/1512.03385) and [Inceptipon](https://arxiv.org/abs/1409.4842); 
2) designing reasonable backbones that are guilded by the analysis of how internal network factors (eg. receptive field, stride, output feature size) affect tracking performances.

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


**Important!!** Since the speed of some opencv versions is relatively slow for some reason, it is recommended that you install packages above.

## Tracking  
SiamFC+ based on CIResNet-22 is provided for reproducing in this repo.
- **Tracking on a specific video**
  - **step1:** place video frames and label file in `dataset` directory. `David3` is provided in this demo.
  - **step2:** run,
```bash
python run_tracker.py --arch SiamFC_plus --resume ./pretrain/CIResNet22.pth --video David3 --vis True
```
`--arch` `--resume` `video` and `vis` indicate model name, pretrained model path, video name and visualization flag respectively.

- **Testing on VOT-2017 benchmark**
   - **step1:** place VOT2017 benchmark files in `dataset` directory. The `dataset` directory should be organized as, <br/>
   |-- dataset  <br/>
   &emsp; |-- VOT2017 <br/>
   &emsp; &emsp; |-- ants1  <br/>
    &emsp; &emsp; |-- ... <br/>

  

   - **step2:** run,

  ```bash
    sh run_benchmark.sh
  ```


    Screen snapshot shows like this (about 70 fps), and the results are saved in `test` directory.
    
    <div style="align: center"> <img src="https://i.postimg.cc/2yfTdzCy/screen-snaps2.png"/> 

  - **step3 (optional):** evaluate results with vot-toolkit. Please refer to offical [vot-toolkit](http://votchallenge.net/howto/integration.html) document for more information.

- **Testing on OTB-2013 benchmark**
   - **step1:** place OTB2013 benchmark files in `dataset` directory.
   - **step2:** modify `--dataset VOT2017` in `run_benchmark.sh` to `--dataset OTB2013`
   - **setp3:** run,
  ```bash
    sh run_benchmark.sh
  ```
  - **step4 (optional):** evaluate results with otb-toolkit.

- **Testing on other benchmarks**  

  If you want to test this demo on other benchmarks, please modify the code to your needs. Object tracking algorithms are sensitive to hyperparameters, so careful fine-tuneing for different benchmarks is necessary.


## License
Licensed under an MIT license.




