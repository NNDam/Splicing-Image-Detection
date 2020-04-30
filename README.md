## Splicing-Image-Detection
Paper: Splicing image forgery detection based on DCT and Local Binary Pattern, 2013 IEEE Global Conference on Signal and Information Processing
<p align="center">
  <img src=/figure/figure.jpg>
</p>

## Requirements
- opencv-python
- sklearn
- skimage
- progressbar2
## Dataset
- CASIA v1: [Gdrive](https://drive.google.com/file/d/14f3jU2VsxTYopgSE1Vvv4hMlFvpAKHUY/view)
- CASIA v2: [Gdrive](https://drive.google.com/file/d/1KvF7EF-rLD2e5AujOzOifBvo5dFTZjbn/view)
## Accuracy
| Dataset       | Cross Validation|
| ------------- |:-------------:|
| CASIA v1      | 98.0 +- 0.01     |
| CASIA v2      | 96.0 +- 0.01      |
## Note
- Change value of block_sizes and strides for feature extraction in multiple case
- Example block_sizes = [16, 32] and strides = [8, 16]
