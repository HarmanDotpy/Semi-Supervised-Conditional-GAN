# Semi-Supervised-Conditional-GAN

For training the cGAN'sin a semi supervised manner, having a large number of unlabelled data and a very small fraction of labelled data.

```bash
mkdir ./temp_saves #for saving the results

python uda.py --unlabelled_datapath <large-unlabelled-dataset-path> --supervised_datapath <small-supervised-dataset-path> --supervised_labels <path-of-labels-of-supervised-dataset> --output_labels <path-of-labelled-image-dataset-given-as-unlabelled-datapath> --output_classifier <path-of-output-classifier-using-UDA-method>

# use saved labels of uda classifier to train GAN
python train_cgan.py --root_path_to_save <directory-to-save-results> --traindatapath <large-unlabelled-dataset-path> --trainlabelspath  <path-of-labelled-image-dataset-given-as-unlabelled-datapath> --train_or_gen train --num_epochs 100

#generate 9k images in form of npy files and save as gen9k.npy and target9k.npy
python train_cgan.py --gen_model_pretr <trained-model-path-from prev step> --gen9k_path <path-to-generated-images> --target9k_path <path-to-generated-image-labels> --train_or_gen generate


#convert the generated npy images in png images,  and 9k real images to png images and save them and then calculate FID score
python harman_scripts/numpy2images.py --savedir <directory-to-save-results> --numpy_images_file <path-to-generated-images> --num_images 9000

# calculate FID assuing we have gpu access ,for this step, you need to install the pytorch FID package
python -m pytorch_fid --device "cuda:0" <directory-to-save-results>/generated_images <path-to-directory-having-real-images>

# we also need a path to the directory having real images <path-to-directory-having-real-images> which can be used to get the FID score between real and generated images from our GAN

```
