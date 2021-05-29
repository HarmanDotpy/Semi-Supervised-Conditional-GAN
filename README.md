# Semi-Supervised-Conditional-GAN

For training the cGAN's

```bash
mkdir ./temp_saves #for saving the results

python uda.py --testing_query_input "$2" --output_testing_query_labels "./part3_temp_saves/testing_query.npy" --output_classifier "./part3_temp_saves/uda_classifier.pth" --query_datapath "./part3_temp_saves/query_64k_images.npy" --target_datapath "./part3_temp_saves/target_64k_images.npy" --supervised_datapath "./part3_temp_saves/kmeans-sampled_15k_1000each/dataX_kmeans_sampled_qt9c.npy" --supervised_labels "./part3_temp_saves/kmeans-sampled_15k_1000each/kmeans_sampled_qt9c_labels.npy" --output_qt_labels "./part3_temp_saves/uda_labels_qt.npy"

# use saved labels of uda classifier to train GAN
python harman_scripts/train_cgan.py --root_path_to_save "./part3_temp_saves/cgan_output" --traindatapath "./part3_temp_saves/query_target_64k_images.npy" --trainlabelspath   "./part3_temp_saves/uda_labels_qt.npy" --train_or_gen train --num_epochs 100

#generate 9k images in form of npy files and save as gen9k.npy and target9k.npy
python harman_scripts/train_cgan.py --gen_model_pretr "./part3_temp_saves/cgan_output/gen_trained.pth" --gen9k_path "$5" --target9k_path "$6" --train_or_gen generate


#convert the generated npy images in png images,  and 9k real images to png images and save them and then calculate FID score
python harman_scripts/numpy2images.py --savedir "./part1_temp_saves/cgan_output/generated_images" --numpy_images_file "$3" --num_images 9000

# calculate FID assuing we have gpu access ,for this step, you need to install the pytorch FID package
python -m pytorch_fid --device "cuda:0" "./part1_temp_saves/cgan_output/generated_images" <path-to-directory-having-real-images>

# we also need a path to the directory having real images <path-to-directory-having-real-images> which can be used to get the FID score between real and generated images from our GAN

```
