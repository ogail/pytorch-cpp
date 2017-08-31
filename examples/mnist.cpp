/*
Example shows how to run a resnet 50 imagenet-trained classification
model on a dummy input and save it to an hdf5 file. This output can be
later on compared to the output acquired from pytorch in a provided .ipynb
notebook -- results differ no more than 10^{-5}.
*/

#include "ATen/ATen.h"
#include "ATen/Type.h"
#include <map>
#include <math.h>

#include <pytorch.cpp>
#include <imagenet_classes.cpp>
#include <mnist_reader.hpp>
#include <opencv2/opencv.hpp>

using namespace at;
using namespace cv;
using namespace std;

Mat resize_image(vector<uint8_t> img, int dim_size, int square_size=224) {
/*
  Resizes via linear interpolation the provided image from 1D vector to 2D 28x28 OpenCV Mat.
*/
  Mat input_mat = Mat(dim_size, dim_size, CV_8UC1);
  memcpy(input_mat.data, img.data(), img.size()*sizeof(uint8_t));
  float scale = ( ( float ) square_size ) / dim_size;
  resize(input_mat, input_mat, Size(0, 0), scale, scale, INTER_LINEAR);
  return input_mat;
}

string train_net(string hdf5_checkpoint) {
/*
  Loads pretrained imagenet resnet50 checkpoint and trains on mnist dataset
  TODO: add backward pass
*/
  auto net = torch::resnet50_imagenet();
  net->load_weights(hdf5_checkpoint);
  // train net
  // net->cuda();
  // conduct training

  // retain checkpoint on disk
  string output_checkpoint = "../resnet50_mnist.h5";
  net->cpu();
  net->save_weights(output_checkpoint);

  return output_checkpoint;
}

void predict(string hdf5_checkpoint) {
/*
  Loads a pretrained MNIST resnet50 model and conducts a forward pass on 1st imag in the dataset to infer its class.
  TODO: change the output classes from image net to MNIST
*/
  auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("../mnist");
  auto net = torch::resnet50_imagenet();
  net->load_weights(hdf5_checkpoint);
  // net->cuda();
  
  auto input_image = dataset.training_images[0];
  int input_size = sqrt(input_image.size());
  Mat resized_img = resize_image(input_image, input_size);

  // Outputs height x width x 3 tensor converted from Opencv's Mat with 0-255 values
  // and convert to 0-1 range
  auto image_tensor = torch::convert_opencv_mat_image_to_tensor(resized_img).toType(CPU(kFloat)) / 255;

  // Reshape image into 1 x 3 x height x width
  image_tensor.resize_({1, 3, image_tensor.sizes()[0], image_tensor.sizes()[1]});

  auto image_batch_normalized_tensor = torch::preprocess_batch(image_tensor);

  // auto input_tensor_gpu = image_batch_normalized_tensor.toBackend(Backend::CUDA);
  auto input_tensor_gpu = image_batch_normalized_tensor.toBackend(Backend::CPU);

  auto result = net->forward(input_tensor_gpu);

  auto softmaxed = torch::softmax(result);

  Tensor top_probability_indexes;
  Tensor top_probabilies;

  tie(top_probabilies, top_probability_indexes) = topk(softmaxed, 5, 1, true);

  top_probability_indexes = top_probability_indexes.toBackend(Backend::CPU).view({-1});

  auto accessor = top_probability_indexes.accessor<int64_t,1>();

  cout << imagenet_classes[ accessor[0] ] << endl;
}

int main()
{
  string pretrained_checkpoint = "../resnet50_imagenet.h5";
  string checkpoint_filename = train_net(pretrained_checkpoint);
  predict(checkpoint_filename);
  return 0;
}
