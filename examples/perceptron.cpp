/*
Example shows how to run a resnet 50 imagenet-trained classification
model on a dummy input and save it to an hdf5 file. This output can be
later on compared to the output acquired from pytorch in a provided .ipynb
notebook -- results differ no more than 10^{-5}.
*/

#include "ATen/ATen.h"
#include "ATen/Type.h"
#include <map>
#include <vector>
#include <math.h>

#include <pytorch.cpp>
#include <imagenet_classes.cpp>
#include <mnist_reader.hpp>
#include <opencv2/opencv.hpp>

using namespace at;
using namespace cv;
using namespace std;
using namespace mnist;

struct experiment_result {
  vector<double> costs;
  Tensor Y_prediction_test;
  Tensor Y_prediction_train;
  Tensor w;
  Tensor b;
  double learning_rate;
  int num_iterations;
};

struct propagate_result {
  double cost;
  Tensor dw;
  Tensor db;
};

struct optimize_result {
  map<string, Tensor> params;
  map<string, Tensor> grads;
  vector<double> costs;
};

struct dataset_preprocess_result {
  Tensor X;
  Tensor Y;
};

struct init_result {
  Tensor w;
  Tensor b;
};

Mat resize_image(vector<uint8_t> img, int dim_size, int square_size=224) {
/*
  Resizes via linear interpolation the provided image from 1D vector to 2D 28x28 OpenCV Mat.
*/
  Mat input_mat = Mat(dim_size, dim_size, CV_8UC1);
  memcpy(input_mat.data, img.data(), img.size()*sizeof(uint8_t));
  double scale = ( ( double ) square_size ) / dim_size;
  resize(input_mat, input_mat, Size(0, 0), scale, scale, INTER_LINEAR);
  return input_mat;
}

string train_net(string hdf5_checkpoint, vector<vector<uint8_t>> training_imgs, vector<uint8_t> labels) {
/*
  Loads pretrained imagenet resnet50 checkpoint and trains on mnist dataset
  TODO: add backward pass
*/
  auto net = torch::resnet50_imagenet();
  net->load_weights(hdf5_checkpoint);
  // train net
  // net->cuda();
  // conduct training
  
  // 1. create input tensor for training images
  int num_examples = training_imgs.size();
  int img_size = training_imgs[0].size();
  double* train_data = new double[num_examples * img_size];
  double* ptr_train_data = train_data;
  for (int i=0; i < training_imgs.size(); i++) {
    copy(training_imgs[i].begin(), training_imgs[i].end(), ptr_train_data);
    ptr_train_data += training_imgs[i].size();
  }
  Tensor train_imgs = CPU(kDouble).tensorFromBlob(train_data, {num_examples, img_size});

  // 2. create input tensor for training labels
  double* train_labels_data = new double[num_examples];
  copy(labels.begin(), labels.end(), train_labels_data);
  Tensor train_labels = CPU(kDouble).tensorFromBlob(train_labels_data, {num_examples, 1});

  // 3. normalize training data (max pixel is 255)
  train_imgs = train_imgs / 255;

  // retain checkpoint on disk
  string output_checkpoint = "../resnet50_mnist.h5";
  net->cpu();
  net->save_weights(output_checkpoint);

  return output_checkpoint;
}

void predict(string hdf5_checkpoint, const std::vector<uint8_t> &input_image) {
/*
  Loads a pretrained MNIST resnet50 model and conducts a forward pass on 1st imag in the dataset to infer its class.
  TODO: change the output classes from image net to MNIST
*/
  auto net = torch::resnet50_imagenet();
  net->load_weights(hdf5_checkpoint);
  // net->cuda();

  int input_size = sqrt(input_image.size());
  Mat resized_img = resize_image(input_image, input_size);

  // Outputs height x width x 3 tensor converted from Opencv's Mat with 0-255 values
  // and convert to 0-1 range
  auto image_tensor = torch::convert_opencv_mat_image_to_tensor(resized_img).toType(CPU(kDouble)) / 255;

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

/*
This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

Argument:
dim -- size of the w vector we want (or number of parameters in this case)

Returns:
w -- initialized vector of shape (dim, 1)
b -- initialized scalar (corresponds to the bias)
*/
init_result initialize_with_zeros(int dim) {
  Tensor w = CPU(kDouble).zeros({dim,1});
  Tensor b = CPU(kDouble).scalarTensor(0);
  init_result res;
  res.w = w;
  res.b = b;
  return res;
}

/*
Does a one backward propagation step.
Arguments:
w -- weights, a numpy array of size (num_px * num_px * 3, 1)
b -- bias, a scalar
X -- data of size (num_px * num_px * 3, number of examples)
Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

Return:
cost -- negative log-likelihood cost for logistic regression
dw -- gradient of the loss with respect to w, thus same shape as w
db -- gradient of the loss with respect to b, thus same shape as b
*/
propagate_result propagate(Tensor w, Tensor b, Tensor X, Tensor Y) {
  double m = X.size(1);
  Tensor A = ((w.t().mm(X))+b).sigmoid();
  double cost = -((Y*A.log())+((1-Y)*(1-A).log())).sum().toDouble()/m;
  // dw = np.dot(X, (A-Y).T)/m
  Tensor dw = (X.mm((A-Y).t()))/m;
  Tensor db = CPU(kDouble).scalarTensor((A-Y).sum().toDouble()/m);
  propagate_result res;
  res.cost = cost;
  res.dw = dw;
  res.db = db;
  return res;
}

/*
This function optimizes w and b by running a gradient descent algorithm

Arguments:
w -- weights, a numpy array of size (num_px * num_px * 3, 1)
b -- bias, a scalar
X -- data of shape (num_px * num_px * 3, number of examples)
Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
num_iterations -- number of iterations of the optimization loop
learning_rate -- learning rate of the gradient descent update rule
print_cost -- True to print the loss every 100 steps

Returns:
params -- dictionary containing the weights w and bias b
grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

Tips:
You basically need to write down two steps and iterate through them:
    1) Calculate the cost and the gradient for the current parameters. Use propagate().
    2) Update the parameters using gradient descent rule for w and b.
*/
optimize_result optimize(Tensor w, Tensor b, Tensor X, Tensor Y, int num_iterations, double learning_rate, bool print_cost) {
  vector<double> costs;
  Tensor dw, db;
  for (int i=0; i < num_iterations; i++) {
    // cost and gradient calculation
    propagate_result propagate_res = propagate(w, b, X, Y);
    
    // retrieve derivatives from grads
    dw = propagate_res.dw;
    db = propagate_res.db;

    // update rule
    w = w - (learning_rate * dw);
    b = b - (learning_rate * db);

    if (!(i % 100)) {
      costs.push_back(propagate_res.cost);
      cout << "Cost after iteration " << i << ": " << propagate_res.cost << endl;;
    }
  }

  optimize_result res;
  map<string, Tensor> params = { {"w", w}, {"b", b} };
  map<string, Tensor> grads = { {"dw", dw}, {"b", db} };
  res.params = params;
  res.grads = grads;
  res.costs = costs;
  return res;
}

/*
Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

Arguments:
w -- weights, a numpy array of size (num_px * num_px * 3, 1)
b -- bias, a scalar
X -- data of size (num_px * num_px * 3, number of examples)

Returns:
Y_prediction -- a Tensor containing all predictions (0/1) for the examples in X
*/
Tensor predict(Tensor w, Tensor b, Tensor X) {
  int m = X.size(1);
  Tensor Y_prediction = CPU(kDouble).zeros({1,m});
  auto Y_prediction_a = Y_prediction.accessor<double,2>();
  w.resize_({X.size(0), 1});
  Tensor A = (w.t().mm(X)+b).sigmoid();
  auto A_a = A.accessor<double,2>();
  for (int i=0; i < A.size(1); i++) {
     if (A_a[0][i] <= 0.5) {
      Y_prediction_a[0][i] = 0;
     } else {
      Y_prediction_a[0][i] = 1;
     }
  }
  return Y_prediction;
}

/*
Builds the logistic regression model by calling the function you've implemented previously

Arguments:
X_train -- training set represented by a numpy array of shape (28*28, m_train)
Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
X_test -- test set represented by a numpy array of shape (28*28, m_test)
Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
print_cost -- Set to true to print the cost every 100 iterations

Returns:
result -- experiment_result containing information about the model.
*/
experiment_result model(Tensor X_train, Tensor Y_train, Tensor X_test, Tensor Y_test, int num_iterations = 2000, double learning_rate = 0.5, bool print_cost = false) {
  // initialize parameters with zeros
  init_result init_res = initialize_with_zeros(X_train.size(0));
  Tensor w = init_res.w;
  Tensor b = init_res.b;
  

  // Gradient descent
  optimize_result res = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost);
  map<string, Tensor> parameters=res.params, grads=res.grads;
  vector<double> costs=res.costs;

  // Retrieve parameters w and b from dictionary "parameters"
  w = parameters["w"];
  b = parameters["b"];

  // Predict test/train set examples
  Tensor Y_prediction_test = predict(w, b, X_test);
  Tensor Y_prediction_train = predict(w, b, X_train);

  // Print accuracy
  Tensor train_diff = Y_prediction_train - Y_train;
  Tensor train_abs_s = train_diff.abs() * 100;
  double train_acc = 100 - train_abs_s.mean().toDouble();

  Tensor test_diff = Y_prediction_test - Y_test;
  Tensor test_abs_s = test_diff.abs() * 100;
  double test_acc = 100 - test_abs_s.mean().toDouble();
  
  cout << "train accuracy: " << train_acc << endl;
  cout << "test accuracy: " << test_acc << endl;

  experiment_result result;
  result.Y_prediction_test = Y_prediction_test;
  result.Y_prediction_train = Y_prediction_train;
  result.w = w;
  result.b = b;
  result.learning_rate = learning_rate;
  result.num_iterations = num_iterations;

  return result;
}

dataset_preprocess_result preprocess_dataset(vector<vector<uint8_t>> images, vector<uint8_t> labels) {
  // update mnist labels so the problem is binary, detecting if provided image has 'a' or not.
  for (int i=0; i < labels.size(); i++) {
    if (labels[i]) {
      // this label holds letter other than 'a'
      labels[i] = 0;
    } else {
      // this label holds 'a'
      labels[i] = 1;
    }
  }

  // common variables
  int num_examples = labels.size();

  // create images tensor and normalize the values
  int img_size = images[0].size();
  double* data = new double[num_examples * img_size];
  double* ptr_data = data;
  for (int i=0; i < images.size(); i++) {
    copy(images[i].begin(), images[i].end(), ptr_data);
    ptr_data += images[i].size();
  }
  Tensor X = CPU(kDouble).tensorFromBlob(data, {num_examples, img_size});
  X = X / 255;
  X.resize_({X.size(1), X.size(0)});

  // create labels tensor
  double* labels_data = new double[num_examples];
  copy(labels.begin(), labels.end(), labels_data);
  Tensor Y = CPU(kDouble).tensorFromBlob(labels_data, {num_examples, 1});
  Y.resize_({Y.size(1), Y.size(0)});

  dataset_preprocess_result result;
  result.X = X;
  result.Y = Y;
  return result;
}

int main() {
  auto dataset = read_dataset<vector, vector, uint8_t, uint8_t>("../mnist");
  Tensor train_set_x, train_set_y, test_set_x, test_set_y;
  dataset_preprocess_result res;
  
  // preprocess training dataset
  res = preprocess_dataset(dataset.training_images, dataset.training_labels);
  train_set_x = res.X;
  train_set_y = res.Y;
  
  // preprocess training dataset
  res = preprocess_dataset(dataset.test_images, dataset.test_labels);
  test_set_x = res.X;
  test_set_y = res.Y;

  experiment_result r = model(train_set_x, train_set_y, test_set_x, test_set_y, 2000, 0.005, true);
  return 0;
}
