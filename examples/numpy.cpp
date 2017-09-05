/*
Example shows how to repeat numpy operations using pytorch Tensor
*/

#include "ATen/ATen.h"
#include "ATen/Type.h"
#include <math.h>

using namespace at;
using namespace std;

void basic_sigmoid(int x) {
  double s = 1/(1+exp(-x));
  cout << s << endl;
}

void tensor_sigmoid(Tensor x) {
  x = x*-1;
  Tensor s = 1/(1+x.exp());
  cout << s << endl;
}

void tensor_sigmoid_derv(Tensor x) {
  Tensor s = x.sigmoid();
  s = s*(1-s);
  cout << s << endl;
}

void reshape() {
  Tensor a = CPU(kFloat).rand({3, 3, 2});
  cout << a.sizes() << endl;
  a.resize_({a.sizes()[0]*a.sizes()[1]*a.sizes()[2], 1});
  cout << a.sizes() << endl;
  cout << a << endl;
}

void norm() {
  float float_buffer[] = {0, 3, 4,
                          1, 6, 4};
  Tensor a = CPU(kFloat).tensorFromBlob(float_buffer, {2,3});
  cout << a << endl;
  a = a.norm(0, 1, true);
  cout << a << endl;
}

int main() {
  // TODO: start developing numpy alike methods
  // basic_sigmoid(3);
  float float_buffer[] = {1,2,3};
  tensor_sigmoid(CPU(kFloat).tensorFromBlob(float_buffer, {1,3}));
  // tensor_sigmoid_derv(CPU(kFloat).tensorFromBlob(float_buffer, {1,3}));
  // reshape();
  // norm();
  return 0;
}
