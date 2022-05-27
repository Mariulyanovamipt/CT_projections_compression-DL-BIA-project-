import numpy as np
np.random.seed(777)
s = np.random.uniform(1,10,(5,5)).astype('float32')
print(s.shape)
print(s)
print('---------')
import tensorflow_compression as tfc
import tensorflow as tf

tf.enable_eager_execution()

entropy_bottleneck_mv = tfc.EntropyBottleneck() #Quantize????
string = entropy_bottleneck_mv.compress(s) #type Tensor
with tf.Session() as sess:  print(string.eval())

print(string)
#a = string
#tf.Print(a, [a], message="This is a: ")
print(string[0])
for i in string:
    print(i)


s2 = np.random.normal(5, 3, (10,10))
entropy_bottleneck_mv = tfc.EntropyBottleneck() #Quantize????
string2 = entropy_bottleneck_mv.compress(s) #type Tensor
#with tf.Session() as sess:  print(string2.eval())
