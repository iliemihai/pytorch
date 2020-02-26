import torch
import os

points = torch.tensor([[1.0,4.0], [2.0, 3.0], [4.0,5.0]])
print("Shape is: ", points.shape)
print("Storage is: ", points.storage())
points_storage = points.storage()
points_storage[0] = 2.0
print("New storage is: ", points_storage)
#size, storage offset, strides
second_point = points[1]
print("Second point storage: ", second_point.storage_offset())
print("Second point size: ", second_point.size())
i = 1
j = 1
print("Accesing i and j in 2D tensor", points.storage()[points.stride()[0]*i+points.stride()[1]*j])
print("Extracting subtensor: ")
second_point = points[1]
print("Size ", second_point.size())
print("Offset ", second_point.storage_offset())
print("Stride ", second_point.stride())
points_t = points.t()
print("Transpose: ", points_t)
assert(id(points.storage())==id(points_t.storage()))
print("Stride 1: ", points.stride(), "stride 2: ", points_t.stride())
some_tensor = torch.ones(3, 4, 5)
print("Shape of tensor is: ", some_tensor.shape, some_tensor.stride())
some_tensor_t = some_tensor.transpose(0, 2)
print("Shape of tensor transpose is: ", some_tensor_t.shape, some_tensor_t.stride())
print("Are points contiguos: ", points.is_contiguous(), points.stride())
print("Are points_t contiguous: ", points_t.is_contiguous(), points_t.stride())
print("Storage points_t is: ", points_t.storage())
points_t_cont = points_t.contiguous()
print("Are points_t_cont contiguous: ", points_t_cont.is_contiguous(), points_t_cont.stride())
print("Storage points_t_cont is: ", points_t_cont.storage())
# numeric types
torch.float32 or torch.float #32-bit floating-point
torch.float64 or torch.double #64-bit, double-precision floating-point
torch.float16 or torch.half #16-bit, half-precision floating-point
torch.int8 #Signed 8-bit integers
torch.uint8 #Unsigned 8-bit integers
torch.int16 or torch.short #Signed 16-bit integers
torch.int32 or torch.int #Signed 32-bit integers
torch.int64 or torch.long #Signed 64-bit integers
short_points = torch.tensor([[1,2],[3,4]], dtype=torch.int16)
print("Type is: ", short_points.dtype)
#numpy interoperability
new_points = torch.ones(3,5)
points_np = new_points.numpy()
print("NUmpy points: ", points_np)
#serializing tensors
path = "" +"./outpoints.t"
torch.save(points, path)
points = torch.load(path)
#exercises
a = torch.tensor(list(range(9)))
b = a.view(3, 3)
c = b[1:, 1:]
print(a,b,c)
