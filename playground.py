import crypten
import torch


crypten.init()
model = crypten.nn.Linear(3, 2)
print("c0c0c0")
print("============")
print(model.weight)
# print("c2 c2 c2")
print(model._parameters["weight"])

model._parameters["weight"] = torch.nn.Parameter(torch.ones((2, 3)))
print("============")
# model.test()
# print("c1c1c1")
print(model.weight)
# print("c2 c2 c2")
print(model._parameters["weight"])

model.weight = torch.nn.Parameter(torch.zeros((2, 3)))
print("============")

# print("c1c1c1")
print(model.weight)
# print("c2 c2 c2")
print(model._parameters["weight"])
