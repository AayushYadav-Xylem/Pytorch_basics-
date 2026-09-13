# Pytorch_basics-
here in this fundamentals of the pytorch 

# Model creadtion 
Here in this we are trainging the model for the xor output 
import torch 
import torch.nn as nn

**create the tensor just like below**
X = torch.tensor([0., 0.], [0., 1.], [1., 0.], [1., 1.])
Y = torch.tensor([0.], [1.], [1.], [0.])

**_create the model so that we can predict_**
model == nn.Sequential(
          nn.Linear(2, 4),
          nn.Sigmoid(),
          nn.Linear(4, 1),
          nn.Sigmoid()
          )

**_Do some prediction and see how good is the model_**
y_pred = model(X)
print(y_pred)

**_the y_pred value and y differes and the difference between them is our actual loss_**
loss_fn = nn.BCELoss()
loss = loss_fn(y_pred, y)
print(loss)

**_you have to adjust the valu of it in the model using weight this is the single epoach_**
optimizer.torch.optim.SGD(model.parameters(), lr=0.5)
optimizer.zero_grad()
loss.backwards
optimizer.step()
y_pred_after = model(X)
print(y_pred)

**_here in this use the same logic but in the loop_**
for epoch in range(5000)
  y_pred = model(X)
  loss = loss_fn(y_pred, Y)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  if epoch % 500 == 0
    print(epoch, loss.item)

**_here in this as we have use the bce then it is the ln function so here it shoudl start nearly 0.69 as its indicate 50 chances_**
