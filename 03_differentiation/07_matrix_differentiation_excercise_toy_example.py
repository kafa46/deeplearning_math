'''소프트웨어 꼰대강의 - 선형 시스템 편미분 (실습)'''

from torch import nn
import torch

class Network(nn.Module):
    '''Toy Network'''
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        '''Forward Propagation'''
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


class Train():
    '''Toy Train'''
    def __init__(self) -> None:
        self.model = Network(4, 5, 2)
    
    def run(self,):
        '''Run network'''
        inputs = torch.randn(1, 4)
        output = self.model(inputs)
        
        # Loss 계산
        label = torch.tensor([1.0, 0.0])
        loss = torch.sum((output - label)**2)
        print(f'Predict (Y_hat): {output}')
        print(f'Loss: {loss}')
        
        loss.backward()
        
        gradients = [param for param in self.model.parameters()]
        
        print('Gradients Test')
        for idx, grad in enumerate(gradients):
            print('----' * 20)
            print(f'Layer {idx}, shape: {grad.shape}')
            print(grad)

if __name__=='__main__':
    train = Train()
    train.run()
