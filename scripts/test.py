import torch


def test_1():
    def f(x, y):
        return 2*x + x*y

    # Define the input values
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)

    # Calculate the output
    output = f(x, y)

    # Compute the gradients
    output.backward()

    # Print the gradients
    print("Gradient of f with respect to x:", x.grad)
    print("Gradient of f with respect to y:", y.grad)


def test_2():

    # Define the perceptron
    class Perceptron(torch.nn.Module):
        def __init__(self):
            super(Perceptron, self).__init__()
            self.linear = torch.nn.Linear(2, 1)

        def forward(self, x):
            return torch.relu(self.linear(x))

    # Create an instance of the perceptron
    model = Perceptron()

    # Define the input
    x = torch.tensor([[2.0, 3.0]], dtype=torch.float32, requires_grad=True)

    # Forward pass
    output = model(x)

    # Compute the gradients
    output.backward()

    # Print the gradients
    print("Gradients:", model.linear.weight.grad)


if __name__ == '__main__':
    # test_1()
    test_2()
