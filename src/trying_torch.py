import torch
def main():
    data_1 = [1,3,5]
    data_2 = [2,4,6]
    torch_1 = torch.tensor(data_1)
    torch_2 = torch.tensor(data_2)
    print(torch.stack((torch_1,torch_2),dim=1).view(-1,1).flatten())
main()