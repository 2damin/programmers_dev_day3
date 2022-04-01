import sys, os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from torchvision import datasets
from model.models import *
from loss.loss import *
from util.tools import *


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST")
    parser.add_argument('--mode', dest='mode', help="train / eval",
                        default=None, type=str)
    parser.add_argument('--output_dir', dest='output_dir', help="output directory",
                        default='./output', type=str)
    parser.add_argument('--checkpoint', dest='checkpoint', help="checkpoint trained model",
                        default=None, type=str)
    parser.add_argument('--data', dest='data', help="data directory",
                        default='./output', type=str)
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    args = parser.parse_args()
    return args

def main():
    print(torch.__version__)
    
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    
    if torch.cuda.is_available():
        print("gpu")
        device = torch.device("cuda")
    else:
        print("cpu")
        device = torch.device("cpu")
        
    #Dataset
    train_dir = os.path.join(args.data, 'train')
    valid_dir = os.path.join(args.data, 'val')
    
    train_transform = transforms.Compose([transforms.Resize(64),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485,0.456,0.406],
                                                               std=[0.229, 0.224, 0.225])])
    valid_transform = transforms.Compose([transforms.Resize(64),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485,0.456,0.406],
                                                               std=[0.229, 0.224, 0.225])])
    
    train_dataset = datasets.ImageFolder(train_dir,
                                               train_transform)
    
    valid_dataset = datasets.ImageFolder(valid_dir,
                                               valid_transform)
    
    #Make Dataloader
    train_loader = DataLoader(train_dataset,
                              batch_size=8,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True,
                              shuffle=True)
    eval_loader = DataLoader(valid_dataset,
                             batch_size=1,
                             num_workers=0,
                             pin_memory=True,
                             drop_last=False,
                             shuffle=False)
    
    _model = get_model('Darknet53')
    
    #LeNet5
    
    if args.mode == "train":
        model = _model(batch = 8, n_classes=200, in_channel=3, in_width=64, in_height=64, is_train=True)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        pretrained_state_dict = checkpoint['state_dict']
        model_state_dict = model.state_dict()                 
        for key, value in pretrained_state_dict.items():
            if key == 'fc.weight' or key == 'fc.bias':
                continue
            else:
                model_state_dict[key] = value
        
        model.load_state_dict(model_state_dict)
        model.to(device)
        model.train()
        
        #optimizer & scheduler
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        criterion = get_criterion(crit='bce', device=device)
        
        epoch = 5
        iter = 0
        for e in range(epoch):
            total_loss = 0
            for i, batch in enumerate(train_loader):
                img = batch[0]
                gt = batch[1]
                
                img = img.to(device)
                gt = gt.to(device)
                
                out = model(img)
                
                loss_val = criterion(out, gt)
                
                #backpropagation
                loss_val.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss_val.item()
                
                if iter % 100 == 0:
                    print("{} epoch {} iter loss : {}".format(e, iter, loss_val.item()))   
                iter += 1
            
            mean_loss = total_loss / i
            scheduler.step()
            
            print("->{} epoch mean loss : {}".format(e, mean_loss))
            torch.save(model.state_dict(), args.output_dir + "/model_epoch"+str(e)+".pt")
        print("Train end")
    elif args.mode == "eval":
        model = _model(batch = 1, n_classes=200, in_channel=3, in_width=64, in_height=64)
        #load trained model
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval() #not train()
        
        acc = 0
        num_eval = 0
        
        for i, batch in enumerate(eval_loader):
            img = batch[0]
            gt = batch[1]
            
            img = img.to(device)
            
            #inference
            out = model(img)
            
            out = out.cpu()

            if out == gt:
                acc += 1
            num_eval += 1
            
        print("Evaluation score : {} / {}".format(acc, num_eval))

if __name__ == "__main__":
    args = parse_args()
    main()