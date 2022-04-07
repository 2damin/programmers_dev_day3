import sys, os
import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from torchvision import datasets
from model.models import *
from loss.loss import *
from util.tools import *
from torchsummary import summary
from tensorboardX import SummaryWriter

N_CLASSES = 1
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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

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
        
    torch_writer = SummaryWriter("./output")
        
    #Dataset
    train_dir = os.path.join(args.data)
    valid_dir = os.path.join(args.data, 'val')
    
    train_transform = transforms.Compose([transforms.Resize((224,224)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation((-0.1,0.1)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485,0.456,0.406],
                                                               std=[0.229, 0.224, 0.225])
                                          ])

    valid_transform = transforms.Compose([transforms.Resize((224,224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485,0.456,0.406],
                                                               std=[0.229, 0.224, 0.225])])
    
    train_dataset = datasets.ImageFolder(train_dir,
                                               train_transform)
    
    valid_dataset = datasets.ImageFolder(train_dir,
                                               valid_transform)
    
    #Make Dataloader
    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              num_workers=4,
                              pin_memory=True,
                              drop_last=True,
                              shuffle=True)
    eval_loader = DataLoader(valid_dataset,
                             batch_size=1,
                             num_workers=0,
                             pin_memory=True,
                             drop_last=False,
                             shuffle=True)
    
    #numger of data per each class
    num_data_class = torch.unique(torch.tensor(train_dataset.targets), return_counts=True)
    #total counts
    total_counts = torch.sum(num_data_class[1])
    #cross-entropy weights
    ce_weights = torch.tensor([1 - t / total_counts for t in num_data_class[1]], dtype=torch.float32)
    print(ce_weights)
    
    _model = get_model('Darknet53')

    if args.mode == "train":
        model = _model(batch = 32, n_classes=N_CLASSES, in_channel=3, in_width=224, in_height=224, is_train=True)
        
        summary(model, input_size=(3, 224, 224), device='cpu' )
        if args.checkpoint is not None:
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
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.1) #momentum=0.9,
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        
        criterion = get_criterion(crit='bce', device=device)
        
        epoch = 150
        iter = 0
        for e in range(epoch):
            model.train()
            total_loss = 0
            for i, batch in enumerate(train_loader):
                img = batch[0]
                gt = batch[1]
                
                # img_data = np.array(np.transpose(img.detach().numpy()[0]*255, (1,2,0)), dtype=np.uint8)
                # print(img_data.shape)
                # imgshow = Image.fromarray(img_data)
                # imgshow.save(str(gt[0].numpy())+ "_"+str(i)+".png")

                img = img.to(device)
                gt = gt.type(torch.FloatTensor).to(device)
                gt = gt.reshape((gt.shape[0], 1))
                
                optimizer.zero_grad()
                
                out = model(img)
                
                loss_val = criterion(out, gt)
                
                #backpropagation
                loss_val.backward()
                optimizer.step()
                
                total_loss += loss_val.item()
                
                if iter % 10 == 0:
                    print("{} epoch {} iter loss : {} lr : {}".format(e, iter, loss_val.item(), get_lr(optimizer)))
                    torch_writer.add_scalar('loss', loss_val.item(), iter)
                    torch_writer.add_scalar('lr', get_lr(optimizer), iter)
                
                iter += 1
            
            mean_loss = total_loss / i
            scheduler.step()
            
            #evaluation
            model.eval()
            acc = 0
            num_eval = 0

            for i, batch in enumerate(eval_loader):
                #skip 20 images
                if i % 20 != 0:
                    continue
                img = batch[0]
                gt = batch[1]
                img = img.to(device)
                #inference
                out = model(img)
                out = out.detach().cpu()
                out = 1 if out > 0.5 else 0
                
                score = out == gt
                if score.item() == 1:
                    acc += 1
                num_eval += 1
            print("Evaluation score : {} / {}".format(acc, num_eval))
            torch_writer.add_scalar('acc', acc/num_eval, iter)
            
            print("->{} epoch mean loss : {}".format(e, mean_loss))
            torch.save(model.state_dict(), args.output_dir + "/model_epoch"+str(e)+".pt")
        print("Train end")
    elif args.mode == "eval":
        model = _model(batch = 1, n_classes=N_CLASSES, in_channel=3, in_width=224, in_height=224)
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
            
            out = torch.nn.functional.softmax(out,dim=1)
            print(out.shape)
            print(torch.max(out), out[0,torch.argmax(out)])
            out = torch.argmax(out)
            print(out, gt)
            out = out.detach().cpu()
            
            score = out == gt
            
            if score.item() == 1:
                acc += 1
            num_eval += 1
            
        print("Evaluation score : {} / {}".format(acc, num_eval))

if __name__ == "__main__":
    args = parse_args()
    main()