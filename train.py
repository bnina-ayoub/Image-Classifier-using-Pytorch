from get_train_args import get_args
from lab_check_functions import *
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F



def main():
    args = get_args()
    check_train_command_line_arguments(args)
    
    data_directory = args.data_directory
    save_directory = args.save_dir
    arch = args.arch
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    print_every = args.print_every
    use_gpu = args.gpu
    
    class_index, train_loaders, test_loaders, val_loaders = load_data(data_directory)
    if use_gpu :
        device = 'cuda'
    else:
        device = 'cpu'
    model, optimizer, criterion = load_model_for_training(arch, learning_rate, hidden_units, device)
    print("Started")
    running_loss = 0
    for epoch in range(epochs):
        steps = 0
        for images, labels in train_loaders:
            steps += 1
            print(steps)
            images, labels = images.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss+= loss.item()
            if steps % print_every == 0:
                model.eval()
                val_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for images, labels in val_loaders:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)
                        batch_loss = criterion(logps, labels)
                        val_loss += batch_loss.item()
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.cuda.FloatTensor)).item()
                    
                print(f"Steps {steps} Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Val loss: {val_loss/len(val_loaders):.3f}.. "
                  f"Val accuracy: {accuracy/len(val_loaders):.3f}")
                save_model_checkpoint(model, optimizer, arch, hidden_units, epoch, running_loss, class_index)
                running_loss = 0
                model.train()
    
if __name__ == "__main__":
    main()