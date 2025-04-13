import os
import torch
from factories.losses_factory import get_loss
from rigidTransformations import apply_random_transformation
from tqdm import tqdm

cuda = True if torch.cuda.is_available() else False
device = 'cuda' if cuda else 'cpu'

def train(model, train_loader, test_loader, args):

    train_loss = []
    test_loss = []

    criterion = get_loss(args.loss)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    for epoch in range(args.num_epochs):
        cum_loss = 0
        for vertices, crown_output, masked_teeth, jaw in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}'):
            try:
                vertices, crown_output, masked_teeth, jaw = vertices.to(device), crown_output.to(device), masked_teeth.to(device), jaw.to(device)

                if args.rigid_augmentation_train:
                    vertices = apply_random_transformation(vertices, rotat=args.rotat, trans=args.trans)
                vertices = vertices - vertices.mean(dim=1, keepdim=True)

                outputs = model(vertices, masked_teeth, jaw)
                loss = criterion(outputs, crown_output)
                cum_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            except Exception as e:
                print("Error on batch:")
                print("vertices shape:", vertices.shape)
                print("crown_output shape:", crown_output.shape)
                print("masked_teeth shape:", masked_teeth.shape)
                print("jaw shape:", jaw.shape)
                raise e

        # Calculate average loss
        cum_loss /= len(train_loader)
        train_loss.append(cum_loss)

        model.eval()
        t_loss = 0

        with torch.no_grad():
            for vertices, crown_output, masked_teeth, jaw in tqdm(test_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}'):
                try:
                    vertices, crown_output, masked_teeth, jaw = vertices.to(device), crown_output.to(device), masked_teeth.to(device), jaw.to(device)

                    if args.rigid_augmentation_test:
                        vertices = apply_random_transformation(vertices, rotat=args.rotat, trans=args.trans)
                    vertices = vertices - vertices.mean(dim=1, keepdim=True)

                    # Forward pass
                    outputs = model(vertices, masked_teeth, jaw)

                    t_loss += criterion(outputs, crown_output).item()
                except Exception as e:
                    print("Error on batch:")
                    print("vertices shape:", vertices.shape)
                    print("crown_output shape:", crown_output.shape)
                    print("masked_teeth shape:", masked_teeth.shape)
                    print("jaw shape:", jaw.shape)
                    raise e

        # Calculate average loss
        t_loss /= len(test_loader)

        # Append metric  and loss to lists
        test_loss.append(t_loss)

        print(f'Epoch [{epoch + 1}/{args.num_epochs}], train_Loss: {cum_loss:.4f}, test_Loss: {t_loss:.4f}')
    print('Training finished.')

    torch.save(model.state_dict(), os.path.join(args.output, f"{args.model}_{epoch + 1}.pth"))

    return train_loss, test_loss
