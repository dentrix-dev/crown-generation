import os
import numpy as np
import torch
from factories.losses_factory import get_loss
from rigidTransformations import apply_random_transformation
from tqdm import tqdm
from sklearn.metrics import accuracy_score

cuda = True if torch.cuda.is_available() else False
device = 'cuda' if cuda else 'cpu'

def train(model, train_loader, test_loader, args):

    train_accuracy = []
    train_loss = []
    test_accuracy = []
    test_loss = []

    train_miou = []
    test_miou = []
    train_acc = []
    test_acc = []

    criterion = get_loss(args.loss)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    for epoch in range(args.num_epochs):
        cum_loss = 0

        for vertices, crown_output, labels, jaw in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}'):

            if args.rigid_augmentation_train:
                vertices = apply_random_transformation(vertices, rotat=args.rotat, trans=args.trans)

            vertices, crown_output, labels, jaw = vertices.to(device), crown_output.to(device), labels.to(device), jaw.to(device)

            # Forward pass
            outputs = model(vertices, jaw)

            loss = criterion(outputs, crown_output)
            cum_loss += loss.item()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        # Calculate average loss
        cum_loss /= len(train_loader)
        train_loss.append(cum_loss)

        model.eval()
        t_loss = 0

        with torch.no_grad():
            for vertices, crown_output, labels, jaw in tqdm(test_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}'):
                vertices, crown_output, labels, jaw = vertices.to(device), crown_output.to(device), labels.to(device), jaw.to(device)

                if args.rigid_augmentation_test:
                    vertices = apply_random_transformation(vertices, rotat=args.rotat, trans=args.trans)

                # Forward pass
                outputs = model(vertices, jaw)

                t_loss += criterion(outputs, crown_output).item()

        # Calculate average loss
        t_loss /= len(test_loader)

        # Append metric  and loss to lists
        test_loss.append(t_loss)

        print(f'Epoch [{epoch + 1}/{args.num_epochs}], train_Loss: {cum_loss:.4f}, test_Loss: {t_loss:.4f}')
    print('Training finished.')

    torch.save(model.state_dict(), os.path.join(args.output, f"{args.model}_{epoch + 1}.pth"))

    return train_loss, test_loss
