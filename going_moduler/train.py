"""
Train a Pytorch image classification model...
"""

import os
import torch
import transformations, data_setup, model_builder, engine, utils


# 1. Define Constants
NUM_EPOCHS = 3
BATCH_SIZE = 8
HIDDEN_UNITS = 10
LR = 0.001


# 2. Setup directories
train_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, "Datasets", "pizza_steak_sushi", "train"))
test_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, "Datasets", "pizza_steak_sushi", "test"))


# 3. Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device}")


# 4. Create transforms
data_transform = transformations.data_transform_function(img_size = 64)


# 5. Create Datasets, DataLoaders and get class_names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir = train_dir, 
                                                                               test_dir = test_dir,
                                                                               transform = data_transform,
                                                                               batch_size = BATCH_SIZE)



# 5. Initialize Model
model = model_builder.TinyVGG(input_shape = 3, 
                              hidden_units = HIDDEN_UNITS, 
                              output_shape = len(class_names)).to(device)


# 6. Setup Loss and Optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr = LR)


# 7. Start the training wuth help from engine.py
engine.train(model = model,
             train_dataloader = train_dataloader,
             test_dataloader = test_dataloader,
             loss_fn = loss_fn,
             optimizer = optimizer,
             epochs = NUM_EPOCHS,
             device = device
             )


# 8. Save the trained model to file
utils.save_model(model = model,
                 target_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, "models")),
                 model_name = "13_going_moduler_tiny_vgg_model.pth")
