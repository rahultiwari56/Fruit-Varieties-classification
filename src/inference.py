'''
    For reference
'''


import os
import torch
import torch.nn as nn
import torchvision.transforms as tt

import torchvision.models as models


from PIL import Image


class ResNetCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet34(pretrained=True)     # You can change the resnet model here
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 131)          # Output classes
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))
    
    def freeze(self):
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network.fc.parameters():
            param.require_grad = True
    
    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True


classes = ['Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3', 'Apple Granny Smith', 'Apple Pink Lady', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3', 'Apple Red Delicious', 'Apple Red Yellow 1', 'Apple Red Yellow 2', 'Apricot', 'Avocado', 'Avocado ripe', 'Banana', 'Banana Lady Finger', 'Banana Red', 'Beetroot', 'Blueberry', 'Cactus fruit', 'Cantaloupe 1', 'Cantaloupe 2', 'Carambula', 'Cauliflower', 'Cherry 1', 'Cherry 2', 'Cherry Rainier', 'Cherry Wax Black', 'Cherry Wax Red', 'Cherry Wax Yellow', 'Chestnut', 'Clementine', 'Cocos', 'Corn', 'Corn Husk', 'Cucumber Ripe', 'Cucumber Ripe 2', 'Dates', 'Eggplant', 'Fig', 'Ginger Root', 'Granadilla', 'Grape Blue', 'Grape Pink', 'Grape White', 'Grape White 2', 'Grape White 3', 'Grape White 4', 'Grapefruit Pink', 'Grapefruit White', 'Guava', 'Hazelnut', 'Huckleberry', 'Kaki', 'Kiwi', 'Kohlrabi', 'Kumquats', 'Lemon', 'Lemon Meyer', 'Limes', 'Lychee', 'Mandarine', 'Mango', 'Mango Red', 'Mangostan', 'Maracuja', 'Melon Piel de Sapo', 'Mulberry', 'Nectarine', 'Nectarine Flat', 'Nut Forest', 'Nut Pecan', 'Onion Red', 'Onion Red Peeled', 'Onion White', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Peach 2', 'Peach Flat', 'Pear', 'Pear 2', 'Pear Abate', 'Pear Forelle', 'Pear Kaiser', 'Pear Monster', 'Pear Red', 'Pear Stone', 'Pear Williams', 'Pepino', 'Pepper Green', 'Pepper Orange', 'Pepper Red', 'Pepper Yellow', 'Physalis', 'Physalis with Husk', 'Pineapple', 'Pineapple Mini', 'Pitahaya Red', 'Plum', 'Plum 2', 'Plum 3', 'Pomegranate', 'Pomelo Sweetie', 'Potato Red', 'Potato Red Washed', 'Potato Sweet', 'Potato White', 'Quince', 'Rambutan', 'Raspberry', 'Redcurrant', 'Salak', 'Strawberry', 'Strawberry Wedge', 'Tamarillo', 'Tangelo', 'Tomato 1', 'Tomato 2', 'Tomato 3', 'Tomato 4', 'Tomato Cherry Red', 'Tomato Heart', 'Tomato Maroon', 'Tomato Yellow', 'Tomato not Ripened', 'Walnut', 'Watermelon']

# load the model
resnet_modelzz = torch.load('models/fruit_modals_resnet_model.pt', map_location=torch.device('cpu'))


def predict_image(img, model):
    # Convert to a batch of 1
    # xb = img.unsqueeze(0)
    # xb = to_device(img.unsqueeze(0), device)

    # Get predictions from model
    # model.eval()
    yb = model(img)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    print(preds[0].item())
    # Retrieve the class label
    print('54')
    # print(yb[0][16]*100)
    res_value, res_indices = yb.topk(3)
    print(res_value)
    print(res_indices)

    res_score1 = res_value[0][0].item()*100
    res_score2 = res_value[0][1].item()*100
    res_score3 = res_value[0][2].item()*100
    print(res_score1)
    print(res_score2)
    print(res_score3)

    print('-------------------')
    print(res_indices[0][0].item())

    res_val1 = classes[res_indices[0][0].item()]
    res_val2 = classes[res_indices[0][1].item()]
    res_val3 = classes[res_indices[0][2].item()]
    print(res_val1)
    print(res_val2)
    print(res_val3)

    return classes[preds[0].item()]

def pred(img_name):

    # Load the image, using PIL
    img = Image.open(img_name)
    # img = Image.open("dummy/5.jpeg")
    # width, height = img.size 

    img = img.resize((100, 100)) 
    img.save('121.jpeg')
    print(type(img))

    # valid_tfms = tt.Compose([tt.ToTensor()])
    # valid_ds = ImageFolder(TEST_DIR, valid_tfms)


    # convert the image to tensor
    # print(img.shape())
    img_tensor = tt.ToTensor()
    print(img_tensor(img).unsqueeze(0).shape)

    inp_img = img_tensor(img).unsqueeze(0)

    # print('Predicted:', predict_image(inp_img, resnet_modelzz))
    return predict_image(inp_img, resnet_modelzz)


# convert the dimension of image to 4D
# ex: (1, 100, 100, 3)

# # send this to model for prediction



# img, label = valid_ds[2500]
# plt.imshow(img.permute(1, 2, 0))
# print('Label:', valid_ds.classes[label], ', Predicted:', predict_image(img_tensor, resnet_model))


# print(pred('1.png'))
print(pred('2_100.jpg'))