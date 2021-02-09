import os
from flask import Flask, request, jsonify, render_template

import torch
import torch.nn as nn
import torchvision.transforms as tt

import torchvision.models as models


from PIL import Image


app = Flask(__name__)
# model = pickle.load(open('models/model.pkl', 'rb'))


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
resnet_model = torch.load('models/fruit_modals_resnet_model.pt', map_location=torch.device('cpu'))


# Passing the image for prediction
def predict_image(img, model):

    yb = model(img)
    _, preds  = torch.max(yb, dim=1)

    # top 3 predictions
    res_value, res_indices = yb.topk(3)
    
    # calculating the % score 
    res_score1 = f'{round(res_value[0][0].item()*100)}%'
    res_score2 = f'{round(res_value[0][1].item()*100)}%'
    res_score3 = f'{round(res_value[0][2].item()*100)}%'

    # finding the class
    res_val1 = classes[res_indices[0][0].item()]
    res_val2 = classes[res_indices[0][1].item()]
    res_val3 = classes[res_indices[0][2].item()]

    result = {'res_val1': res_val1, 'res_val2': res_val2, 'res_val3': res_val3, 'res_score1': res_score1, 'res_score2': res_score2, 'res_score3': res_score3}

    # return classes[preds[0].item()]
    return result


def pred(img_name):

    img = Image.open(img_name)

    # Resizing the image
    img = img.resize((100, 100)) 
    img.save('sample.jpeg')

    img_tensor = tt.ToTensor()
    # print(img_tensor(img).unsqueeze(0).shape)

    inp_img = img_tensor(img).unsqueeze(0)

    return predict_image(inp_img, resnet_model)




@app.route('/')
def home():
    return render_template('home.html')

app.config['UPLOAD_FOLDER'] = 'static/pred_images/'


# file_name_i = '1'
@app.route('/predict_res',methods=['POST'])
def predict_res():
    '''
    For rendering results on HTML GUI
    '''

    if request.method == 'POST':
        file = request.files['file']

        file_extension = file.filename.split('.')[-1]
        # print(file_extension)
        # print('101')
        f_read = open('file_name.txt','r').read()

        file_name = f'{f_read}.{file_extension}'

        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)

        file.save(image_path)
        # file.save(os.path.join(file_name))

        result = pred(image_path)

        f_write = open('file_name.txt','w')
        f_write.write(str(int(f_read)+1))

        return render_template('result.html', value1 = result['res_val1'], value2 = result['res_val2'], value3 = result['res_val3'], score1=result['res_score1'], score2=result['res_score2'], score3=result['res_score3'], img_url = image_path)

    return 'failed'




# The below function will return the json response
file_name_i = '1'
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    if request.method == 'POST':
        file = request.files['file']

        file_extension = file.filename.split('.')[-1]
        # print(file_extension)
        # print('101')

        file_name = f'dummy/{file_name_i}.{file_extension}'

        # file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        file.save(os.path.join(file_name))

        result = pred(file_name)

        return {'result': result}

    return 'failed'


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port = 8000)