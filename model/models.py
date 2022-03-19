from model.Darknet53 import DarkNet53
from model.lenet5 import Lenet5
from model.Darknet53 import DarkNet53
def get_model(model_name):
    if(model_name == "lenet5"):
        return Lenet5
    elif(model_name == "Darknet53"):
        return DarkNet53
    else:
        print("unknown model")