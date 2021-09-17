import os

data_path = '../data'
data_name = ['parking-lot', 'urban-road', 'residential-area']
yaml_name = '64S2'
mu, theta = 11.445, 10.451 / 2
nbframe = 3
nepoch = 150
CKPT_PATH = '../checkpoint'
GRAPH_PATH = '../graph'
# CKPT_FILE = os.path.join(CKPT_PATH, 'ckpt-0915-2156.pth') # add batch norm のコミットの時の model, Adam(lr=0.01)
CKPT_FILE = os.path.join(CKPT_PATH, 'ckpt-0917-1401.pth') # change to relu のコミットの時の model, Adam(lr=0.001, weight_decay=5e-4)