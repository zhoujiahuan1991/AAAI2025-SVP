fgvc_split.zip contains split information for the five FGVC tasks we evaluated. The training data for the datasize ablation experiments are also included.

All the json files share the same structure: {image_path : class_id}. They are compatible to the JSONDataset class from src/data/datasets/json_dataset.py.

The json files should be placed into the DATA.DATAPATH dir. And it is recommended to place the json files in the respective downloaded dataset folders:


CUB_200_2011
├── README
├── ...
├── images
├── attributes
├── *.json

nabirds
├── README
├── bounding_boxes.txt
├── ...
├── images
├── attributes
├── *.json

OxfordFlower
├── imagelabels.mat
├── jpg
├── ...
├── setid.mat
├── *.json


Stanford-cars
├── car_ims
├── cars_annos.mat
├── *.json


Stanford-dogs
├── Annotation
├── Images
├── ...
├── file_list.mat
├── *.json
