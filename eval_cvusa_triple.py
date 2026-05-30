import os
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader
from camnet.dataset.cvusa import CVUSADatasetEval as CVUSADatasetEval
from camnet.dataset.cvusa_bev import CVUSADatasetEval as CVUSADatasetEval_Bev
from camnet.dataset.cvusa_polar import CVUSADatasetEval as CVUSADatasetEval_Polar
from camnet.transforms import get_transforms_val as get_transforms_val
from camnet.transforms_bev import get_transforms_val as get_transforms_val_bev
from camnet.transforms_polar import get_transforms_val as get_transforms_val_polar
from camnet.evaluate.cvusa_and_cvact import evaluate_triple
from camnet.model import TimmModel


@dataclass
class Configuration:
    
    # Model
    model_1: str = 'convnext_base.fb_in22k_ft_in1k_384'
    model_2: str = 'convnext_base.fb_in22k_ft_in1k_384'
    model_3: str = 'convnext_base.fb_in22k_ft_in1k_384'
    
    # Override model image size
    img_size: int = 384
    
    # Evaluation
    batch_size: int = 32
    verbose: bool = True
    gpu_ids: tuple = (0,)
    normalize_features: bool = True
    
    # Dataset
    data_folder = "../CVUSA"    
    
    # Checkpoint to start from
    
    checkpoint_start_1 = 'BEV model path'   
    checkpoint_start_2 = 'Polar model path'
    checkpoint_start_3 = 'Raw model path'
  
    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    

#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = Configuration() 


if __name__ == '__main__':

    #-----------------------------------------------------------------------------#
    # Model_1                                                                       #
    #-----------------------------------------------------------------------------# 
    print("\nModel_1: {}".format(config.model_1))
    model_1 = TimmModel(config.model_1,
                      pretrained=True,
                      img_size=config.img_size)          
    data_config_1 = model_1.get_config()
    mean_1 = data_config_1["mean"]
    std_1 = data_config_1["std"]
    img_size = config.img_size
    image_size_sat_1 = (img_size, img_size)
    img_size_ground_1 = image_size_sat_1
    # load pretrained Checkpoint    
    if config.checkpoint_start_1 is not None:  
        print("Start from:", config.checkpoint_start_1)
        model_state_dict_1 = torch.load(config.checkpoint_start_1)  
        model_1.load_state_dict(model_state_dict_1, strict=False)     
    # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model_1 = torch.nn.DataParallel(model_1, device_ids=config.gpu_ids)
            
    # Model to device   
    model_1 = model_1.to(config.device)

    print("\nImage Size Sat_1:", image_size_sat_1)
    print("Image Size Ground_1:", img_size_ground_1)
    print("Mean1: {}".format(mean_1))
    print("Std1:  {}\n".format(std_1)) 
    #-----------------------------------------------------------------------------#
    # Model_2                                                                       #
    #-----------------------------------------------------------------------------#
    print("\nModel_2: {}".format(config.model_2))
    model_2 = TimmModel(config.model_2,
                      pretrained=True,
                      img_size=config.img_size)             
    data_config_2 = model_2.get_config()
    mean_2 = data_config_2["mean"]
    std_2 = data_config_2["std"]
    new_width = img_size * 2    
    new_hight = round((224 / 1232) * new_width)
    image_size_sat_2 = (new_hight, new_width)
    img_size_ground_2 = image_size_sat_2
    # load pretrained Checkpoint    
    if config.checkpoint_start_2 is not None:  
        print("Start from:", config.checkpoint_start_2)
        model_state_dict_2 = torch.load(config.checkpoint_start_2)  
        model_2.load_state_dict(model_state_dict_2, strict=False)     
    # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model_2 = torch.nn.DataParallel(model_2, device_ids=config.gpu_ids)
    # Model to device   
    model_2 = model_2.to(config.device)
    print("\nImage Size Sat_2:", image_size_sat_2)
    print("Image Size Ground_2:", img_size_ground_2)
    print("Mean2: {}".format(mean_2))
    print("Std2:  {}\n".format(std_2)) 

    #-----------------------------------------------------------------------------#
    # Model_3                                                                       #
    #-----------------------------------------------------------------------------#

    print("\nModel_3: {}".format(config.model_3))


    model_3 = TimmModel(config.model_3,
                      pretrained=True,
                      img_size=config.img_size)
                          
    data_config_3 = model_3.get_config()
    mean_3 = data_config_1["mean"]
    std_3 = data_config_1["std"]
    image_size_sat_3 = (img_size, img_size)
    img_size_ground_3 = (new_hight, new_width)
     
    # load pretrained Checkpoint    
    if config.checkpoint_start_3 is not None:  
        print("Start from:", config.checkpoint_start_3)
        model_state_dict_3 = torch.load(config.checkpoint_start_3)  
        model_3.load_state_dict(model_state_dict_3, strict=False)     

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model_3 = torch.nn.DataParallel(model_3, device_ids=config.gpu_ids)
            
    # Model to device   
    model_3 = model_3.to(config.device)

    print("\nImage Size Sat_3:", image_size_sat_3)
    print("Image Size Ground_3:", img_size_ground_3)
    print("Mean3: {}".format(mean_3))
    print("Std3:  {}\n".format(std_3)) 


    #-----------------------------------------------------------------------------#
    # DataLoader_1                                                                  #
    #-----------------------------------------------------------------------------#
    # Eval
    sat_transforms_val_1, ground_transforms_val_1 = get_transforms_val_bev(image_size_sat_1,
                                                               img_size_ground_1,
                                                               mean=mean_1,
                                                               std=std_1,
                                                               )
    # Reference Satellite Images
    reference_dataset_test_1 = CVUSADatasetEval_Bev(data_folder=config.data_folder ,
                                              split="test",
                                              img_type="reference",
                                              transforms=sat_transforms_val_1,
                                              )
    reference_dataloader_test_1 = DataLoader(reference_dataset_test_1,
                                           batch_size=config.batch_size,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True)
    # Query Ground Images Test
    query_dataset_test_1 = CVUSADatasetEval_Bev(data_folder=config.data_folder ,
                                          split="test",
                                          img_type="query",    
                                          transforms=ground_transforms_val_1,
                                          )
    query_dataloader_test_1 = DataLoader(query_dataset_test_1,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    print("Reference Images Test:", len(reference_dataset_test_1))
    print("Query Images Test:", len(query_dataset_test_1))

    #-----------------------------------------------------------------------------#
    # DataLoader_2                                                                  #
    #-----------------------------------------------------------------------------#

    # Eval
    sat_transforms_val_2, ground_transforms_val_2 = get_transforms_val_polar(image_size_sat_2,
                                                               img_size_ground_2,
                                                               mean=mean_2,
                                                               std=std_2,
                                                               )
    # Reference Satellite Images
    reference_dataset_test_2 = CVUSADatasetEval_Polar(data_folder=config.data_folder ,
                                              split="test",
                                              img_type="reference",
                                              transforms=sat_transforms_val_2,
                                              )
    reference_dataloader_test_2 = DataLoader(reference_dataset_test_2,
                                           batch_size=config.batch_size,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True)
    # Query Ground Images Test
    query_dataset_test_2 = CVUSADatasetEval_Polar(data_folder=config.data_folder ,
                                          split="test",
                                          img_type="query",    
                                          transforms=ground_transforms_val_2,
                                          )
    query_dataloader_test_2 = DataLoader(query_dataset_test_2,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    print("Reference Images Test:", len(reference_dataset_test_2))
    print("Query Images Test:", len(query_dataset_test_2))

    #-----------------------------------------------------------------------------#
    # DataLoader_3                                                                  #
    #-----------------------------------------------------------------------------#
        
    
    # Eval
    sat_transforms_val_3, ground_transforms_val_3 = get_transforms_val(image_size_sat_3,
                                                               img_size_ground_3,
                                                               mean=mean_3,
                                                               std=std_3,
                                                               )


    # Reference Satellite Images
    reference_dataset_test_3 = CVUSADatasetEval(data_folder=config.data_folder ,
                                              split="test",
                                              img_type="reference",
                                              transforms=sat_transforms_val_3,
                                              )
    
    reference_dataloader_test_3 = DataLoader(reference_dataset_test_3,
                                           batch_size=config.batch_size,
                                           num_workers=config.num_workers,
                                           shuffle=False,
                                           pin_memory=True)
    
    
    
    # Query Ground Images Test
    query_dataset_test_3 = CVUSADatasetEval(data_folder=config.data_folder ,
                                          split="test",
                                          img_type="query",    
                                          transforms=ground_transforms_val_3,
                                          )
    
    query_dataloader_test_3 = DataLoader(query_dataset_test_3,
                                       batch_size=config.batch_size,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    
    print("Reference Images Test:", len(reference_dataset_test_3))
    print("Query Images Test:", len(query_dataset_test_3))

    #-----------------------------------------------------------------------------#
    # Evaluate                                                                    #
    #-----------------------------------------------------------------------------#
    
    print("\n{}[{}]{}".format(30*"-", "CVUSA", 30*"-"))  

    r1_test = evaluate_triple(config=config,
                       model_1=model_1, 
                       model_2=model_2,
                       model_3=model_3,
                       reference_dataloader1=reference_dataloader_test_1, 
                       reference_dataloader2=reference_dataloader_test_2,
                       reference_dataloader3=reference_dataloader_test_3,
                       query_dataloader1=query_dataloader_test_1, 
                       query_dataloader2=query_dataloader_test_2, 
                       query_dataloader3=query_dataloader_test_3,
                       ranks=[1, 5, 10],
                       step_size=1000,
                       cleanup=True)
