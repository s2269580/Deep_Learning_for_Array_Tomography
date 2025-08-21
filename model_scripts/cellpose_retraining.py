import numpy as np
from skimage import io
from cellpose import models, core, train
from cellpose import utils, io
from cellpose.io import logger_setup
from pathlib import Path
from tqdm import trange
import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import tifffile as tiff
from sklearn.metrics import jaccard_score, f1_score
import pandas as pd
import torch
from skimage import io as skimage_io
from scipy.ndimage import label
from skimage import io
import torch
import gc
import optuna
import argparse
import os
import json
import time
from datetime import datetime
import pandas as pd
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
import warnings
import cv2

# Suppress specific sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning) 

# Start timer for timing script run time
script_start_time = time.time()
use_GPU = core.use_gpu()
logger_setup() # run this to get printing of progress


  
def list_data_files(file_loc):
  print(file_loc)
  file_names = sorted(os.listdir(file_loc))
  file_names = [f for f in file_names if f.endswith('.tif')]
  return file_names


def normalize01(x, eps=1e-8):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + eps)   # eps avoids /0 if array is constant


def check_for_persistent_mask(test_data, img_indices, N, visualise):
  # Ensure masks are binary (0 or 1)
  binary_masks = [(test_data[m] > 0).astype(np.uint8) for m in img_indices]

  # --- Parameters ---
  height, width = binary_masks[0].shape

  # --- Initialize mask for objects present in N+ consecutive slices ---
  persistent_mask = np.zeros((height, width), dtype=np.uint8)

  # --- Loop through sliding window of N slices ---
  for i in range(len(binary_masks) - N + 1):
      # Compute intersection across N consecutive slices
      overlap = binary_masks[i].copy()
      for j in range(1, N):
          overlap = cv2.bitwise_and(overlap, binary_masks[i + j])
      # Accumulate all overlapping pixels
      persistent_mask = cv2.bitwise_or(persistent_mask, overlap)

  # --- Optional: remove small noise ---
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  persistent_mask_label = cv2.morphologyEx(persistent_mask, cv2.MORPH_OPEN, kernel)

  # --- Extract connected components ---
  num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(persistent_mask_label)

  # --- Visualize persistent objects ---
  if visualise:
    output_img = cv2.cvtColor(persistent_mask * 255, cv2.COLOR_GRAY2BGR)
    for i in range(1, num_labels):  # Skip background
        x, y, w, h, area = stats[i]
        if area > 1:  # Filter tiny components
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    print(f"Objects in {N}+ consecutive slices")
    cv2_imshow(output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  return persistent_mask_label
  
  
def create_batch_data(list_files, input_loc, gt_loc, verbose=False):
  image_names = []
  for i, file_name in enumerate(list_files):
    if verbose:
      print(i)
    input_img = tiff.imread(input_loc+file_name)
    # Create a list of 2D images from the input stack
    input_imgs_2D = [normalize01(input_img[i]).astype(float) for i in range(input_img.shape[0])]
    for j in range(len(input_imgs_2D)):
      image_names.append(list_files[i] + "_image_" + str(j))

    gt_img = tiff.imread(gt_loc+file_name)
    if 1==0 and verbose:
      plt.imshow(gt_img)
      plt.show()
    if verbose:
      print("Max before using label:",gt_img.max())

    gt_img, num_points = label(gt_img)
    if 1==0:
      plt.imshow(gt_img)
    if verbose:
      print("Number points", num_points)
      print("Max after label:", gt_img.max())
      print("\n\n")
    # Create a list of 2D ground truth images from the stack
    gt_imgs_2D = [gt_img[i].astype(float) for i in range(gt_img.shape[0])]

    if i==0:
      input_data = input_imgs_2D
      gt_data = gt_imgs_2D
    else:
      input_data+=input_imgs_2D
      gt_data+=gt_imgs_2D

  return input_data, gt_data, image_names


def calculate_metrics(seg_out, gt_out, list_files):
    results_df = pd.DataFrame(columns=[
        'File name',
        'Image number',
        'Dice Coefficient',
        'IoU (Jaccard Index)',
        'Precision',
        'Recall',
        'Model Pixels Label %',
        'Lab Pixels Label %'
    ])

    for i in range(len(seg_out)):
        maski = seg_out[i]
        gt_images_i = gt_out[i]

        # Flatten to 1D boolean arrays for sklearn
        gt_flat = gt_images_i.flatten().astype(bool)
        pred_flat = maski.flatten().astype(bool)
        fname = list_files[i]

        # Compute metrics
        dice = f1_score(gt_flat, pred_flat, zero_division=0)
        iou = jaccard_score(gt_flat, pred_flat, zero_division=0)
        precision = precision_score(gt_flat, pred_flat, zero_division=0)
        recall = recall_score(gt_flat, pred_flat, zero_division=0)

        # Pixel percentage stats
        pct = np.count_nonzero(maski) / maski.size
        lab_pct = np.count_nonzero(gt_images_i) / gt_images_i.size

        # Only append if not completely empty
        new_row_df = pd.DataFrame({
            'File name': [fname],
            'Image number': [i],
            'Dice Coefficient': [dice],
            'IoU (Jaccard Index)': [iou],
            'Precision': [precision],
            'Recall': [recall],
            'Model Pixels Label %': [pct],
            'Lab Pixels Label %': [lab_pct]
        })

        if not new_row_df.empty:
            results_df = pd.concat([results_df, new_row_df], ignore_index=True)

    return results_df
    
    
def evaluate_model(eval_data, model_path, run_mode, model_name):
    if model_path:
        print("Loading trained model")
        model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))
    else:
        print("Loading default nuclei cellpose model")
        model = models.Cellpose(gpu=True,model_type='nuclei')
        
    if model_name in ['GFAP', 'MXO4']:
        print('Running GFAP/MXO4 eval')
        masks = model.eval(eval_data, batch_size=32, max_size_fraction=0.8, flow_threshold = -.7, 
        cellprob_threshold = -0.3, resample=True, diameter=None)[0]
    elif model_name in ['PSD', 'SY38']:
        print('Running PSD/SY38 eval')
        masks = model.eval(eval_data, batch_size=32, max_size_fraction=0.5, 
        cellprob_threshold = 0.5, resample=True,  diameter=None)[0]
    return masks
    
    
    
def train_model(model_name, model_save_path, train_data, train_labels, n_epochs, best_parameters):
                    
  if best_parameters['pretrained_model_name']:
    model = models.CellposeModel(gpu=True, model_type=best_parameters['pretrained_model_name'])
  else:
    model = models.CellposeModel(gpu=True)
  
  if model_name in ['GFAP', 'MXO4']:
      new_model_path, train_losses, test_losses = train.train_seg(model.net, 
                                                                  train_data=train_data[:-50],
                                                                  train_labels=train_labels[:-50],
                                                                  test_data=train_data[-50:],
                                                                  test_labels=train_labels[-50:],
                                                                  rescale=best_parameters['rescale'],
                                                                  normalize=best_parameters['normalize'], SGD=best_parameters['SGD'],
                                                                  batch_size=best_parameters['batch_size'],
                                                                  n_epochs=n_epochs,
                                                                  learning_rate=best_parameters['learning_rate'],
                                                                  weight_decay=best_parameters['weight_decay'],
                                                                  model_name=model_save_path,
                                                                  min_train_masks=0,
                                                                  nimg_per_epoch=40,
                                                                  save_path="/content/drive/MyDrive/Colab Notebooks/Segmentation_Pipeline/")
  else:
      new_model_path, train_losses, test_losses = train.train_seg(model.net, 
                                                                  train_data=train_data[:-50],
                                                                  train_labels=train_labels[:-50],
                                                                  test_data=train_data[-50:],
                                                                  test_labels=train_labels[-50:],
                                                                  rescale=best_parameters['rescale'],
                                                                  normalize=best_parameters['normalize'], SGD=best_parameters['SGD'],
                                                                  batch_size=best_parameters['batch_size'],
                                                                  n_epochs=n_epochs,
                                                                  channels = [0,0], 
                                                                  learning_rate=best_parameters['learning_rate'],
                                                                  weight_decay=best_parameters['weight_decay'],
                                                                  model_name=model_save_path,
                                                                  min_train_masks=0,
                                                                  nimg_per_epoch=40,
                                                                  save_path="/content/drive/MyDrive/Colab Notebooks/Segmentation_Pipeline/")
  return new_model_path, train_losses, test_losses




def make_objective(model_name, tune_input, tune_labels):
    def objective(trial):
        try:

            if model_name == 'GFAP':
                learning_rate = trial.suggest_float("learning_rate", 0.00005, 0.0005)
                weight_decay = trial.suggest_float("weight_decay", 0.00005, 0.01)
                batch_size = trial.suggest_categorical("batch_size",[2])
                rescale = trial.suggest_categorical("rescale", [True])
                normalize = trial.suggest_categorical("normalize", [True, False])
                SGD = trial.suggest_categorical("SGD", [True, False])
                pretrained_model_name = trial.suggest_categorical("pretrained_model_name", ['nuclei', 'cyto3', ''])
            elif model_name == 'MXO4':
                learning_rate = trial.suggest_float("learning_rate", 0.00005, 0.0005)
                weight_decay = trial.suggest_float("weight_decay", 0.00005, 0.01)
                batch_size = trial.suggest_categorical("batch_size",[2])
                rescale = trial.suggest_categorical("rescale", [False])
                normalize = trial.suggest_categorical("normalize", [True])
                SGD = trial.suggest_categorical("SGD", [False])
                pretrained_model_name = trial.suggest_categorical("pretrained_model_name", [''])
            elif model_name == 'SY38':
                learning_rate = trial.suggest_float("learning_rate", 0.00005, 0.05)
                weight_decay = trial.suggest_float("weight_decay", 0.00005, 0.05)
                batch_size = trial.suggest_categorical("batch_size",[2])
                rescale = trial.suggest_categorical("rescale", [True])
                normalize = trial.suggest_categorical("normalize", [False])
                SGD = trial.suggest_categorical("SGD", [False])
                pretrained_model_name = trial.suggest_categorical("pretrained_model_name", [None, 'nuclei'])
            elif model_name == 'PSD':
                learning_rate = trial.suggest_float("learning_rate", 0.00005, 0.05)
                weight_decay = trial.suggest_float("weight_decay", 0.00005, 0.05)
                batch_size = trial.suggest_categorical("batch_size",[2])
                rescale = trial.suggest_categorical("rescale", [True])
                normalize = trial.suggest_categorical("normalize", [False])
                SGD = trial.suggest_categorical("SGD", [False])
                pretrained_model_name = trial.suggest_categorical("pretrained_model_name", [None, 'nuclei'])
                

            # Reinitialize model fresh for each trial
            if pretrained_model_name:
              model = models.CellposeModel(gpu=True, model_type=pretrained_model_name)
            else:
              model = models.CellposeModel(gpu=True)
            net = model.net
    
            # Run training
            if model_name in ['GFAP', 'MXO4']:
                    new_model_path, train_losses, test_losses = train.train_seg(
                        net,
                        train_data=tune_input[:-50], 
                        train_labels=tune_labels[:-50],
                        test_data=tune_input[-50:],
                        test_labels=tune_labels[-50:],
                        batch_size=batch_size,
                        n_epochs=50,  # lower for tuning
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        rescale=rescale,
                        normalize=normalize,
                        SGD=SGD,
                        model_name="optuna_trial_model",
                        min_train_masks=0,
                        nimg_per_epoch=20
                    )
            else:
                new_model_path, train_losses, test_losses = train.train_seg(
                    net,
                    train_data=tune_input[:-50], 
                    train_labels=tune_labels[:-50],
                    test_data=tune_input[-50:],
                    test_labels=tune_labels[-50:],
                    channels = [0,0], 
                    batch_size=batch_size,
                    n_epochs=50,  # lower for tuning
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    rescale=rescale,
                    normalize=normalize,
                    SGD=SGD,
                    model_name="optuna_trial_model",
                    min_train_masks=0,
                    nimg_per_epoch=20
                )
    
            # Clear GPU memory after training
            del net
            torch.cuda.empty_cache()
            gc.collect()
            return train_losses[-1] 
    
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            raise optuna.exceptions.TrialPruned()  # skip the trial
        
    return objective


def read_and_format_files(input_folders, gt_folders):
    list_files = list_data_files(input_folders)
    input_np, gt_np, image_names = create_batch_data(list_files, input_folders, gt_folders)
    return input_np, gt_np, image_names
    
    
def choose_tune_train_test_batch(train_test_split, train_test_mode, input_folders_1, gt_folders_1, input_folders_2, gt_folders_2):
    
    input_np_1, gt_np_1, image_names_1 = read_and_format_files(input_folders_1, gt_folders_1)
    full_input, full_labels = input_np_1, gt_np_1
    slice_pos = int(len(full_input)*train_test_split)
    
    if train_test_mode == 'train_test_same_batch':
        train_input =  full_input[:slice_pos]
        test_input =  full_input[slice_pos:]
        train_image_names = image_names_1[:slice_pos]
        test_image_names = image_names_1[slice_pos:]
      
        train_labels =  full_labels[:slice_pos]
        test_labels =  full_labels[slice_pos:]
        
    else:
        input_np_2, gt_np_2, image_names_2 = read_and_format_files(input_folders_2, gt_folders_2)
        if train_test_mode == 'train_test_diff_batch':
            train_input, train_labels = input_np_1, gt_np_1
            test_input, test_labels = input_np_2, gt_np_2
            train_image_names = image_names_1
            test_image_names = image_names_2
            
        elif train_test_mode == 'train_test_mixed_batch':
            slice_pos_2 = int(len(input_np_2)*train_test_split)
            train_input = input_np_1[:slice_pos]+input_np_2[:slice_pos_2]
            train_labels = gt_np_1[:slice_pos]+gt_np_2[:slice_pos_2]
            test_input = input_np_1[slice_pos:]+input_np_2[slice_pos_2:]
            test_labels = gt_np_1[slice_pos:]+gt_np_2[slice_pos_2:]
            train_image_names = image_names_1[:slice_pos]+image_names_2[:slice_pos_2]
            test_image_names = image_names_1[slice_pos:]+image_names_2[slice_pos_2:]
            
    return train_input, train_labels, test_input, test_labels, train_image_names, test_image_names
    
    
def choose_indices(list_imgs, prop):
  all_indices = range(len(list_imgs))
  chosen_indices = np.random.choice(all_indices, int(len(list_imgs)*prop), replace=False)
  return chosen_indices
  
  
def create_tune_data(tune_split, tune_indice_prop, train_indice_prop, test_indice_prop, 
                        train_input, train_labels,test_input, test_labels, test_image_names):
    slice_pos = int(len(train_input)*tune_split)

    tune_input_all = train_input[:slice_pos]
    train_input_all = train_input[slice_pos:]
    tune_labels_all = train_labels[:slice_pos]
    train_labels_all = train_labels[slice_pos:]

    tune_indices = choose_indices(tune_input_all, tune_indice_prop)
    tune_input = [tune_input_all[i] for i in tune_indices]
    tune_labels = [tune_labels_all[i] for i in tune_indices]

    train_indices = choose_indices(train_input_all, train_indice_prop)
    train_input = [train_input_all[i] for i in train_indices]
    train_labels = [train_labels_all[i] for i in train_indices]
    
    test_slice_pos = int(len(test_input)*test_indice_prop)

    test_input = test_input[:test_slice_pos]
    test_labels = test_labels[:test_slice_pos]
    test_image_names = test_image_names[:test_slice_pos]

    return tune_input, tune_labels, train_input, train_labels, test_input, test_labels, test_image_names
    
    
def run_hyperparameter_tuning(model_name, num_tune_trials, tune_input, tune_labels):
    study = optuna.create_study(direction="minimize")
    study.optimize(make_objective(model_name, tune_input, tune_labels), n_trials=num_tune_trials, gc_after_trial=True)
    
    print("Best trial:")
    trial = study.best_trial
    
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    return trial.params
    
    
def plot_overlap(base_img, gt_out, seg_out, title, image_save_dir, save_path):
        
      # Normalize grayscale base image to [0, 1]
      base_img = base_img.astype(np.float32)
      base_img = (base_img - base_img.min()) / (base_img.max() - base_img.min())
      # Convert base image to RGB
      if base_img.ndim == 2:
        rgb = np.stack([base_img] * 3, axis=-1)
      else:
        rgb = base_img # If it's already RGB
      
      # Boolean masks
      cellpose_mask = seg_out.astype(bool)
      gt_mask = gt_out.astype(bool)

      # Calculate regions
      only_cellpose = cellpose_mask & ~gt_mask
      only_gt = gt_mask & ~cellpose_mask
      overlap = cellpose_mask & gt_mask

      # Overlay RGB
      overlay = rgb.copy()

      # Cellpose only – Red
      overlay[only_cellpose] = [1, 0, 0]

      # Ground truth only – Green
      overlay[only_gt] = [0, 1, 0]

      # Overlap – Blue (or pick another standout color like white or magenta)
      overlay[overlap] = [0, 0, 1]

      # Plot
      plt.figure(figsize=(10, 6))
      plt.imshow(overlay)
      plt.title(f"{title}:\nRed: Cellpose | Green: Ground Truth | Blue: Overlap")
      plt.axis('off')
      plt.savefig(image_save_dir + save_path)
      
      
def plot_multiple_views(seg_list, gt_list, img_list, img_names, mask_indices, title, image_save_dir, save_path):
    """
    Plots multiple rows of four-view segmentation comparisons.
    Each row: [Original, Cellpose Mask, GT Mask, Overlap]
    """
    n = len(mask_indices)
    fig, axs = plt.subplots(n, 4, figsize=(20, 5 * n))

    if n == 1:
        axs = np.expand_dims(axs, 0)  # ensure 2D indexable if n=1

    for i, m  in enumerate(mask_indices):
        seg_out = seg_list[m].astype(bool)
        gt_out = gt_list[m].astype(bool)
        base_img = img_list[m].astype(np.float32)
        base_img = (base_img - base_img.min()) / (base_img.max() - base_img.min())
        rgb = np.stack([base_img] * 3, axis=-1)

        # Create overlays
        img_orig = rgb.copy()

        img_mask = rgb.copy()
        img_mask[seg_out] = [1, 0, 0]  # Red

        img_gt = rgb.copy()
        img_gt[gt_out] = [0, 1, 0]  # Green

        only_mask = seg_out & ~gt_out
        only_gt = gt_out & ~seg_out
        overlap = seg_out & gt_out
        img_overlap = rgb.copy()
        img_overlap[only_mask] = [1, 0, 0]
        img_overlap[only_gt] = [0, 1, 0]
        img_overlap[overlap] = [0, 0, 1]
        
        # Plot each view in row i
        axs[i, 0].imshow(img_orig)
        axs[i, 0].set_title("Original")

        axs[i, 1].imshow(img_mask)
        axs[i, 1].set_title("ML Mask")

        axs[i, 2].imshow(img_gt)
        axs[i, 2].set_title("Ground Truth Mask")

        axs[i, 3].imshow(img_overlap)
        axs[i, 3].set_title("Overlap\nRed: CP | Green: GT | Blue: Both")

        for j in range(len(mask_indices)):
            axs[i, j].axis('off')

    # Main title
    plt.suptitle(title, fontsize=18, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.axis('off')
    plt.savefig(image_save_dir + save_path)
      
      
    
def choose_masks_to_visualise(df):
    df["high_performance_score"] = df["Dice Coefficient"]*(df["Lab Pixels Label %"]/abs(df["Model Pixels Label %"]-df["Lab Pixels Label %"]))
    high_performing_masks = df.sort_values(by=["Dice Coefficient"], ascending=False)["Image number"].head(3).tolist()
    print(df.sort_values(by=["Dice Coefficient"], ascending=False).head(3))
        
    df["low_performance_score"] = np.where(df["Dice Coefficient"] != 0, 1- df["Dice Coefficient"], 0)
    low_performing_masks = df.sort_values(by=["low_performance_score"], ascending=False)["Image number"].head(3).tolist()
    print(df.sort_values(by=["low_performance_score"], ascending=True).head(3))
    return high_performing_masks, low_performing_masks
    
    
def save_model_details(model_params, details_path, train_test_split, tune_split, tune_indice_prop, 
                        train_indice_prop, num_tune_trials):
    # End timer
    script_end_time = time.time()
    total_runtime = script_end_time - script_start_time

    # Add runtime and date
    model_params["tune_split"] = tune_split
    model_params["tune_indice_prop"] = tune_indice_prop
    model_params["train_test_split"] = train_test_split
    model_params["train_indice_prop"] = train_indice_prop
    model_params["num_tune_trials"] = num_tune_trials
    model_params["total_runtime_seconds"] = total_runtime
    model_params["run_datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(details_path, "w") as f:
        json.dump(model_params, f, indent=4)
    
    print(f"Model details saved to: {details_path}")
    
    
def main(args):
    print("Create train and test data by reading and formatting input files and applying train-test split\n")
    train_input, train_labels, test_input, test_labels, train_image_names, test_image_names = (choose_tune_train_test_batch(
        args.train_test_split, args.train_test_mode, args.input_folders_1, args.gt_folders_1, args.input_folders_2, args.gt_folders_2))
        
    if args.run_mode != 'eval':
        print("\nCreate tune data\n")
        tune_input, tune_labels, train_input, train_labels, test_input, test_labels, test_image_names = create_tune_data(args.tune_split, args.tune_indice_prop, args.train_indice_prop, args.test_indice_prop, train_input, train_labels, test_input, test_labels, test_image_names)
        if args.run_mode == 'tune':
            print("\nRun hyperparameter tuning\n")
            best_parameters = run_hyperparameter_tuning(args.model_name, args.num_tune_trials, tune_input, tune_labels)
        else:
            best_parameters = {"pretrained_model_name":"nuclei", "learning_rate":0.00003,
            "weight_decay":0.00164,"batch_size":2, "SGD": False, "rescale":True, "normalize":True}
        print("\nRun training\n")
        model_save_path = "Cellpose/" + args.model_name + '_' + args.train_test_mode + '_' + str(args.num_epoch) + '_' + str(args.run_mode)
        model_path, train_losses, test_losses = train_model(args.model_name, model_save_path, train_data=train_input, 
        train_labels=train_labels, n_epochs=args.num_epoch, best_parameters=best_parameters)
        print("Model path saved from training")
        print(model_path)

        print("Save model details")
        details_save_path = os.path.join("/content/drive/MyDrive/Colab Notebooks/Segmentation_Pipeline/results/Model_details/", f"Cellpose_{args.model_name}_{args.train_test_mode}_{args.num_epoch}_{args.run_mode}.json")
        save_model_details(best_parameters, details_save_path, args.train_test_split, args.tune_split, args.tune_indice_prop, 
                            args.train_indice_prop, args.num_tune_trials)
        
    elif args.run_mode == 'eval':
        print("\nSkipping training and loading pretrained model")
        if args.pretrained_model_path:
            print(f"Loading model {args.pretrained_model_path}")
        else: 
            print("Running Nuclei model")
        model_path = args.pretrained_model_path
        print(model_path)
        
    print("\nRun evaluation\n")
    masks = evaluate_model(test_input, model_path, args.run_mode, args.model_name)
    
    print("\nDetermine resulting performance\n")
    results_df = calculate_metrics(masks, test_labels, test_image_names)
    results_df.to_csv("/content/drive/MyDrive/Colab Notebooks/Segmentation_Pipeline/results/Results_files/Cellpose/"
    +args.model_name+ '/' + args.train_test_mode + '_' + str(args.num_epoch) + '_' + str(args.run_mode) + '_results.csv')
    
    print("\nSave example masks\n")
    high_performing_masks, low_performing_masks = choose_masks_to_visualise(results_df)
    print("HPM:", high_performing_masks)
    print("LPM:", low_performing_masks)
    
    print("\nCreate image directory if it doesn't already exist")
    image_save_dir = os.path.join("/content/drive/MyDrive/Colab Notebooks/Segmentation_Pipeline/results/Images/Cellpose/", args.model_name)
    os.makedirs(image_save_dir, exist_ok=True)

                     
    plot_multiple_views(masks, test_labels, test_input, test_image_names, high_performing_masks,
    f"High Performing Mask- {args.model_name}", image_save_dir, "/"+args.model_name+ '_' + args.train_test_mode + '_' 
    + str(args.num_epoch) + '_' + str(args.run_mode) + "_hpm_mask_subset.jpeg")
    
    plot_multiple_views(masks, test_labels, test_input, test_image_names, low_performing_masks,
    f"Low Performing Mask- {args.model_name}", image_save_dir, "/"+args.model_name+ '_' + args.train_test_mode + '_' 
    + str(args.num_epoch) + '_' + str(args.run_mode) + "_lpm_mask_subset.jpeg")
    
    print('Run overal post-processing and save summary results')
    file_names = [s.split('.')[0] for s in test_image_names]
    persistent_lab_masks = []
    persistent_model_masks = []
    for fn in list(set(file_names)):
      img_indices = [i for i, s in enumerate(file_names) if s == fn]
      lab_pers = check_for_persistent_mask(test_labels, img_indices, N=2, visualise=False)
      persistent_lab_masks.append(lab_pers)
    
      masks_pers = check_for_persistent_mask(masks, img_indices, N=2, visualise=False)
      persistent_model_masks.append(masks_pers)
    
    results_df_summarised = calculate_metrics(persistent_model_masks, persistent_lab_masks, list(set(file_names)))
    results_df_summarised.to_csv("/content/drive/MyDrive/Colab Notebooks/Segmentation_Pipeline/results/Results_files/Cellpose/"
        +args.model_name+ '/' + args.train_test_mode + '_' + str(args.num_epoch) + '_' + str(args.run_mode) + '_summarised_results.csv')
    
    return None



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Cellpose model")
    parser.add_argument('-mn', '--model_name', type=str, default='SY38', help='Name of stain')
    parser.add_argument('-tts', '--train_test_split', type=float, default=0.8, help='Proportion of data in train vs test set')
    parser.add_argument('-mode', '--train_test_mode', type=str, default='train_test_same_batch', help='Batch running mode')
    parser.add_argument('-input1', '--input_folders_1', type=str,default=None, help='Path to first set of input images')
    parser.add_argument('-gt1', '--gt_folders_1', type=str, default=None, help='Path to first set of lab masks')
    parser.add_argument('-input2', '--input_folders_2', type=str, default=None, help='Path to second set of input images')
    parser.add_argument('-gt2', '--gt_folders_2', type=str, default=None, help='Path to second set of lab masks')
    parser.add_argument('-ts', '--tune_split', type=float, default=0.3, help='Proportion of data from train ste reserved for tuning')
    parser.add_argument('-tunep', '--tune_indice_prop', type=float, default=1, help='Proportion of tuning data to be kept')
    parser.add_argument('-trainp', '--train_indice_prop', type=float, default=1, help='Proportion of training data to be kept')
    parser.add_argument('-testp', '--test_indice_prop', type=float, default=1, help='Proportion of test data to be kept')
    parser.add_argument('-numt', '--num_tune_trials', type=int, default=1, help='Number of tuning trials to run')
    parser.add_argument('-nume', '--num_epoch', type=int, default=100, help='Number of model epochs')
    parser.add_argument('-run_mode', '--run_mode', type=str, default='eval', help='Mode for running e.g. train/tune/eval')
    parser.add_argument('-model_path', '--pretrained_model_path', type=str, default=None, help='Path to saved model')
    
    args = parser.parse_args()
    print(args)
    main(args)
  
  
  
