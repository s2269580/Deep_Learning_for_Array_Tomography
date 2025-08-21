import torch
import torch.nn.utils
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
from scipy.ndimage import label
import tifffile as tiff
import argparse
import optuna
import gc
import os
import json
import time
from datetime import datetime
import warnings
import cv2
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
import pandas as pd

# Suppress specific sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Optional: for cleaner output

# Start timer at the very beginning of script
script_start_time = time.time()

def list_data_files(file_loc):
  print(file_loc)
  file_names = sorted(os.listdir(file_loc))
  file_names = [f for f in file_names if f.endswith('.tif')]
  return file_names
  
  
def read_and_format_files(input_folders, gt_folders):
    list_files = list_data_files(input_folders)
    input_np, gt_np, image_names = create_batch_data(list_files, input_folders, gt_folders)
    return input_np, gt_np, image_names


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
    gt_imgs_2D = [normalize01(gt_img[i]).astype(float) for i in range(gt_img.shape[0])]

    if i==0:
      input_data = input_imgs_2D
      gt_data = gt_imgs_2D
    else:
      input_data+=input_imgs_2D
      gt_data+=gt_imgs_2D

  return input_data, gt_data, image_names
    
    
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


def normalize01(x, eps=1e-8):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + eps)   # eps avoids /0 if array is constant
    
    
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


def get_points(mask, num_points):  # Sample points inside the input mask
   points = []
   coords = np.argwhere(mask > 0)
   if len(coords) == 0:  # Handle case where mask is empty
       return np.empty((0, 1, 2), dtype=np.int64) # Return empty array with expected shape

   # Ensure we don't sample more points than available coordinates
   num_points_to_sample = min(len(coords), num_points)

   for i in range(num_points_to_sample):
       yx = np.array(coords[np.random.randint(len(coords))])
       points.append([[yx[1], yx[0]]])
   return np.array(points)

    
def process_slice_new(input_np, labels_np, visualize_data=False):
    select_slice = np.random.randint(len(input_np))
    # Get full paths
    # Read a single slice from the tif stack
    Img_slice = input_np[select_slice].astype(np.float32)
    

    # Check the number of dimensions in the slice
    if Img_slice.ndim == 2:
        # If it's a grayscale image (H, W), convert to 3 channels (H, W, 3) by stacking the channel
        Img = np.stack([Img_slice] * 3, axis=-1)
        # Find the minimum and maximum pixel values in your image
        min_val = Img.min()
        max_val = Img.max()

        # Check if the range is already 0-255
        if min_val >= 0 and max_val <= 255 and Img.dtype == np.uint8:
            if visualize_data:
                print("Image is already in the 0-255 range and uint8 format.")
        else:
            # Normalize the data to the 0-1 range
            normalized_image = (Img - min_val) / (max_val - min_val)

            # Scale the normalized data to the 0-255 range
            scaled_image = normalized_image * 255

            # Convert the scaled data to unsigned 8-bit integers (uint8)
            Img = scaled_image.astype(np.uint8)

            if visualize_data:
              print(f"Image converted from range [{min_val}..{max_val}] to [0..255].")

    # Read annotation as grayscale
    ann_map =   labels_np[select_slice].astype(np.float32)
    ann_map, num_objects = label(ann_map)
    # print() # Removing print for cleaner output during training

    # Resize image and mask
    r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])  # Scaling factor
    # Resize image and ensure it stays float32
    img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)

    # Initialize a single binary mask
    binary_mask = np.zeros_like(ann_map, dtype=np.uint8)
    points = []

    # Get binary masks and combine them into a single mask
    inds = np.unique(ann_map)[1:]  # Skip the background (index 0)
    for ind in inds:
        mask = (ann_map == ind).astype(np.uint8)  # Create binary mask for each unique index
        binary_mask = np.maximum(binary_mask, mask)  # Combine with the existing binary mask

    # Erode the combined binary mask to avoid boundary points
    eroded_mask = cv2.erode(binary_mask, np.ones((2, 2), np.uint8), iterations=1)
    # print(eroded_mask) # Removing print for cleaner output during training

    # Get all coordinates inside the eroded mask and choose a random point
    coords = np.argwhere(eroded_mask > 0)
    if len(coords) > 0:
        # Ensure we don't sample more points than available coordinates
        num_points_to_sample = min(len(coords), len(inds))
        for _ in range(num_points_to_sample):
           # Select as many points as there are unique labels (up to the number of available coords)
           yx = np.array(coords[np.random.randint(len(coords))])
           points.append([yx[1], yx[0]])

    points = np.array(points)


    if visualize_data:
        # Plotting the images and points
        plt.figure(figsize=(15, 5))

        # Original Image
        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(img.astype(np.uint8))
        plt.axis('off')

        # Segmentation Mask (binary_mask)
        plt.subplot(1, 3, 2)
        plt.title('Binarized Mask')
        plt.imshow(binary_mask, cmap='gray')
        plt.axis('off')

        # Mask with Points in Different Colors
        plt.subplot(1, 3, 3)
        plt.title('Binarized Mask with Points')
        plt.imshow(binary_mask, cmap='gray')

        # Plot points in different colors
        colors = list(mcolors.TABLEAU_COLORS.values())
        # Ensure points array is not empty before plotting
        if points.size > 0:
            for i, point in enumerate(points):
                plt.scatter(point[0], point[1], c=colors[i % len(colors)], s=100, label=f'Point {i+1}')  # Corrected to plot y, x order

        # plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    binary_mask = np.expand_dims(binary_mask, axis=-1)  # Now shape is (H, W, 1)
    binary_mask = binary_mask.transpose((2, 0, 1)) # Transpose to (1, H, W)
    # Ensure points is not empty before expanding dims
    if points.size > 0:
        points = np.expand_dims(points, axis=1)
    else:
        # If no points are found, return empty points array with correct shape for consistency
        points = np.empty((0, 1, 2), dtype=np.int64)

    # Return the image (now 3-channel), binarized mask, points, and number of masks
    return img, binary_mask, points, len(inds)


    
def train_sam2(predictor, optimizer, scheduler, scaler, fine_tuned_model_name, input_data, labels_data, no_of_steps=1000, accumulation_steps=4, max_prompts=40, visualize_training=False):
    mean_iou = 0
    for step in range(1, no_of_steps + 1):
        with torch.amp.autocast(device_type='cuda'):
            image, mask, input_point, num_masks =  process_slice_new(input_data, labels_data, visualize_data=visualize_training)

            if image is None or mask is None or num_masks == 0:
                # print('skip 1')
                continue

            input_label = np.ones((num_masks, 1)) # Shape (N_points, 1)
            if not isinstance(input_point, np.ndarray) or not isinstance(input_label, np.ndarray):
                # print('skip 2')
                continue

            if input_point.size == 0 or input_label.size == 0:
                # print('skip 3')
                continue

            # Sample max_prompts if num_masks is larger
            if num_masks > max_prompts:
                indices = np.random.choice(num_masks, max_prompts, replace=False)
                input_point = input_point[indices] # Shape (max_prompts, 1, 2)
                input_label = input_label[indices] # Shape (max_prompts, 1)
                # Ensure input_label is shape (max_prompts, 1) - remove potential extra dim if added previously
                if input_label.ndim == 2 and input_label.shape[1] != 1:
                     input_label = input_label.reshape(-1, 1)
                elif input_label.ndim == 3 and input_label.shape[2] == 1:
                     input_label = input_label.squeeze(axis=2) # Remove the last dimension
                # input_label = input_label[:, np.newaxis] # Remove this potentially problematic line
                num_masks = max_prompts # Update num_masks to the sampled count

            # print(f"Shapes before _prep_prompts: input_point.shape={input_point.shape}, input_label.shape={input_label.shape}")

            predictor.set_image(image)
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)

            # print(f"Shapes after _prep_prompts: unnorm_coords.shape={unnorm_coords.shape if unnorm_coords is not None else 'None'}, labels.shape={labels.shape if labels is not None else 'None'}")

            if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
                # print('skip 4')
                continue

            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels), boxes=None, masks=None,
            )

            batched_mode = unnorm_coords.shape[0] > 1
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]

            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
            )
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks[:, 0])
            seg_loss = (-gt_mask * torch.log(prd_mask + 0.000001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()

            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = seg_loss + score_loss * 0.05

            loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)

            if step % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                predictor.model.zero_grad()

            scheduler.step()

            if step % 100 == 0:
                FINE_TUNED_MODEL = fine_tuned_model_name + "_" + str(step) + ".torch"
                torch.save(predictor.model.state_dict(), FINE_TUNED_MODEL)
                highest_epoch_model = FINE_TUNED_MODEL

            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())

            if visualize_training and step % 100 == 0:
                print("Step " + str(step) + ":\t", "Accuracy (IoU) = ", mean_iou)
                print(f"Visualizing results for step {step}")
                plt.figure(figsize=(15, 5))

                plt.subplot(1, 3, 1)
                plt.title('Input Image')
                if image.dtype == np.float32:
                    plt.imshow(image.astype(np.uint8))
                else:
                    plt.imshow(image)
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.title('Ground Truth Mask')
                plt.imshow(gt_mask.squeeze().cpu().numpy(), cmap='gray')
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.title('Predicted Mask (Thresholded)')
                if prd_masks.shape[0] > 0:
                    plt.imshow((torch.sigmoid(prd_masks[0, 0]) > 0.5).squeeze().cpu().numpy(), cmap='gray')
                else:
                    print("No predicted masks to visualize.")
                plt.axis('off')

                plt.tight_layout()
                plt.show()
            elif step % 100 == 0:
                 print("Step " + str(step) + ":\t", "Accuracy (IoU) = ", mean_iou)

    return mean_iou, FINE_TUNED_MODEL # Return the final mean IoU
    

def make_objective(train_input, train_labels, model_cfg, sam2_checkpoint):
    def objective(trial):
        try:
            # Suggest hyperparameters for SAM2 training
            learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
            step_size = trial.suggest_int("step_size", 50, 500, step=50)
            gamma = trial.suggest_float("gamma", 0.01, 0.9)
            accumulation_steps = trial.suggest_categorical("accumulation_steps", [2, 4, 8])
            max_prompts = trial.suggest_int("max_prompts", 5, 100, step=5)
            no_of_steps = 600 # Increased steps for tuning

            # Build SAM2 model
            sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
            predictor = SAM2ImagePredictor(sam2_model)

            # Configure optimizer, scheduler, and scaler
            optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            scaler = torch.cuda.amp.GradScaler()

            # Train the model using the modified train_sam2 function
            # Pass train_input and train_labels directly to train_sam2
            # Disable visualization during tuning
            final_iou, FINE_TUNED_MODEL = train_sam2(
                predictor=predictor,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                fine_tuned_model_name=f"optuna_sam2_trial_{trial.number}",
                input_data=train_input,
                labels_data=train_labels,
                no_of_steps=no_of_steps,
                accumulation_steps=accumulation_steps,
                max_prompts=max_prompts,
                visualize_training=False # Disable visualization
            )

            # Clean up GPU memory after training
            del predictor
            del sam2_model
            torch.cuda.empty_cache()
            gc.collect()

            # Optuna minimizes the objective, so return the negative of the IoU
            return -final_iou

        except torch.cuda.OutOfMemoryError:
            # Clean up before returning failure
            torch.cuda.empty_cache()
            gc.collect()
            raise optuna.exceptions.TrialPruned()  # skip the trial
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            # Clean up before returning failure
            torch.cuda.empty_cache()
            gc.collect()
            raise optuna.exceptions.TrialPruned() # Prune trial for other errors as well

    return objective


# The run_hyperparameter_tuning function will call make_objective
def run_hyperparameter_tuning(num_tune_trials, train_input, train_labels,  model_cfg, sam2_checkpoint):
    study = optuna.create_study(direction="minimize")
    # Pass relevant data and model config to make_objective
    objective = make_objective(train_input, train_labels, model_cfg, sam2_checkpoint)
    study.optimize(objective, n_trials=num_tune_trials, gc_after_trial=True)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return trial.params           


def training_step(train_input, train_labels, sam2_checkpoint, model_cfg, best_params, NO_OF_STEPS, FINE_TUNED_MODEL_NAME):
    
    # Build SAM2 model
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
    
    # Train mask decoder.
    predictor.model.sam_mask_decoder.train(True)
    
    # Train prompt encoder.
    predictor.model.sam_prompt_encoder.train(True)
    
    # Configure optimizer.
    optimizer=torch.optim.AdamW(params=predictor.model.parameters(),lr=best_params['learning_rate'],weight_decay=best_params['weight_decay'])
    
    # Mix precision.
    scaler = torch.cuda.amp.GradScaler()

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=best_params['step_size'], gamma=best_params['gamma'])
    accumulation_steps = best_params['accumulation_steps']
    max_prompts = best_params['max_prompts']
    
    # Train the model using the full training data
    print(f"Starting final training for {NO_OF_STEPS} steps...")
    final_train_iou, FINE_TUNED_MODEL = train_sam2(
        predictor=predictor,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        fine_tuned_model_name=FINE_TUNED_MODEL_NAME,
        input_data=train_input, # Use the full train_input
        labels_data=train_labels, # Use the full train_labels
        no_of_steps=NO_OF_STEPS,
        accumulation_steps=accumulation_steps,
        max_prompts=max_prompts,
        visualize_training=False # Keep visualization enabled for final training if desired
    )
    print(f"\nFinal training finished with mean IoU: {final_train_iou}")
    return FINE_TUNED_MODEL
    

def eval_step(test_input, test_labels, full_model_name, model_cfg, sam2_checkpoint):
    # Use the trained model to predict on the test data
    # Use test_input and test_labels directly
    # The predict_batch function is modified to accept these lists of slices
    segmented_results = predict_batch_2(test_input, test_labels, full_model_name, model_cfg, sam2_checkpoint)
    
    original_predicted_masks = []
    original_binary_masks = []
    for slice_idx, seg_map in enumerate(segmented_results):
        original_mask_slice = test_labels[slice_idx]
        original_binary_mask = (original_mask_slice > 0).astype(np.uint8) * 255
        predicted_binary_mask = (seg_map > 0).astype(np.uint8)
        original_binary_masks.append(original_binary_mask)
        # Get the corresponding original image and mask from test_input and test_labels
        original_image_slice = test_input[slice_idx]
        # original_mask_slice = test_labels[slice_idx] # This is the labeled mask (float)
    
        # Convert original_image_slice to uint8 for display (assuming it's float [0,1])
        if original_image_slice.ndim == 2:
             original_image_display = np.stack([original_image_slice] * 3, axis=-1)
        else:
             original_image_display = original_image_slice # Assuming it's already 3 channels
        original_image_display = (original_image_display * 255).astype(np.uint8)
    
    
        # Convert original_mask_slice to a binary mask for display (assuming it's labeled float)
        original_binary_mask_display = (original_mask_slice > 0).astype(np.uint8) 
    
        # Prepare data for metric calculation
        # Convert seg_map to binary mask (if it's labeled)
        predicted_binary_mask = (seg_map > 0).astype(np.uint8)
    
        # The resizing logic for metric calculation is removed to restore previous state
        if predicted_binary_mask.shape != original_binary_mask_display.shape:
            print(f"Warning: Predicted mask shape {predicted_binary_mask.shape} does not match ground truth shape {original_binary_mask_display.shape} for slice {slice_idx}. Resizing predicted mask.")
            predicted_binary_mask_resized = cv2.resize(predicted_binary_mask,
                                                     (original_binary_mask_display.shape[1], original_binary_mask_display.shape[0]),
                                                     interpolation=cv2.INTER_NEAREST)
            print(f"New shapes following resizing: Predicted mask shape {predicted_binary_mask.shape}, ground truth shape {original_binary_mask_display.shape} for slice {slice_idx}.")
        else:
            predicted_binary_mask_resized = predicted_binary_mask
        original_predicted_masks.append(predicted_binary_mask_resized)
    
        # Calculate metrics for this slice (optional - you can calculate after the loop for all slices)
        # Use the original predicted_binary_mask and original_binary_mask_display for potential error reproduction
        flattened_predicted = predicted_binary_mask_resized.flatten()
        flattened_ground_truth = original_binary_mask_display.flatten()

        if flattened_predicted.max() == 0:
            print("No predicted stains")
            dice = 0
            iou = 0
        else:
            dice = f1_score(flattened_ground_truth, flattened_predicted)
            iou = jaccard_score(flattened_ground_truth, flattened_predicted)
        print(f"Slice {slice_idx} Metrics: Dice={dice:.4f}, IoU={iou:.4f}")
    return original_predicted_masks, original_binary_masks, seg_map, original_binary_mask_display, original_image_display


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

        # Compute metrics safely
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
        # axs[i, 0].set_ylabel(f"Image: {img_names[m]}", fontsize=14, labelpad=10)

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



def predict_batch_2(image_list, mask_list, full_model_name, model_cfg, sam2_checkpoint, device="cuda", num_samples=30):
    # Load the fine-tuned model
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

    # Build net and load weights
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.model.load_state_dict(torch.load(full_model_name))

    segmented_images = []

    for idx, (image_slice, mask_slice) in enumerate(zip(image_list, mask_list)):
        # Generate random points for the input
        # Need to convert the mask_slice back to uint8 if it's not already
        input_points = get_points((mask_slice * 255).astype(np.uint8), num_samples)

        # Skip prediction if no input points are generated
        if input_points.shape[0] == 0:
            # print(f"Skipping prediction for slice {idx} with no input points.")
            # Append an empty segmentation map for consistency
            # The shape should match the expected output segmentation map shape after resizing
            # We can get this from the resized image shape
            # r = np.min([1024 / image_slice.shape[1], 1024 / image_slice.shape[0]])
            resized_shape = (int(image_slice.shape[0]), int(image_slice.shape[1]))
            segmented_images.append(np.zeros(resized_shape, dtype=np.uint8))
            continue

        # Perform inference and predict masks
        with torch.no_grad():
            # Need to ensure image_slice is in the correct format for predictor
            # Assuming image_slice is already normalized and in float format from process_slice
            # Convert to uint8 (0-255) and ensure it's 3 channels
            if image_slice.ndim == 2:
                 Img_for_predictor = np.stack([image_slice] * 3, axis=-1)
            else:
                 Img_for_predictor = image_slice # Assuming it's already 3 channels
            # Scale float [0,1] to uint8 [0,255]
            Img_for_predictor = (Img_for_predictor * 255).astype(np.uint8)

            predictor.set_image(Img_for_predictor)
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=np.ones([input_points.shape[0], 1])
            )

        # Process the predicted masks and sort by scores
        np_masks = np.array(masks[:, 0])
        np_scores = scores[:, 0]
        sorted_masks = np_masks[np.argsort(np_scores)][::-1]

        # Initialize segmentation map and occupancy mask
        # The shape should match the shape of the predicted masks
        if sorted_masks.shape[0] > 0:
            seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
            occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)
        else:
            # If sorted_masks is empty (e.g., num_samples was 0), create an empty mask
            r = np.min([1024 / image_slice.shape[1], 1024 / image_slice.shape[0]])
            resized_shape = (int(image_slice.shape[0] * r), int(image_slice.shape[1] * r))
            segmented_images.append(np.zeros(resized_shape, dtype=np.uint8))
            continue


        # Combine masks to create the final segmentation map
        for i in range(sorted_masks.shape[0]):
            mask = sorted_masks[i]
            if (mask * occupancy_mask).sum() / mask.sum() > 0.25:
                continue

            mask_bool = mask.astype(bool)
            mask_bool[occupancy_mask] = False  # Set overlapping areas to False in the mask
            seg_map[mask_bool] = i + 1  # Use boolean mask to index seg_map
            occupancy_mask[mask_bool] = True  # Update occupancy_mask

        segmented_images.append(seg_map)

    return segmented_images
    
    
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
    sam2_checkpoint = "checkpoints/sam2_hiera_tiny.pt"  # Use the tiny model checkpoint
    model_cfg = "sam2_hiera_t.yaml" # Use the tiny model configuration
    NO_OF_STEPS = args.num_epoch
    FINE_TUNED_MODEL_NAME = args.model_name
    
    print("Create train and test data by reading and formatting input files and applying train-test split\n")
    train_input, train_labels, test_input, test_labels, train_image_names, test_image_names = (choose_tune_train_test_batch(
        args.train_test_split, args.train_test_mode, args.input_folders_1, args.gt_folders_1, args.input_folders_2, args.gt_folders_2))

    if args.run_mode != 'eval':
        print("\nCreate tune data\n")
        tune_input, tune_labels, train_input, train_labels, test_input, test_labels, test_image_names = create_tune_data(args.tune_split, args.tune_indice_prop, args.train_indice_prop, args.test_indice_prop, train_input, train_labels, test_input, test_labels, test_image_names)
        
        if args.run_mode == 'tune':
            print("\nRun hyperparameter tuning\n")
            best_parameters = run_hyperparameter_tuning(args.num_tune_trials, tune_input, tune_labels, model_cfg, sam2_checkpoint)
        else:
            best_parameters = {'learning_rate': 7.626890932901104e-05, 'weight_decay': 0.004, 'step_size': 450, 
            'gamma': 0.12, 'accumulation_steps': 2, 'max_prompts': 20}
            
        print("Save model details")
        details_save_path = os.path.join("/content/drive/MyDrive/Colab Notebooks/Segmentation_Pipeline/results/Model_details/", 
        f"SAM2_{args.model_name}_{args.train_test_mode}_{args.num_epoch}_{args.run_mode}.json")
        save_model_details(best_parameters, details_save_path, args.train_test_split, args.tune_split, args.tune_indice_prop, 
                        args.train_indice_prop, args.num_tune_trials)
    
        print("\nRun training\n")
        model_save_path = "/content/drive/MyDrive/Colab Notebooks/Segmentation_Pipeline/models/SAM2/" + args.model_name 
        + '_' + args.train_test_mode + '_' + str(args.num_epoch) + '_' + str(args.run_mode)
        full_model_name = training_step(train_input, train_labels, sam2_checkpoint, model_cfg, best_parameters, 
        NO_OF_STEPS, model_save_path)
        
    elif args.run_mode == 'eval':
        full_model_name = args.pretrained_model_path
        print("\nSkipping training and using previous SAM2 model for evaluation")
        print(full_model_name)
        
        
    print("\nRun evaluation\n")      
    original_predicted_masks, original_binary_masks, seg_map, original_binary_mask_display, original_image_display = (
        eval_step(test_input, test_labels, full_model_name, model_cfg, sam2_checkpoint))

    print("\nDetermine resulting performance\n")
    results_df = calculate_metrics(original_predicted_masks, original_binary_masks, test_image_names)
    results_df.to_csv("/content/drive/MyDrive/Colab Notebooks/Segmentation_Pipeline/results/Results_files/SAM2/"+args.model_name+ '/' + args.train_test_mode + '_' + str(args.num_epoch) + '_' + str(args.run_mode) + '_results.csv')
    
    print("\nSave overlapping segmentation masks\n")
    high_performing_masks, low_performing_masks = choose_masks_to_visualise(results_df)
    
    print("\nCreate image directory if it doesn't already exist")
    image_save_dir = os.path.join("/content/drive/MyDrive/Colab Notebooks/Segmentation_Pipeline/results/Images/SAM2/", args.model_name)
    os.makedirs(image_save_dir, exist_ok=True)

    plot_multiple_views(original_predicted_masks, test_labels, test_input, test_image_names, high_performing_masks,
    f"High Performing Mask- {args.model_name}", image_save_dir, "/"+args.model_name+ '_' + args.train_test_mode + '_' 
    + str(args.num_epoch) + '_' + str(args.run_mode) +"_hpm_mask_subset.jpeg")
    
    plot_multiple_views(original_predicted_masks, test_labels, test_input, test_image_names, low_performing_masks,
    f"Low Performing Mask- {args.model_name}", image_save_dir, "/"+args.model_name+ '_' + args.train_test_mode + '_' 
    + str(args.num_epoch) + '_' + str(args.run_mode) + "_lpm_mask_subset.jpeg")
    
                        
    file_names = [s.split('.')[0] for s in test_image_names]
    persistent_lab_masks = []
    persistent_model_masks = []
    for fn in list(set(file_names)):
      img_indices = [i for i, s in enumerate(file_names) if s == fn]
      lab_pers = check_for_persistent_mask(test_labels, img_indices, N=2, visualise=False)
      persistent_lab_masks.append(lab_pers)
    
      masks_pers = check_for_persistent_mask(original_predicted_masks, img_indices, N=2, visualise=False)
      persistent_model_masks.append(masks_pers)
    
    
    results_df_summarised = calculate_metrics(persistent_model_masks, persistent_lab_masks, list(set(file_names)))
    results_df_summarised.to_csv("/content/drive/MyDrive/Colab Notebooks/Segmentation_Pipeline/results/Results_files/SAM2/"
        +args.model_name+ '/' + args.train_test_mode + '_' + str(args.num_epoch) + '_' + str(args.run_mode) + '_summarised_results.csv')
        
    return None



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train SAM2 model")
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
  
  
  
