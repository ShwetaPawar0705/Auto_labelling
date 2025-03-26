import os
import cv2

from path_config import HOME
from SAM import segment_instances   # Function to create 1st segmentation for every skip frame
# from V_segment_masks import plot_and_save_masks       # Plotting objects masks in folder -> segment_masks
from mask_auto_annotation import auto_BB_annotate     # Function for bounding box annotation
from segmented_frames_to_video import frames_to_video       # Create video from segmented frames(final)
from SAM_for_video import bbox_to_sam, visualize_segment_video, video_to_frames, FRAMES_DIR_PATH     # For video_propagation 


# this is the main function to execute the entire flow of the project
def main_func(input_path, output_path, input_classes,file_id,desired_fps):

    # Define source directory for images
    SOURCE_DIR = f"{HOME}/data"
    OUTPUT_DIR_SM = f"{HOME}/segment_masks"     # for storing individual object masks

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR_SM, exist_ok=True)

    # objects classes to detect, passed by user
    CLASSES = input_classes

    # Get all image files in the data directory
    # image_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(('.jpeg', '.jpg', '.png'))]
    # print(f'image_files: {type(image_files)}')
    


    #1 generate frames from video 
    fps, totalNoFrames=video_to_frames(input_path, FRAMES_DIR_PATH)
    print(f'\nfps: {fps}\ntotal number of frames: {totalNoFrames}\n\n')
    skip_frames = int(totalNoFrames*(desired_fps/fps))
    final_skip_frames = int(totalNoFrames/skip_frames)
    print("skip_frames: ", skip_frames)
    print("Final_skipped_frames: ", final_skip_frames)
    
    image_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(('.jpeg', '.jpg', '.png'))]
    print(f'Total frames extracted: {totalNoFrames}')
    print(f'Frames in SOURCE_DIR: {len(image_files)}')
    # initialize imp variables
    FRAME_IDX=0
    toggle=0

    # create the progress dir to be accessed by STATUS endpoint
    progress_dir = "progress"
    os.makedirs(progress_dir, exist_ok=True)
    progress_file = os.path.join(progress_dir, f"progress_{file_id}.txt")


    # Loop through each image and process it__annote and segment loop (skipping fps/4 frames)
    for index, image_file in enumerate(sorted(image_files)):
        
        # get enumerated image path
        source_image_path = os.path.join(SOURCE_DIR, image_file)

        # condition for skipping frames based on fps
        if index % final_skip_frames != 0:
            print('-=-=-=-=-=-=-=-=-skipped: ', index % final_skip_frames)
            # toggle variable to decide if skipped frames should be saved ...otherwise will be accessed later by the inference state
            if toggle == 1:
                frame = cv2.imread(source_image_path)
                cv2.imwrite(f"segmented_frames/frame_{index}.jpg", frame)
            continue
        else:
            print('-=-=--=-=-=--=-=-=-=Not skipped: ', index % skip_frames)
        # initialize inference index
        inference_index = index%final_skip_frames
        
        print(f"Processing: {source_image_path}")  # Debugging output

        #2 Perform bounding box annotation
        image, detections, labels, OUTPUT_IMAGE_PATH = auto_BB_annotate(source_image_path, CLASSES)

        print(f'detections.mask: {detections.xyxy}')
        print(f'labels: {labels}')

        # no mask or object detected in frame(every {int(fps/4)}th frame taken)
        if len(detections.xyxy)==0:
            print("No detections found.")
            
            # toggle variable updated
            toggle = 1
            frame = cv2.imread(source_image_path)
            cv2.imwrite(f"segmented_frames/frame_{index}.jpg", frame)
            continue

        # toggle variable updated
        toggle=0

        #3 Perform instance segmentation
        mask = segment_instances(OUTPUT_IMAGE_PATH, image, detections, CLASSES)

        #4 Add detected masks to the model and initialize the inference state
        object_ids, mask_logits, inference_state = bbox_to_sam(final_skip_frames, SOURCE_DIR, source_image_path, FRAME_IDX, inference_index, detections, CLASSES, image, mask)

        # ✅ Check for None before accessing `xyxy`
        if detections is not None and hasattr(detections, 'xyxy'):
            print(f'detections.mask: {detections.xyxy}')
        else:
            print("No detections or empty detections found.")
            toggle = 1
            frame = cv2.imread(source_image_path)
            cv2.imwrite(f"segmented_frames/frame_{index}.jpg", frame)
            continue
        
        print(f'FRAME_IDX: {FRAME_IDX}')
        
        # update the frame ID
        FRAME_IDX+=1

        # to check if inference is not None
        assert inference_state is not None, "Error: inference_state is None!"

        #5 Propagate masks in video...
        video_info = visualize_segment_video(index, input_path, inference_state)
        print('Video propagation complete!!!....')

        progress = (index / totalNoFrames) * 100

        # Save progress
        with open(progress_file, "w") as f:
            f.write(str(int(progress))+'\n')

        print(f"{file_id} - Progress: {int(progress)}%")

        # Save segmentation masks
        # plot_and_save_masks(detections, CLASSES, OUTPUT_DIR_SM, image_file)

    # initialize dirs for final frame to video conversion
    input_frames_dir = 'segmented_frames'  # Directory containing frames
    output_video_path = output_path     # path for output video

    #6 Frames to video conversion
    frames_to_video(input_frames_dir, output_video_path, frame_rate=video_info.fps, width=video_info.width, height=video_info.height)

    print("Processing complete!")



# def main_func(input_path, output_path, input_classes, file_id, fps=35):
#     """
#     Main function to process video frames, perform segmentation, and generate the final video.
#     """
#     print("This is the fps passed by user",fps)
#     # Define source directory for images
#     SOURCE_DIR = f"{HOME}/data"
#     OUTPUT_DIR_SM = f"{HOME}/segment_masks"  # for storing individual object masks

#     # Ensure output directory exists
#     os.makedirs(OUTPUT_DIR_SM, exist_ok=True)

#     # Objects classes to detect, passed by user
#     CLASSES = input_classes

#     # 1. Generate frames from video 
#     totalNoFrames = video_to_frames(input_path, FRAMES_DIR_PATH)
#     print(f'\nfps: {fps}\ntotal number of frames: {totalNoFrames}\n\n')

#     # Get all image files in the data directory
#     image_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(('.jpeg', '.jpg', '.png'))]
#     print(f'Total frames extracted: {totalNoFrames}')
#     print(f'Fram in SOURCE_DIR: {len(image_files)}')

#     # Ensure the number of frames matches
#     if len(image_files) != totalNoFrames:
#         raise ValueError(f"Mismatch between extracted frames ({totalNoFrames}) and frames in SOURCE_DIR ({len(image_files)})")

#     # Calculate skip frames based on fps
#     skip_frames = int(fps)  # Ensure skip_frames is at least 1

#     # Initialize important variables
#     FRAME_IDX = 0
#     toggle = 0

#     # Create the progress dir to be accessed by STATUS endpoint
#     progress_dir = "progress"
#     os.makedirs(progress_dir, exist_ok=True)
#     progress_file = os.path.join(progress_dir, f"progress_{file_id}.txt")

#     # Loop through each image and process it (annotate and segment loop, skipping frames based on fps)
#     for index, image_file in enumerate(sorted(image_files)):
#         # Get enumerated image path
#         source_image_path = os.path.join(SOURCE_DIR, image_file)

#         # Print for all frames
#         print(f'Index: {index}, Skip Condition: {index % skip_frames}')



#         # Condition for skipping frames based on fps
#         if index % skip_frames != 0:
#             print('-=-=-=-=-=-=-=-=-skipped: ',index%skip_frames)
            
#             # Toggle variable to decide if skipped frames should be saved
#             if toggle == 1:
#                 frame = cv2.imread(source_image_path)
#                 cv2.imwrite(f"segmented_frames/frame_{index}.jpg", frame)
#             continue
#         else:
#             print('-=-=--=-=-=--=-=-=-=Not skipped: ',index%skip_frames)
#         # Initialize inference index
#         inference_index = index % skip_frames
        
#         print(f"Processing: {source_image_path}")  # Debugging output

#         # 2. Perform bounding box annotation
#         image, detections, labels, OUTPUT_IMAGE_PATH = auto_BB_annotate(source_image_path, CLASSES)

#         print(f'detections.mask: {detections.xyxy}')
#         print(f'labels: {labels}')

#         # No mask or object detected in frame (every {int(fps/4)}th frame taken)
#         if len(detections.xyxy) == 0:
#             print("No detections found.")
#             # Toggle variable updated
#             toggle = 1
#             frame = cv2.imread(source_image_path)
#             cv2.imwrite(f"segmented_frames/frame_{index}.jpg", frame)
#             continue

#         # Toggle variable updated
#         toggle = 0

#         # 3. Perform instance segmentation
#         mask = segment_instances(OUTPUT_IMAGE_PATH, image, detections, CLASSES)

#         # 4. Add detected masks to the model and initialize the inference state
#         object_ids, mask_logits, inference_state = bbox_to_sam(skip_frames, SOURCE_DIR, source_image_path, FRAME_IDX, inference_index, detections, CLASSES, image, mask)
        
#         print(f'FRAME_IDX: {FRAME_IDX}')
        
#         # Update the frame ID
#         FRAME_IDX += 1

#         # Check if inference is not None
#         assert inference_state is not None, "Error: inference_state is None!"

#         # 5. Propagate masks in video...
#         video_info = visualize_segment_video(index, input_path, inference_state)
#         print('Video propagation complete!!!....')

#         # Calculate progress
#         progress = (index / totalNoFrames) * 100

#         # Save progress
#         with open(progress_file, "w") as f:
#             f.write(str(int(progress)) + '\n')

#         print(f"{file_id} - Progress: {int(progress)}%")

#     # Initialize dirs for final frame to video conversion
#     input_frames_dir = 'segmented_frames'  # Directory containing frames
#     output_video_path = output_path  # Path for output video

#     # 6. Frames to video conversion
#     frames_to_video(input_frames_dir, output_video_path, frame_rate=video_info.fps, width=video_info.width, height=video_info.height)

#     print("Processing complete!")

