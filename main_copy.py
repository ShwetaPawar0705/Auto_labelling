def main_func(input_path, output_path, input_classes, file_id, desired_fps):
    import os
    import cv2
    import shutil
    import supervision as sv
    from path_config import HOME
    from SAM import segment_instances
    from mask_auto_annotation import auto_BB_annotate
    from segmented_frames_to_video import frames_to_video
    from SAM_for_video import bbox_to_sam, visualize_segment_video, video_to_frames, FRAMES_DIR_PATH

    SCALE_FACTOR = 1.0

    try:
        video_info = sv.VideoInfo.from_video_path(input_path)
        video_info.width = int(video_info.width * SCALE_FACTOR)
        video_info.height = int(video_info.height * SCALE_FACTOR)
    except Exception as e:
        print(f"Error reading video info: {e}")
        return

    SOURCE_DIR = f"{HOME}/data"
    OUTPUT_DIR_SM = f"{HOME}/segment_masks"
    os.makedirs(OUTPUT_DIR_SM, exist_ok=True)
    os.makedirs("segmented_frames", exist_ok=True)

    annotated_dir = os.path.join(HOME, "segmented_frames_annotated", file_id)
    os.makedirs(annotated_dir, exist_ok=True)

    CLASSES = input_classes

    try:
        fps, totalNoFrames = video_to_frames(input_path, FRAMES_DIR_PATH)
        print(f'\nfps: {fps}\ntotal number of frames: {totalNoFrames}\n\n')
    except Exception as e:
        print(f"Error converting video to frames: {e}")
        return

    skip_frames = int(totalNoFrames * (desired_fps / fps))
    final_skip_frames = int(totalNoFrames / skip_frames)
    print("skip_frames: ", skip_frames)
    print("Final_skipped_frames: ", final_skip_frames)

    try:
        image_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(('.jpeg', '.jpg', '.png'))]
        print(f'image_files: {type(image_files)}, len{len(image_files)}')
    except Exception as e:
        print(f"Error reading image files: {e}")
        return

    FRAME_IDX = 0
    detection_enabled = True  # control detection toggle

    progress_dir = "progress"
    os.makedirs(progress_dir, exist_ok=True)
    progress_file = os.path.join(progress_dir, f"progress_{file_id}.txt")

    INTERRUPT_DIR = "interrupts"
    os.makedirs(INTERRUPT_DIR, exist_ok=True)
    interrupt_file = os.path.join(INTERRUPT_DIR, f"interrupt_{file_id}.txt")

    for index, image_file in enumerate(sorted(image_files)):

        # ✅ Early interrupt check
        interrupt = None
        if os.path.exists(interrupt_file):
            with open(interrupt_file, "r") as f:
                interrupt = f.read().strip()
            print(f"[Interrupt Check] Status: '{interrupt}'")
            if interrupt == "STOPPROCESS":
                print("Hard stop requested.")
                break
            elif interrupt == "STOPDETECTION":
                print("Detection disabled. Proceeding with raw frames.")
                detection_enabled = False

        source_image_path = os.path.join(SOURCE_DIR, image_file)

        if index % final_skip_frames != 0:
            frame = cv2.imread(source_image_path)
            cv2.imwrite(f"segmented_frames/frame_{index}.jpg", frame)
            continue

        print(f"Processing: {source_image_path}")
        frame = cv2.imread(source_image_path)

        if detection_enabled:
            try:
                result = auto_BB_annotate(source_image_path, CLASSES)
                if result is None:
                    raise ValueError("auto_BB_annotate returned None")
                image, detections, labels, OUTPUT_IMAGE_PATH = result
            except Exception as e:
                print(f"Error during bounding box annotation: {e}")
                cv2.imwrite(f"segmented_frames/frame_{index}.jpg", frame)
                continue

            if detections is None or not hasattr(detections, 'xyxy') or len(detections.xyxy) == 0:
                print("No valid detections found.")
                cv2.imwrite(f"segmented_frames/frame_{index}.jpg", frame)
                continue

            # Recheck for interrupts just before SAM
            interrupt = None
            if os.path.exists(interrupt_file):
                with open(interrupt_file, "r") as f:
                    interrupt = f.read().strip()
                if interrupt == "STOPPROCESS":
                    print("Stopped before SAM.")
                    break
                elif interrupt == "STOPDETECTION":
                    print("Detection was just disabled.")
                    detection_enabled = False
                    cv2.imwrite(f"segmented_frames/frame_{index}.jpg", frame)
                    continue

            # Segmentation + SAM
            try:
                mask = segment_instances(OUTPUT_IMAGE_PATH, image, detections, CLASSES)
                object_ids, mask_logits, inference_state = bbox_to_sam(
                    file_id, False, final_skip_frames, SOURCE_DIR, source_image_path, FRAME_IDX,
                    index % final_skip_frames, detections, CLASSES, image, mask)
            except Exception as e:
                print(f"Error during SAM processing: {e}")
                continue

            try:
                video_info = visualize_segment_video(index, input_path, inference_state, file_id)
            except Exception as e:
                print(f"Error during video propagation: {e}")
                continue

            FRAME_IDX = min(FRAME_IDX + 1, len(image_files))
        else:
            # If detection is stopped, write raw frame
            print("Skipping detection, writing raw frame.")
            cv2.imwrite(f"segmented_frames/frame_{index}.jpg", frame)

        progress = (index / totalNoFrames) * 100
        with open(progress_file, "w") as f:
            f.write(str(int(progress)) + '\n')
        print(f"{file_id} - Progress: {int(progress)}%")

    # Final video generation
    input_frames_dir = 'segmented_frames'
    try:
        frames_to_video(input_frames_dir, output_path, frame_rate=video_info.fps, width=video_info.width, height=video_info.height)
        print('Output video created successfully.')
    except Exception as e:
        print(f"Error creating output video: {e}")
        return

    # Cleanup
    dir_to_clear = ['segmented_frames', 'data', 'bounding_box', 'bbox_output', 'instance_segmentation', 'inference_frames']
    for dir_path in dir_to_clear:
        print('deleting - ', dir_path)
        shutil.rmtree(dir_path, ignore_errors=True)
        os.makedirs(dir_path, exist_ok=True)

    with open(interrupt_file, "w") as f:
        f.write("Ended\n")
        print('Interrupt file updated to "Ended"')
    print("Processing complete!")
