# import os
# import time
# import shutil
# import logging
# import threading
# from datetime import datetime

# from tqdm import tqdm
# from uuid import uuid4
# from typing import List
# from main import main_func  # Import your video processing function
# from fastapi import APIRouter, File, UploadFile, Form
# from fastapi.responses import FileResponse, JSONResponse

# import threading
# processing_lock = threading.Lock()  # Global lock to ensure only one video is processed at a time


# # Initialize API router
# router = APIRouter()

# # Directories
# UPLOAD_DIR = "uploads"
# PROCESSED_DIR = "processed"
# PROGRESS_DIR = "progress"
# LOGS_DIR = "logs"

# # Ensure directories exist
# for directory in [UPLOAD_DIR, PROCESSED_DIR, PROGRESS_DIR, LOGS_DIR]:
#     os.makedirs(directory, exist_ok=True)

# # Dictionary to store task statuses
# tasks = {}



# # ------------------------ Logging Setup ------------------------

# # Ensure logs directory exists
# LOGS_DIR = "logs"
# os.makedirs(LOGS_DIR, exist_ok=True)

# # Generate a timestamped log filename
# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# LOG_FILE = os.path.join(LOGS_DIR, f"logs_{timestamp}.log")

# # Create a logger instance
# logger = logging.getLogger("    [FastAPI-App]")  # Give logger a unique name
# logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all log levels

# # Remove any existing handlers (to prevent duplicate logs)
# if logger.hasHandlers():
#     logger.handlers.clear()

# # Create a file handler to log messages to a file
# file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
# file_handler.setLevel(logging.DEBUG)

# # Define a log format
# formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# # Apply the formatter to both handlers
# file_handler.setFormatter(formatter)

# # Add the handlers to the logger
# logger.addHandler(file_handler)



# # ------------------------ Upload Video ------------------------
# @router.post("/upload/")
# async def upload_video(file: UploadFile = File(...)):
#     file_id = str(uuid4())  # Generate unique ID
#     file_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")

#     try:
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
#         logger.info(f" File uploaded successfully: {file_id}")
#         return {"file_id": file_id, "message": "File uploaded successfully"}
#     except Exception as e:
#         logger.error(f" Upload failed: {str(e)}")
#         return JSONResponse(content={"error": "File upload failed"}, status_code=500)


# # ------------------------ Track Processing Progress ------------------------
# def track_progress(file_id: str):
#     """ Monitors and updates progress for video processing """
#     logger.info(f" Progress Tracking started for {file_id} ")

#     progress_file = os.path.join(PROGRESS_DIR, f"progress_{file_id}.txt")
    
#     with tqdm(total=100, desc=f"Processing {file_id}", unit="%", ncols=80) as pbar:
#         last_progress = 0
#         while last_progress < 100:
#             if os.path.exists(progress_file):
#                 with open(progress_file, "r") as f:
#                     try:
#                         progress = int(f.read().strip())
#                     except ValueError:
#                         progress = last_progress  # Retain last known progress

#                 if progress > last_progress:
#                     pbar.update(progress - last_progress)
#                     last_progress = progress

#             time.sleep(2)


# # ------------------------ Background Processing ------------------------
# def long_video_processing(file_id: str, input_classes: List[str]):
#     """ Runs the video processing function in the background """
#     logger.info(f" Processing started for {file_id} with classes: {input_classes}")

#     input_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")
#     output_path = os.path.join(PROCESSED_DIR, f"{file_id}_processed.mp4")

#     if not os.path.exists(input_path):
#         logger.error(f" File {file_id} not found. Aborting processing.")
#         tasks[file_id] = "Failed - File not found"
#         return

#     tasks[file_id] = "Processing..."

#     # Start tracking progress
#     progress_thread = threading.Thread(target=track_progress, args=(file_id,))
#     progress_thread.start()

#     try:
#         # Call main processing function
#         main_func(input_path, output_path, input_classes, file_id)
#         tasks[file_id] = "Completed"
#         logger.info(f" Processing completed successfully for {file_id}")
#     except Exception as e:
#         tasks[file_id] = f"Failed - {str(e)}"
#         logger.error(f" Error processing {file_id}: {str(e)}")

#     progress_thread.join()


# # ------------------------ Start Processing Endpoint ------------------------
# @router.post("/process/")
# async def process_video(file_id: str = Form(...), classes: str = Form(...)):
#     """ Initiates video processing in a background thread """

#     input_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")
#     if not os.path.exists(input_path):
#         logger.error(f"Processing failed: File {file_id} not found.")
#         return JSONResponse(content={"error": "File not found"}, status_code=404)

#     # Prevent multiple videos from being processed simultaneously
#     if processing_lock.locked():
#         logger.warning(f"Processing request rejected: Another video is currently being processed.")
#         return JSONResponse(content={"error": "Another video is already being processed"}, status_code=400)

#     # Convert comma-separated classes into a list
#     input_classes = [cls.strip() for cls in classes.split(",") if cls.strip()]
#     logger.info(f"Processing request received for {file_id} with classes: {input_classes}")

#     def video_processing_wrapper():
#         """ Wrapper to ensure the lock is released after processing """
#         with processing_lock:
#             long_video_processing(file_id, input_classes)
    
#     # Start background processing with lock control
#     threading.Thread(target=video_processing_wrapper, daemon=True).start()

#     return {"file_id": file_id, "message": "Processing started", "input_classes": input_classes}


# # ------------------------ Check Processing Status ------------------------
# @router.get("/status/{file_id}")
# async def get_status(file_id: str):
#     """ Retrieves the processing status or progress percentage """
#     progress_file = os.path.join(PROGRESS_DIR, f"progress_{file_id}.txt")

#     if os.path.exists(progress_file):
#         with open(progress_file, "r") as f:
#             progress = f.read().strip()
#         if progress=="99":
#             logger.info(f" Status check for {file_id}: Completed")
#             return {"file_id": file_id, "status": "Completed"}
#         logger.info(f" Status check for {file_id}: {progress}% completed")
#         return {"file_id": file_id, "status": f"Processing {progress}%"}

#     logger.info(f" Status check for {file_id}: {tasks.get(file_id, 'Not started')}")
#     return {"file_id": file_id, "status": tasks.get(file_id, "Not started")}


# # ------------------------ Download Processed Video ------------------------
# @router.get("/download/{file_id}")
# async def download_video(file_id: str):
#     """ Allows users to download the processed video file """
#     file_path = os.path.join(PROCESSED_DIR, f"{file_id}_processed.mp4")

#     if os.path.exists(file_path):
#         logger.info(f" Download request for {file_id}")
#         return FileResponse(file_path, media_type="video/mp4", filename=f"{file_id}_processed.mp4")

#     logger.error(f" Download failed: Processed file {file_id} not found.")
#     return JSONResponse(content={"error": "Processed file not found"}, status_code=404)

# import os
# import time
# import shutil
# import logging
# import threading
# from datetime import datetime

# from tqdm import tqdm
# from uuid import uuid4
# from typing import List
# from main import main_func  # Import your video processing function
# from fastapi import APIRouter, File, UploadFile, Form
# from fastapi.responses import FileResponse, JSONResponse

# processing_lock = threading.Lock()  # Global lock to ensure only one video is processed at a time

# # Initialize API router
# router = APIRouter()

# # Directories
# UPLOAD_DIR = "uploads"
# PROCESSED_DIR = "processed"
# PROGRESS_DIR = "progress"
# LOGS_DIR = "logs"

# # Ensure directories exist
# for directory in [UPLOAD_DIR, PROCESSED_DIR, PROGRESS_DIR, LOGS_DIR]:
#     os.makedirs(directory, exist_ok=True)

# # Dictionary to store task statuses
# tasks = {}

# # ------------------------ Logging Setup ------------------------
# LOGS_DIR = "logs"
# os.makedirs(LOGS_DIR, exist_ok=True)

# # Generate a timestamped log filename
# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# LOG_FILE = os.path.join(LOGS_DIR, f"logs_{timestamp}.log")

# # Create a logger instance
# logger = logging.getLogger("[FastAPI-App]")  
# logger.setLevel(logging.DEBUG)  

# # Remove any existing handlers
# if logger.hasHandlers():
#     logger.handlers.clear()

# # Create a file handler to log messages
# file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
# file_handler.setLevel(logging.DEBUG)

# # Define log format
# formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

# # # ------------------------ Upload Video ------------------------
# @router.post("/upload/")
# async def upload_video(file: UploadFile = File(...)):
#     file_id = str(uuid4())  # Generate unique ID
#     file_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")

#     try:
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
#         logger.info(f" File uploaded successfully: {file_id}")
#         return {"file_id": file_id, "message": "File uploaded successfully"}
#     except Exception as e:
#         logger.error(f" Upload failed: {str(e)}")
#         return JSONResponse(content={"error": "File upload failed"}, status_code=500)


# # ------------------------ Track Processing Progress ------------------------
# def track_progress(file_id: str):
#     """ Monitors and updates progress for video processing """
#     logger.info(f" Progress Tracking started for {file_id} ")

#     progress_file = os.path.join(PROGRESS_DIR, f"progress_{file_id}.txt")
    
#     with tqdm(total=100, desc=f"Processing {file_id}", unit="%", ncols=80) as pbar:
#         last_progress = 0
#         while last_progress < 100:
#             if os.path.exists(progress_file):
#                 with open(progress_file, "r") as f:
#                     try:
#                         progress = int(f.read().strip())
#                     except ValueError:
#                         progress = last_progress  # Retain last known progress

#                 if progress > last_progress:
#                     pbar.update(progress - last_progress)
#                     last_progress = progress

#             time.sleep(2)


# # ------------------------ Background Processing ------------------------
# def long_video_processing(file_id: str, input_classes: List[str], desired_fps: int):
#     """ Runs the video processing function in the background with configurable FPS """
#     logger.info(f"Processing started for {file_id} with classes: {input_classes} at {desired_fps} FPS")

#     input_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")
#     output_path = os.path.join(PROCESSED_DIR, f"{file_id}_processed.mp4")

#     if not os.path.exists(input_path):
#         logger.error(f"File {file_id} not found. Aborting processing.")
#         tasks[file_id] = "Failed - File not found"
#         return

#     tasks[file_id] = "Processing..."

#     try:
#         # ✅ Pass the user-defined FPS to `main_func`
#         main_func(input_path, output_path, input_classes, file_id, desired_fps)
#         tasks[file_id] = "Completed"
#         logger.info(f"Processing completed successfully for {file_id}")
#     except Exception as e:
#         tasks[file_id] = f"Failed - {str(e)}"
#         logger.error(f"Error processing {file_id}: {str(e)}")

# # ------------------------ Start Processing Endpoint ------------------------
# @router.post("/process/")
# async def process_video(
#     file_id: str = Form(...), 
#     classes: str = Form(...), 
#     desired_fps: int = Form(...)  # ✅ Accept FPS from user input
# ):
#     """ Initiates video processing with user-defined FPS """

#     input_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")
#     if not os.path.exists(input_path):
#         logger.error(f"Processing failed: File {file_id} not found.")
#         return JSONResponse(content={"error": "File not found"}, status_code=404)

#     # Prevent multiple videos from being processed simultaneously
#     if processing_lock.locked():
#         logger.warning(f"Another video is currently being processed.")
#         return JSONResponse(content={"error": "Another video is already being processed"}, status_code=400)

#     # Convert classes into a list
#     input_classes = [cls.strip() for cls in classes.split(",") if cls.strip()]
#     logger.info(f"Received processing request for {file_id} with classes: {input_classes} at {desired_fps} FPS")

#     def video_processing_wrapper():
#         """ Wrapper to ensure the lock is released after processing """
#         with processing_lock:
#             long_video_processing(file_id, input_classes, desired_fps)

#     # Start background processing with lock control
#     threading.Thread(target=video_processing_wrapper, daemon=True).start()

#     return {
#         "file_id": file_id,
#         "message": "Processing started",
#         "input_classes": input_classes,
#         "fps": desired_fps
#     }

# # ------------------------ Check Processing Status ------------------------
# @router.get("/status/{file_id}")
# async def get_status(file_id: str):
#     """ Retrieves the processing status or progress percentage """
#     progress_file = os.path.join(PROGRESS_DIR, f"progress_{file_id}.txt")

#     if os.path.exists(progress_file):
#         with open(progress_file, "r") as f:
#             progress = f.read().strip()
#         if progress == "99":
#             logger.info(f"Status check for {file_id}: Completed")
#             return {"file_id": file_id, "status": "Completed"}
#         logger.info(f"Status check for {file_id}: {progress}% completed")
#         return {"file_id": file_id, "status": f"Processing {progress}%"}

#     logger.info(f"Status check for {file_id}: {tasks.get(file_id, 'Not started')}")
#     return {"file_id": file_id, "status": tasks.get(file_id, "Not started")}

# # ------------------------ Download Processed Video ------------------------
# @router.get("/download/{file_id}")
# async def download_video(file_id: str):
#     """ Allows users to download the processed video file """
#     file_path = os.path.join(PROCESSED_DIR, f"{file_id}_processed.mp4")

#     if os.path.exists(file_path):
#         logger.info(f"Download request for {file_id}")
#         return FileResponse(file_path, media_type="video/mp4", filename=f"{file_id}_processed.mp4")

#     logger.error(f"Download failed: Processed file {file_id} not found.")
#     return JSONResponse(content={"error": "Processed file not found"}, status_code=404)

import os
import time
import shutil
import logging
import threading
from datetime import datetime

from tqdm import tqdm
from uuid import uuid4
from typing import List
from main import main_func  # Your video processing function
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse

# ------------------------ Directories ------------------------
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
PROGRESS_DIR = "progress"
LOGS_DIR = "logs"

# Ensure directories exist
for directory in [UPLOAD_DIR, PROCESSED_DIR, PROGRESS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ------------------------ Logger Setup ------------------------
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = os.path.join(LOGS_DIR, f"logs_{timestamp}.log")

logger = logging.getLogger("FastAPI-App")
logger.setLevel(logging.DEBUG)

# Clear existing handlers
if logger.hasHandlers():
    logger.handlers.clear()

# Add file handler
file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# ------------------------ Global Variables ------------------------
processing_lock = threading.Lock()  # Lock to prevent simultaneous processing
tasks = {}  # Dictionary to track processing status

# ------------------------ API Router ------------------------
router = APIRouter()

# ------------------------ Upload Video ------------------------
@router.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    """ Uploads a video and stores it in the upload directory """
    file_id = str(uuid4())  # Generate unique ID
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"File uploaded successfully: {file_id}")
        return {"file_id": file_id, "message": "File uploaded successfully"}

    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail="File upload failed")

# ------------------------ Progress Tracker ------------------------
def track_progress(file_id: str):
    """ Monitors and logs processing progress """
    logger.info(f"Tracking progress for {file_id}")
    
    progress_file = os.path.join(PROGRESS_DIR, f"progress_{file_id}.txt")
    
    with tqdm(total=100, desc=f"Processing {file_id}", unit="%", ncols=80) as pbar:
        last_progress = 0
        while last_progress < 100:
            if os.path.exists(progress_file):
                with open(progress_file, "r") as f:
                    try:
                        progress = int(f.read().strip())
                    except ValueError:
                        progress = last_progress

                if progress > last_progress:
                    pbar.update(progress - last_progress)
                    last_progress = progress

            time.sleep(2)

# ------------------------ Background Video Processing ------------------------
def long_video_processing(file_id: str, input_classes: List[str], desired_fps: int):
    """ Runs video processing in the background with safe handling """
    logger.info(f"Processing started for {file_id} with classes: {input_classes} at {desired_fps} FPS")

    input_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")
    output_path = os.path.join(PROCESSED_DIR, f"{file_id}_processed.mp4")

    # ✅ Check if file exists before processing
    if not os.path.exists(input_path):
        logger.error(f"File {file_id} not found. Aborting processing.")
        tasks[file_id] = "Failed - File not found"
        return

    tasks[file_id] = "Processing..."

    try:
        # ✅ Run main_func with proper exception handling
        main_func(input_path, output_path, input_classes, file_id, desired_fps)

        if not os.path.exists(output_path):
            raise Exception("Output file not generated")

        tasks[file_id] = "Completed"
        logger.info(f"Processing completed successfully for {file_id}")

    except Exception as e:
        tasks[file_id] = f"Failed - {str(e)}"
        logger.error(f"Error processing {file_id}: {str(e)}")

# ------------------------ Start Processing Endpoint ------------------------
@router.post("/process/")
async def process_video(
    file_id: str = Form(...), 
    classes: str = Form(...), 
    desired_fps: int = Form(...)
):
    """ Initiates video processing with user-defined FPS """

    input_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")
    
    # ✅ Validate file existence
    if not os.path.exists(input_path):
        logger.error(f"File {file_id} not found.")
        raise HTTPException(status_code=404, detail="File not found")

    # ✅ Prevent simultaneous processing
    if processing_lock.locked():
        logger.warning(f"Another video is currently being processed.")
        raise HTTPException(status_code=400, detail="Another video is being processed")

    # ✅ Convert classes into list
    input_classes = [cls.strip() for cls in classes.split(",") if cls.strip()]
    logger.info(f"Received request for {file_id} with classes: {input_classes} at {desired_fps} FPS")

    def video_processing_wrapper():
        """ Wrapper to release the lock after processing """
        with processing_lock:
            long_video_processing(file_id, input_classes, desired_fps)

    # ✅ Start background processing with lock control
    threading.Thread(target=video_processing_wrapper, daemon=True).start()

    return {
        "file_id": file_id,
        "message": "Processing started",
        "input_classes": input_classes,
        "fps": desired_fps
    }

# ------------------------ Check Processing Status ------------------------
@router.get("/status/{file_id}")
async def get_status(file_id: str):
    """ Retrieves the processing status or progress """
    
    progress_file = os.path.join(PROGRESS_DIR, f"progress_{file_id}.txt")

    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            progress = f.read().strip()

        if progress == "99":
            logger.info(f"Status check for {file_id}: Completed")
            return {"file_id": file_id, "status": "Completed"}

        logger.info(f"Status check for {file_id}: {progress}% completed")
        return {"file_id": file_id, "status": f"Processing {progress}%"}

    logger.info(f"Status check for {file_id}: {tasks.get(file_id, 'Not started')}")
    return {"file_id": file_id, "status": tasks.get(file_id, "Not started")}

# ------------------------ Download Processed Video ------------------------
@router.get("/download/{file_id}")
async def download_video(file_id: str):
    """ Allows users to download the processed video """
    file_path = os.path.join(PROCESSED_DIR, f"{file_id}_processed.mp4")

    # ✅ Check file existence before responding
    if os.path.exists(file_path):
        logger.info(f"Download request for {file_id}")
        return FileResponse(file_path, media_type="video/mp4", filename=f"{file_id}_processed.mp4")

    logger.error(f"Download failed: Processed file {file_id} not found.")
    raise HTTPException(status_code=404, detail="Processed file not found")
