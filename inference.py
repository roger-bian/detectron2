import os
import datetime
import argparse

import cv2
import torch
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo


def resolve_config_and_weights(args):
    if os.path.isfile(args.config):
        # local config.yaml from a previously trained model
        config_file = args.config
        if args.weights is None:
            raise ValueError("--weights is required when --config points to a local config.yaml file")
        weights = args.weights
        used_default_weights = False
    else:
        # detectron2 model zoo backbone, e.g. COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
        config_file = model_zoo.get_config_file(args.config)
        if args.weights is None:
            weights = model_zoo.get_checkpoint_url(args.config)
            used_default_weights = True
        else:
            weights = args.weights
            used_default_weights = False
    return config_file, weights, used_default_weights


def setup_cfg(args):
    config_file, weights, used_default_weights = resolve_config_and_weights(args)

    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh
    if args.cpu or not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()
    return cfg, used_default_weights


def get_metadata(args, cfg, used_default_weights):
    if args.class_names:
        metadata = MetadataCatalog.get("inference_metadata")
        metadata.thing_classes = args.class_names
        return metadata
    if used_default_weights and cfg.DATASETS.TRAIN:
        # pretrained COCO weights -> use the built-in COCO class names
        return MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    return None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


def process_image(image_path, predictor, metadata, output_path, show):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping unreadable image: {image_path}")
        return

    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")

    visualizer = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1)
    out = visualizer.draw_instance_predictions(instances)
    result_image = out.get_image()[:, :, ::-1]

    cv2.imwrite(output_path, result_image)

    if show:
        cv2.imshow("Detectron2 Inference (press any key to continue)", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Saved output image to: {output_path}")


def process_stream(cap, predictor, metadata, output_path, show):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    progress = tqdm(total=total_frames, desc=os.path.basename(output_path), unit="frame")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            outputs = predictor(frame)
            instances = outputs["instances"].to("cpu")

            visualizer = Visualizer(frame[:, :, ::-1], metadata=metadata, scale=1)
            out = visualizer.draw_instance_predictions(instances)
            result_frame = out.get_image()[:, :, ::-1]

            writer.write(result_frame)

            if show:
                cv2.imshow("Detectron2 Inference (press 'q' to quit)", result_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            progress.update(1)
    finally:
        progress.close()
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

    print(f"Saved output video to: {output_path}")


def run_on_video_file(video_path, predictor, metadata, output_dir, timestamp, show):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{video_name}_{timestamp}.mp4")
    process_stream(cap, predictor, metadata, output_path, show=show)


def run_on_image_file(image_path, predictor, metadata, output_dir, timestamp, show):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    ext = os.path.splitext(image_path)[1]
    output_path = os.path.join(output_dir, f"{image_name}_{timestamp}{ext}")
    process_image(image_path, predictor, metadata, output_path, show=show)


def run_on_path(path, predictor, metadata, output_dir, timestamp, show):
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            run_on_image_file(path, predictor, metadata, output_dir, timestamp, show)
        elif ext in VIDEO_EXTENSIONS:
            run_on_video_file(path, predictor, metadata, output_dir, timestamp, show)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        return

    if os.path.isdir(path):
        entries = sorted(os.listdir(path))
        for entry in tqdm(entries, desc="Processing files", unit="file"):
            entry_path = os.path.join(path, entry)
            if not os.path.isfile(entry_path):
                continue
            ext = os.path.splitext(entry)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                run_on_image_file(entry_path, predictor, metadata, output_dir, timestamp, show)
            elif ext in VIDEO_EXTENSIONS:
                run_on_video_file(entry_path, predictor, metadata, output_dir, timestamp, show)
        return

    raise FileNotFoundError(f"Path not found: {path}")


def main(args):
    cfg, used_default_weights = setup_cfg(args)
    predictor = DefaultPredictor(cfg)
    metadata = get_metadata(args, cfg, used_default_weights)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.webcam is not None:
        cap = cv2.VideoCapture(args.webcam)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {args.webcam}")
        output_path = os.path.join(args.output_dir, f"webcam_{timestamp}.mp4")
        process_stream(cap, predictor, metadata, output_path, show=not args.no_display)
    else:
        run_on_path(args.path, predictor, metadata, args.output_dir, timestamp, show=not args.no_display)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a trained detectron2 model on a webcam feed, or on an image/video file "
                    "or a directory of images/videos, saving the annotated output to the "
                    "_output directory."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Either a path to a local config.yaml from a previously trained model "
             "(e.g. _model/side_panel_screw/v2/config.yaml), or a detectron2 model zoo "
             "config name (e.g. COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml)."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model weights. Required when --config is a local config.yaml. "
             "When --config is a model zoo name, defaults to that backbone's pretrained "
             "COCO weights, or can point to a locally trained .pth using that architecture."
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--webcam",
        type=int,
        nargs="?",
        const=0,
        default=None,
        help="Run live detection on a webcam. Optionally provide a camera index (default: 0)."
    )
    source_group.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to an input image or video file, or a directory containing images and/or "
             "videos to run detection on (subdirectories are not scanned)."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="_output",
        help="Directory to save the annotated output video (default: _output)"
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.5,
        help="Minimum score threshold for detections (default: 0.5)"
    )
    parser.add_argument(
        "--class_names",
        type=str,
        nargs="+",
        default=None,
        help="Class names in label order, used to annotate detections (e.g. --class_names screw). "
             "Not needed when using default COCO-pretrained weights; COCO class names are used automatically."
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Do not show a live preview window while processing"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force inference on CPU even if a GPU is available"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
