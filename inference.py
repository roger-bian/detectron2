import os
import datetime

import cv2
import torch
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, _create_text_labels
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

from src.settings import load_settings


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
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.nms_thresh
    if args.min_size_test is not None:
        cfg.INPUT.MIN_SIZE_TEST = args.min_size_test
    if args.max_size_test is not None:
        cfg.INPUT.MAX_SIZE_TEST = args.max_size_test
    if args.class_names and len(args.class_names) != cfg.MODEL.ROI_HEADS.NUM_CLASSES:
        print(
            f"Overriding MODEL.ROI_HEADS.NUM_CLASSES: "
            f"{cfg.MODEL.ROI_HEADS.NUM_CLASSES} -> {len(args.class_names)} "
            f"(to match class_names)"
        )
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(args.class_names)
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


def draw_single_color_instances(visualizer, predictions, mask_color, mask_alpha, box_color):
    num_instances = len(predictions)
    if num_instances == 0:
        return visualizer.output

    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
    labels = _create_text_labels(classes, scores, visualizer.metadata.get("thing_classes", None))
    masks = predictions.pred_masks.numpy() if predictions.has("pred_masks") else None

    if masks is not None:
        visualizer.overlay_instances(masks=masks, assigned_colors=[mask_color] * num_instances, alpha=mask_alpha)
    if boxes is not None:
        edge_color = box_color if box_color is not None else mask_color
        visualizer.overlay_instances(boxes=boxes, labels=labels, assigned_colors=[edge_color] * num_instances)

    return visualizer.output


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


def process_image(image_path, predictor, metadata, output_path, show, color=None, mask_alpha=0.8, box_color=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping unreadable image: {image_path}")
        return

    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")

    visualizer = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1)
    if color is not None:
        out = draw_single_color_instances(visualizer, instances, color, mask_alpha, box_color)
    else:
        out = visualizer.draw_instance_predictions(instances)
        if box_color is not None and instances.has("pred_boxes") and len(instances) > 0:
            out = visualizer.overlay_instances(boxes=instances.pred_boxes, assigned_colors=[box_color] * len(instances))
    result_image = out.get_image()[:, :, ::-1]

    cv2.imwrite(output_path, result_image)

    if show:
        window_name = "Detectron2 Inference (press any key to continue)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Saved output image to: {output_path}")


def process_stream(cap, predictor, metadata, output_path, show, color=None, mask_alpha=0.8, box_color=None):
    writer = None
    if output_path is not None:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None

    window_name = "Detectron2 Inference (press 'q' to quit)"
    if show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    progress_desc = os.path.basename(output_path) if output_path is not None else "webcam"
    progress = tqdm(total=total_frames, desc=progress_desc, unit="frame")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            outputs = predictor(frame)
            instances = outputs["instances"].to("cpu")

            visualizer = Visualizer(frame[:, :, ::-1], metadata=metadata, scale=1)
            if color is not None:
                out = draw_single_color_instances(visualizer, instances, color, mask_alpha, box_color)
            else:
                out = visualizer.draw_instance_predictions(instances)
                if box_color is not None and instances.has("pred_boxes") and len(instances) > 0:
                    out = visualizer.overlay_instances(boxes=instances.pred_boxes, assigned_colors=[box_color] * len(instances))
            result_frame = out.get_image()[:, :, ::-1]

            if writer is not None:
                writer.write(result_frame)

            if show:
                cv2.imshow(window_name, result_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            progress.update(1)
    finally:
        progress.close()
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    if output_path is not None:
        print(f"Saved output video to: {output_path}")


def run_on_video_file(video_path, predictor, metadata, output_dir, timestamp, show, color=None, mask_alpha=0.8, box_color=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{video_name}_{timestamp}.mp4")
    process_stream(
        cap, predictor, metadata, output_path, show=show,
        color=color, mask_alpha=mask_alpha, box_color=box_color,
    )


def run_on_image_file(image_path, predictor, metadata, output_dir, timestamp, show, color=None, mask_alpha=0.8, box_color=None):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    ext = os.path.splitext(image_path)[1]
    output_path = os.path.join(output_dir, f"{image_name}_{timestamp}{ext}")
    process_image(
        image_path, predictor, metadata, output_path, show=show,
        color=color, mask_alpha=mask_alpha, box_color=box_color,
    )


def run_on_path(path, predictor, metadata, output_dir, timestamp, show, color=None, mask_alpha=0.8, box_color=None):
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            run_on_image_file(path, predictor, metadata, output_dir, timestamp, show, color, mask_alpha, box_color)
        elif ext in VIDEO_EXTENSIONS:
            run_on_video_file(path, predictor, metadata, output_dir, timestamp, show, color, mask_alpha, box_color)
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
                run_on_image_file(entry_path, predictor, metadata, output_dir, timestamp, show, color, mask_alpha, box_color)
            elif ext in VIDEO_EXTENSIONS:
                run_on_video_file(entry_path, predictor, metadata, output_dir, timestamp, show, color, mask_alpha, box_color)
        return

    raise FileNotFoundError(f"Path not found: {path}")


WEBCAM = "webcam"


def prompt_webcam_device_id():
    """Ask for a webcam device index. Empty input, or no stdin at all, means 0."""
    while True:
        try:
            reply = input("Enter webcam device id (0 is the default camera): ").strip()
        except EOFError:
            print("No input available; using webcam device 0.")
            return 0
        if not reply:
            return 0
        try:
            device_id = int(reply)
        except ValueError:
            print(f"Not a number: {reply!r}. Enter an integer, or press Enter for 0.")
            continue
        if device_id < 0:
            print(f"Device id must be 0 or greater, got {device_id}.")
            continue
        return device_id


def open_webcam():
    device_id = prompt_webcam_device_id()
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {device_id}")
    return cap


def main(args):
    # Resolve the input source before loading the model: the prompt and the
    # "can this camera be opened" check should not wait on GPU warm-up.
    use_webcam = str(args.path).strip().lower() == WEBCAM
    cap = open_webcam() if use_webcam else None

    cfg, used_default_weights = setup_cfg(args)
    predictor = DefaultPredictor(cfg)
    metadata = get_metadata(args, cfg, used_default_weights)

    color = tuple(c / 255 for c in args.color) if args.color is not None else None
    box_color = tuple(c / 255 for c in args.box_color) if args.box_color is not None else None

    if use_webcam:
        # Live preview only: output_path=None, and output_dir is never created.
        process_stream(
            cap, predictor, metadata, output_path=None, show=not args.no_display,
            color=color, mask_alpha=args.mask_alpha, box_color=box_color,
        )
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_on_path(
            args.path, predictor, metadata, args.output_dir, timestamp, show=not args.no_display,
            color=color, mask_alpha=args.mask_alpha, box_color=box_color,
        )


if __name__ == "__main__":
    args = load_settings("inference")
    main(args)
