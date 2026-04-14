import os
import json
from PIL import Image, ImageDraw
import argparse

def draw_bboxes(root_dir, limit=5):
    """
    Recursively find _annotations.coco.json files and draw bounding boxes on images.
    """
    for subdir, dirs, files in os.walk(root_dir):
        if "annotations.json" in files:
            print(f"Processing folder: {subdir}")
            json_path = os.path.join(subdir, "annotations.json")
            output_dir = os.path.join(subdir, "annotated_samples")
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(output_dir + '/JPEGImages', exist_ok=True)

            with open(json_path, 'r') as f:
                data = json.load(f)

            # Map image_id to list of annotations
            img_to_anns = {}
            for ann in data.get('annotations', []):
                img_id = ann['image_id']
                if img_id not in img_to_anns:
                    img_to_anns[img_id] = []
                img_to_anns[img_id].append(ann)

            # Map category_id to category name
            categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}

            # Process images
            count = 0
            for img_info in data.get('images', []):
                if limit and count >= limit:
                    break

                img_id = img_info['id']
                file_name = img_info['file_name']
                img_path = os.path.join(subdir, file_name)

                if not os.path.exists(img_path):
                    # Try extra['name'] if file_name doesn't exist
                    extra_name = img_info.get('extra', {}).get('name')
                    if extra_name:
                        img_path = os.path.join(subdir, extra_name)

                if not os.path.exists(img_path):
                    print(f"Warning: Image not found at {img_path}")
                    continue

                try:
                    img = Image.open(img_path).convert("RGB")
                    draw = ImageDraw.Draw(img)
                    
                    # Colors for different classes (just a few defaults)
                    colors = ["#39FF14", "#FF3131", "#1F51FF", "#FFFF33", "#BC13FE"]

                    if img_id in img_to_anns:
                        for ann in img_to_anns[img_id]:
                            bbox = ann['bbox'] # [x, y, w, h]
                            cat_id = ann['category_id']
                            cat_name = categories.get(cat_id, "unknown")
                            
                            x, y, w, h = bbox
                            draw.rectangle([x, y, x + w, y + h], outline=colors[cat_id % len(colors)], width=3)
                            
                            # Draw label
                            draw.text((x, y - 10), f"{cat_name}", fill=colors[cat_id % len(colors)])

                    output_path = os.path.join(output_dir, file_name)
                    img.save(output_path)
                    print(f"Saved: {output_path}")
                    count += 1
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw bounding boxes on COCO datasets.")
    parser.add_argument("--limit", type=int, default=0, help="Max images per folder (default: 5, set to 0 for all)")
    parser.add_argument("--dir", type=str)
    args = parser.parse_args()

    draw_bboxes(args.dir, limit=args.limit)
