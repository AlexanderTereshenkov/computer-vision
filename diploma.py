import os
import json
import cv2
import numpy as np
from pathlib import Path


def visualize_solo_bboxes(dataset_path: str, num_images: int = 5, sequence_id: int = 0):

    sequence_dir = Path(dataset_path) / f"sequence.{sequence_id}"
    if not sequence_dir.exists():
        print(f"Ошибка: Директория {sequence_dir} не найдена.")
        return

    processed_count = 0
    for file_path in sorted(sequence_dir.iterdir()):
        if processed_count >= num_images:
            break

        if file_path.is_file() and file_path.name.endswith(".frame_data.json"):
            print(f"\n--- Обработка: {file_path.name} ---")

            with open(file_path, 'r') as f:
                frame_data = json.load(f)

            step_prefix = file_path.stem.replace('.frame', '')
            rgb_image_path = None
            for potential_image in sequence_dir.glob(f"*.png"):
                rgb_image_path = potential_image
                break

            if rgb_image_path is None or not rgb_image_path.exists():
                print(f"  Предупреждение: Не найден RGB файл для {step_prefix}. Пропуск.")
                continue

            img = cv2.imread(str(rgb_image_path))
            if img is None:
                print(f"  Ошибка: Не удалось загрузить изображение {rgb_image_path}")
                continue
            img_height, img_width = img.shape[:2]
            print(f"  Загружено изображение: {rgb_image_path.name} ({img_width}x{img_height})")

            bboxes_to_draw = []
            if "captures" in frame_data:
                for capture in frame_data["captures"]:
                    if "annotations" in capture:
                        for annotation in capture["annotations"]:
                            if annotation.get("@type") == "type.unity.com/unity.solo.BoundingBox2DAnnotation":
                                if "values" in annotation:
                                    for bbox_data in annotation["values"]:
                                        x, y = bbox_data.get("origin", [0, 0])
                                        w, h = bbox_data.get("dimension", [0, 0])


                                        x, y, w, h = int(x), int(y), int(w), int(h)
                                        print(f"Coordinates {x}, {y}, {w}, {h}")

                                        if x < 0 or y < 0 or x + w > img_width or y + h > img_height:
                                            print(f"    Предупреждение: Bounding box ({x},{y},{w},{h}) частично за пределами изображения.")

                                        bboxes_to_draw.append({
                                            "label": bbox_data.get("labelName", "unknown"),
                                            "bbox": (x, y, w, h)
                                        })
                                        print(f"    Найден бокс: {bbox_data.get('labelName', 'N/A')} - ({x}, {y}, {w}, {h})")
            else:
                print("  В frame.json нет поля 'captures'.")

            for bbox_info in bboxes_to_draw:
                label = bbox_info["label"]
                x, y, w, h = bbox_info["bbox"]

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                label_text = label
                (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x, y - text_height - baseline), (x + text_width, y), (0, 255, 0), -1)
                cv2.putText(img, label_text, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            if bboxes_to_draw:
                print(f"Нарисовано {len(bboxes_to_draw)} боксов.")
                cv2.imshow(f"BBoxes - {step_prefix}", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Боксы для отрисовки не найдены.")

            processed_count += 1

    print(f"\nОбработка завершена. Проанализировано кадров: {processed_count}")


for i in range(101):
    visualize_solo_bboxes("D:\DiplomaDataset\signs_2", num_images=1, sequence_id=i)

