import marimo

__generated_with = "0.4.1"
app = marimo.App()


@app.cell
def __():
    import marimo as mo

    EPOCHS = 3
    TEST_PICTURE = '/Users/mewald/Desktop/NVR-Eingang 19-04-2024, 12-26-24.jpg'
    return EPOCHS, TEST_PICTURE, mo


@app.cell
def __():
    import torch
    import torch.backends.mps as mps

    device_str = "mps" if mps.is_available() else "cpu"
    device = torch.device(device_str)

    print(device)
    return device, device_str, mps, torch


@app.cell
def __():
    from ultralytics import settings
    import os 
    import json

    pwd = os.path.abspath('./')
    datasets_dir = os.path.join(pwd, 'datasets')
    runs_dir = os.path.join(pwd, 'runs')

    settings_update = {
        'datasets_dir': datasets_dir,
        'runs_dir': runs_dir,     
    }

    print(json.dumps(settings_update, indent=4))

    settings.update(settings_update)
    return datasets_dir, json, os, pwd, runs_dir, settings, settings_update


@app.cell
def __():
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    return YOLO, model


@app.cell
def __(EPOCHS, model):
    def train(data, settings = {}):
        return model.train(
            data=data,
            epochs=EPOCHS, 
            device='mps',
            **settings
        )
    return train,


@app.cell
def __(model):
    import cv2
    from PIL import Image

    def infer(source, destination):
        results = model.predict(source=source)
        for r in results:
            im_bgr = r.plot() 
            im_rgb = Image.fromarray(im_bgr[..., ::-1]) 
            im_rgb.save(destination)
            return r
    return Image, cv2, infer


@app.cell
def __(train):
    coco128_results = train(data='coco128.yaml')
    return coco128_results,


@app.cell
def __(TEST_PICTURE, infer):
    before_result = infer(TEST_PICTURE, "before-train.jpg")
    for key in before_result.names:
        print(before_result.names[key])
    return before_result, key


@app.cell
def __(train):
    ourcars_results = train(data='./Our Cars.v2i.yolov8/data.yaml')
    return ourcars_results,


@app.cell
def __(TEST_PICTURE, infer):
    after_result = infer(TEST_PICTURE, "after-train.jpg")
    for k in after_result.names:
        print(after_result.names[k])
    return after_result, k


if __name__ == "__main__":
    app.run()
