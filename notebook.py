import marimo

__generated_with = "0.4.1"
app = marimo.App()


@app.cell
def __():
    import marimo as mo

    TEST_PICTURE = '/Users/mewald/Desktop/NVR-Eingang 19-04-2024, 12-26-24.jpg'
    return TEST_PICTURE, mo


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
def __(model):
    import cv2
    from PIL import Image

    def infer(source, destination):
        results = model(source)
        for r in results:
            im_bgr = r.plot() 
            im_rgb = Image.fromarray(im_bgr[..., ::-1]) 
            im_rgb.save(destination)
            return r
    return Image, cv2, infer


@app.cell
def __(TEST_PICTURE, infer):
    pretrained_result = infer(TEST_PICTURE, "pretrain-result.jpg")
    return pretrained_result,


@app.cell
def __(model):
    ourcars_train_results = model.train(
        data='./Our Cars.v2i.yolov8/data.yaml',
        epochs=100
    )
    return ourcars_train_results,


@app.cell
def __(TEST_PICTURE, infer):
    ourcars_results = infer(TEST_PICTURE, "ourcars-result.jpg")
    return ourcars_results,


if __name__ == "__main__":
    app.run()
