import marimo

__generated_with = "0.4.1"
app = marimo.App()


@app.cell
def __(__file__):
    import marimo as mo
    import os

    MODEL_VERSION = "yolov8x.pt"
    TEST_PICTURE = 'test.jpg'
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    return BASE_DIR, MODEL_VERSION, TEST_PICTURE, mo, os


@app.cell(hide_code=True)
def __(MODEL_VERSION, mo):
    mo.md(f"""
    # Pretrained Model: {MODEL_VERSION}
    """)
    return


@app.cell
def __(MODEL_VERSION):
    from ultralytics import YOLO
    from ultralytics import settings

    settings.update({
        'datasets_dir': './datasets',
        'runs_dir': './runs',     
    })

    model = YOLO(MODEL_VERSION)
    return YOLO, model, settings


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
def __(TEST_PICTURE, infer, mo):
    pretrained_result = infer(TEST_PICTURE, "tmp/pretrain-result.jpg")
    mo.image(src="tmp/pretrain-result.jpg")
    return pretrained_result,


@app.cell(hide_code=True)
def __(mo):
    mo.md(f"""
    # Train on Our Cars Dataset
    """)
    return


@app.cell
def __(BASE_DIR, model, os):
    import torch
    import torch.backends.mps as mps

    device_str = "mps" if mps.is_available() else "cpu"
    device = torch.device(device_str)

    ourcars_train_results = model.train(
        data=os.path.join(BASE_DIR, 'datasets/ourcars/data.yaml'),
        epochs=100,
        device=device,
        plots=True,

        degrees=20,
        translate=0.3
    )
    return device, device_str, mps, ourcars_train_results, torch


@app.cell
def __(TEST_PICTURE, infer, mo):
    ourcars_results = infer(TEST_PICTURE, "tmp/ourcars-result.jpg")
    mo.image(src="tmp/ourcars-result.jpg")
    return ourcars_results,


if __name__ == "__main__":
    app.run()
