import marimo

__generated_with = "0.4.1"
app = marimo.App()


@app.cell
def __():
    import marimo as mo

    TEST_PICTURE = '/Users/mewald/Desktop/NVR-Eingang 19-04-2024, 12-26-24.jpg'
    return TEST_PICTURE, mo


@app.cell(hide_code=True)
def __(mo):
    mo.md(f"""
    # Pretrained Model
    """)
    return


@app.cell
def __():
    from ultralytics import YOLO
    from ultralytics import settings

    settings.update({
        'datasets_dir': './datasets',
        'runs_dir': './runs',     
    })

    model = YOLO('yolov8n.pt')
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
    pretrained_result = infer(TEST_PICTURE, "pretrain-result.jpg")
    mo.image(src="pretrain-result.jpg")
    return pretrained_result,


@app.cell(hide_code=True)
def __(mo):
    mo.md(f"""
    # Train on Our Cars Dataset
    """)
    return


@app.cell
def __(model):
    import torch
    import torch.backends.mps as mps

    device_str = "mps" if mps.is_available() else "cpu"
    device = torch.device(device_str)

    ourcars_train_results = model.train(
        data='./Our Cars.v2i.yolov8/data.yaml',
        epochs=100,
        device=device,
        plots=True,
    )
    return device, device_str, mps, ourcars_train_results, torch


@app.cell
def __(TEST_PICTURE, infer, mo):
    ourcars_results = infer(TEST_PICTURE, "ourcars-result.jpg")
    mo.image(src="ourcars-result.jpg")
    return ourcars_results,


if __name__ == "__main__":
    app.run()
