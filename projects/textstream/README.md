# Train a GPT model in the browser using text streaming

## Using WebGPU

Tested on Chrome 115.0.5790.40 beta (64-bit) on 2023/07/27

- `google-chrome-beta --enable-unsafe-webgpu --enable-features=Vulkan http://localhost:8080`
- These flags will not be needed in the future. Check the WebGPU support (for example [here](https://mdn.github.io/dom-examples/webgpu-render-demo/))

## Tracking experiments with MLflow

By default a browser will not be able to send metrics to a MLflow server running on a separate port (because of [CORS](https://en.wikipedia.org/wiki/Cross-origin_resource_sharing)). One way to overcome this is to create a CORS-proxy (using [local-cors-proxy](https://github.com/garmeeh/local-cors-proxy) or similar tools). Another option is disabling web security:

- `mlflow server`
- `google-chrome-beta --disable-web-security --user-data-dir="./temp" http://localhost:8080`

## WebGPU + MLflow

- `google-chrome-beta --disable-web-security --user-data-dir="./temp" --enable-unsafe-webgpu --enable-features=Vulkan http://localhost:8080`
