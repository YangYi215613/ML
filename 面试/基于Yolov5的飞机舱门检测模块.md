# 集成 YOLOv5 的飞机舱门自动检测模块（面试版）

> **技术栈**：Python · PyTorch · YOLOv5 · RTSP · TensorRT · WebSocket（Django/Channels 或 FastAPI）· Docker

------

## 1) 电梯陈述（30–60 秒）

- 我负责一个“从数据到上线”的舱门自动检测模块：自建数据集 → 微调 YOLOv5 → TensorRT 加速 → RTSP 实时推理 → WebSocket 推送到前端。
- 难点与亮点：
  - **域泛化**：针对光照/角度/背景差异做数据增广与 Hard Negative；
  - **轻量低延迟**：YOLOv5s + TensorRT FP16，单路 1080p 实时 40–60 FPS；
  - **工程化**：RTSP 解码 + 异步推理队列 + WebSocket 广播；Docker 一键部署。
- 结果：舱门/闭合状态 mAP@0.5≈**0.93+**（示例），端到端延迟 **<120 ms**（示例，取决于硬件/码率）。

------

## 2) YOLOv5 选择与特性简述（面试可讲）

- **单阶段目标检测**：End-to-End 回归框 + 分类，速度快，适用于实时场景。
- **网络结构**：CSPDarknet 主干 + PANet 颈部 + YOLO 头；支持 s/m/l/x 多尺度模型；可切换 **NMS**、**IoU 损失** 等配置。
- **数据增广**：默认集成 Mosaic、HSV 色彩扰动、Random Affine、MixUp 等；
- **训练生态完善**：自动采用 **AutoAnchor**、**EMA**、标签平滑、Warmup Cosine LR；
- **导出/部署友好**：一行命令导出 ONNX/TorchScript/TF/TensorRT；
- **实用工具**：内置评估、混淆矩阵、PR 曲线、mAP 指标、错误分析、可视化；
- **社区/文档**：成熟稳定、案例丰富，易于复用与二开。

------

## 3) 业务目标与数据

**检测任务**：

- 类别：`door`（舱门）与 `door_closed`（闭合状态，可做成属性或二分类/多标签）
- 输出：`bbox[x1,y1,x2,y2] + class + score`，可扩展 `is_closed` 属性或多任务头

**数据流程**：

1. 采集机坪/廊桥不同光照、角度、机型、天气的图片与视频帧；
2. 用 **LabelImg** 标注，导出 YOLO TXT（class cx cy w h）；
3. 训练集/验证集/测试集（如 7/2/1），保证场景与机型分布一致；
4. 建立 **Hard Negative**（无舱门场景/遮挡），提升误检鲁棒性。

------

## 4) 训练与评估（命令示例）

> 以官方 YOLOv5 代码库为例（结构可按团队规范调整）。

```bash
# 1) 环境
pip install torch torchvision # 根据显卡及 CUDA 选择版本
pip install -r yolov5/requirements.txt

# 2) 数据配置（data/door.yaml）
# path: datasets/door
# train: images/train
# val: images/val
# test: images/test
# names: [door, door_closed]

# 3) 训练（以 s 模型为例）
python yolov5/train.py \
  --img 1280 --batch 16 --epochs 120 \
  --data data/door.yaml \
  --cfg yolov5s.yaml \
  --hyp data/hyps/hyp.scratch-high.yaml \
  --project runs/door --name v5s_doors --exist-ok

# 4) 评估
python yolov5/val.py --img 1280 --data data/door.yaml \
  --weights runs/door/v5s_doors/weights/best.pt --task test

# 5) 导出 ONNX/TensorRT
python yolov5/export.py --weights runs/door/v5s_doors/weights/best.pt \
  --include onnx engine --img 1280 --batch 1 --half
```

**训练策略要点**：

- **增广**：适当调高亮度/对比度抖动；加入 Motion Blur/雨雪/反光合成增强；
- **Anchor/输入尺寸**：机体比例长宽差异大时，开启 AutoAnchor；输入尺寸 960/1280/1536 做权衡；
- **Class/Attribute 设计**：闭合状态可做第二类别（door_closed）或用属性头（需改 Head）。

------

## 5) TensorRT 加速（示例）

- 导出 `best.engine`（FP16/INT8 需校准集）；
- 推理时使用 TensorRT runtime + CUDA stream；

```python
# infer_trt.py (简化示例)
import cv2, numpy as np
import tensorrt as trt, pycuda.driver as cuda, pycuda.autoinit

class TRTDetector:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.input_idx = self.engine.get_binding_index('images')
        self.output_idx = self.engine.get_binding_index('output0')  # 视导出命名
        self.stream = cuda.Stream()
        self.input_shape = self.engine.get_binding_shape(self.input_idx)  # (1,3,H,W)
        self.d_inputs = cuda.mem_alloc(trt.volume(self.input_shape) * np.float16().nbytes)
        self.d_outputs = cuda.mem_alloc(8400 * 85 * np.float16().nbytes)  # 视模型而定

    def preprocess(self, img):
        # letterbox + BGR->RGB + CHW + /255 省略，返回 float16
        pass

    def __call__(self, img):
        blob = self.preprocess(img)
        cuda.memcpy_htod_async(self.d_inputs, blob, self.stream)
        self.ctx.execute_async_v2([int(self.d_inputs), int(self.d_outputs)], self.stream.handle)
        out = np.empty((8400,85), dtype=np.float16)  # 视模型输出
        cuda.memcpy_dtoh_async(out, self.d_outputs, self.stream)
        self.stream.synchronize()
        return out  # 后处理 NMS 略
```

> 生产中建议直接复用 YOLOv5 的 `export.py` + `trt.py` 风格的后处理，降低踩坑成本。

------

## 6) RTSP 实时推理 + WebSocket 推送

**流程**：RTSP 解码（OpenCV/GStreamer）→ 预处理 → TensorRT 推理 → NMS → 绘制 → WebSocket 推送。

```python
# rtsp_ws.py (FastAPI WebSocket 版简化)
import asyncio, cv2, base64
import numpy as np
from fastapi import FastAPI, WebSocket
from infer_trt import TRTDetector

app = FastAPI()
DETECTOR = TRTDetector('best.engine')

async def encode_frame(frame):
    _, buf = cv2.imencode('.jpg', frame)
    return base64.b64encode(buf.tobytes()).decode()

@app.websocket('/ws/stream')
async def ws_stream(ws: WebSocket):
    await ws.accept()
    cap = cv2.VideoCapture('rtsp://user:pwd@ip:554/stream1')
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                await asyncio.sleep(0.02); continue
            preds = DETECTOR(frame)  # TODO: 后处理成 boxes/classes/scores
            # draw_boxes(frame, preds)
            img64 = await encode_frame(frame)
            await ws.send_json({"type":"frame","img": img64, "det": []})
            await asyncio.sleep(0)  # 让出事件循环
    finally:
        cap.release(); await ws.close()
```

> 若使用 **Django**：建议接入 **Django Channels**（Redis Channel Layer），消费者中复用上面逻辑。

------

## 7) Django/Channels 端点（可选）

```python
# consumers.py (简化)
from channels.generic.websocket import AsyncJsonWebsocketConsumer
class StreamConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        await self.accept()
    async def send_frame(self, event):
        await self.send_json(event["payload"])  # 由后台推送
# routing.py
from django.urls import re_path
from .consumers import StreamConsumer
websocket_urlpatterns = [re_path(r'ws/stream/$', StreamConsumer.as_asgi())]
```

------

## 8) Docker 与部署

```yaml
# docker-compose.yml (精简)
version: '3.9'
services:
  api:
    build: ./svc
    ports: ["8000:8000"]
    environment:
      - TRT_ENGINE=/models/best.engine
    volumes:
      - ./models:/models:ro
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
  nginx:
    image: nginx:alpine
    volumes:
      - ./deploy/nginx.conf:/etc/nginx/conf.d/default.conf:ro
    ports: ["80:80"]
```

```dockerfile
# svc/Dockerfile
FROM nvcr.io/nvidia/pytorch:24.02-py3
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"
```

------

## 9) 质量与性能保障

- **鲁棒性**：
  - 类别混淆分析（door vs door_closed），误检热区回填训练；
  - 采集极端光照/背光/强反光；遮挡与小目标专项增广；
- **性能**：
  - 推理：TensorRT FP16；批次=1；尽量避免 CPU↔GPU 往返；
  - 解码：GStreamer 硬解；多路流采用异步队列 + 批量 NMS；
- **监控**：
  - 导出 mAP、FPS、端到端延迟；异常帧回放与误报追踪；
- **可维护性**：
  - 配置化（模型路径/码率/阈值）；统一日志与错误码；
  - 模块化（decoder/infer/postprocess/transport）。

------

## 10) Demo cURL / Web 页面接入

```bash
# WebSocket 调试（浏览器控制台）
const ws = new WebSocket('ws://host/ws/stream');
ws.onmessage = (e)=>{ const d=JSON.parse(e.data); /* d.img 为 base64 */ }
```

------

## 11) 面试 Q&A 备选

- **为何选 YOLOv5 而非 YOLOv8/RT-DETR？**：社区成熟、部署路径清晰（TRT 资料多）、对实时性友好；若追求 SOTA 可评估升级但需验证稳定性与工程成本。
- **如何降低误检/漏检？**：硬样本挖掘、阈值与 NMS 调参、增加负类、区域感兴趣（ROI）过滤、时序一致性（多帧投票）。
- **端到端延迟优化点？**：RTSP 解码硬件加速、Pinned Memory、零拷贝、合并后处理、异步 WebSocket 发送。
- **如何做闭合状态判断？**：
  - 多类别（`door`/`door_closed`）简易直接；
  - 单类 + 属性分类头更灵活，但需改 Head 与标签。
- **多路摄像头扩展？**：每路独立解码协程 + 共享推理池；根据 GPU 利用率动态分配。

