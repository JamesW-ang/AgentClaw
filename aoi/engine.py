"""
AgentClaw v6.1 — AOI 检测引擎（独立模块）

从 demo_ui.py 提取的 AOI 核心模块，包含：
  - 数据结构: DefectType, Severity, DetectionMethod, BoundingBox, Defect, InspectionResult
  - 检测器: ImagePreprocessor, TraditionalDetector, ONNXDetector
  - 主引擎: AOIEngine
  - 可视化: aoi_visualize()
  - 懒加载单例: get_aoi_engine()
  - Agent 工具注册: register_aoi_tools(), aoi_detect_for_agent()

依赖:
    pip install opencv-python numpy
    可选: pip install onnxruntime  (深度学习模式)
"""
import sys
import time
import threading
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Tuple, Optional

# 确保工作目录正确
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from core.logger import get_logger

logger = get_logger("aoi_engine")


# ============================================================
# 枚举 & 数据结构
# ============================================================

class DefectType(Enum):
    SHORT_CIRCUIT = "短路"
    OPEN_CIRCUIT = "断路"
    SOLDER_BRIDGE = "焊桥"
    MISSING_COMPONENT = "元器件缺失"
    COMPONENT_SHIFT = "元器件偏移"
    POLARITY_REVERSE = "极性反接"
    SOLDER_DEFECT = "焊点缺陷"
    SURFACE_DAMAGE = "外观损伤"
    PSEUDO_DEFECT = "伪缺陷"

class Severity(Enum):
    CRITICAL = "严重"
    MAJOR = "主要"
    MINOR = "次要"

class DetectionMethod(Enum):
    TRADITIONAL = "传统算法"
    DEEP_LEARNING = "深度学习"
    HYBRID = "混合检测"


@dataclass
class BoundingBox:
    x: int; y: int; width: int; height: int
    @property
    def area(self): return self.width * self.height
    @property
    def center(self): return (self.x + self.width // 2, self.y + self.height // 2)
    def iou(self, other):
        x1, y1 = max(self.x, other.x), max(self.y, other.y)
        x2, y2 = min(self.x+self.width, other.x+other.width), min(self.y+self.height, other.y+other.height)
        inter = max(0, x2-x1) * max(0, y2-y1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

@dataclass
class Defect:
    defect_id: str; defect_type: DefectType; severity: Severity
    bbox: BoundingBox; confidence: float; method: DetectionMethod
    description: str = ""
    def to_dict(self):
        d = asdict(self)
        d["defect_type"] = self.defect_type.value
        d["severity"] = self.severity.value
        d["method"] = self.method.value
        return d

@dataclass
class InspectionResult:
    task_id: str; product_model: str; timestamp: str
    image_size: Tuple[int, int]; defects: List[Defect] = field(default_factory=list)
    total_time_ms: float = 0.0; pass_flag: bool = True
    @property
    def defect_count(self): return len(self.defects)
    @property
    def critical_count(self): return sum(1 for d in self.defects if d.severity == Severity.CRITICAL)


# ============================================================
# 检测器
# ============================================================

class ImagePreprocessor:
    def __init__(self):
        self.config = {"clahe_clip_limit": 2.0, "clahe_grid_size": (8, 8)}

    def process(self, image, clip_limit=2.0):
        import cv2  # 延迟加载：只在使用 AOI 时才占用内存
        self.config["clahe_clip_limit"] = clip_limit
        t0 = time.perf_counter()
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        denoised = cv2.bilateralFilter(gray, 9, 15, 15)
        clahe = cv2.createCLAHE(clipLimit=self.config["clahe_clip_limit"],
                                tileGridSize=self.config["clahe_grid_size"])
        enhanced = clahe.apply(denoised)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"AOI 预处理完成: {elapsed:.1f}ms")
        return enhanced


class TraditionalDetector:
    def __init__(self):
        self.config = {"canny_low": 50, "canny_high": 150,
                       "contour_min_area": 50, "contour_max_area": 50000}

    def detect(self, image, canny_low=50, canny_high=150, min_area=50):
        import cv2
        import numpy as np  # 延迟加载
        self.config["canny_low"] = canny_low
        self.config["canny_high"] = canny_high
        self.config["contour_min_area"] = min_area
        t0 = time.perf_counter()
        defects = []
        counter = 0

        edges = cv2.Canny(image, canny_low, canny_high)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = [c for c in contours
                    if min_area < cv2.contourArea(c) < self.config["contour_max_area"]]
        logger.info(f"边缘检测候选: {len(filtered)} 个区域")

        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours2, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours2:
            areas = [cv2.contourArea(c) for c in contours2]
            median_area = np.median(areas)
            threshold_area = median_area * 4.5
            for c in contours2:
                area = cv2.contourArea(c)
                if area > threshold_area:
                    x, y, w, h = cv2.boundingRect(c)
                    counter += 1
                    defects.append(Defect(
                        defect_id=f"T-SC-{counter:04d}", defect_type=DefectType.SHORT_CIRCUIT,
                        severity=Severity.CRITICAL, bbox=BoundingBox(x, y, w, h),
                        confidence=0.85, method=DetectionMethod.TRADITIONAL,
                        description=f"疑似短路，面积={area}px"))
        mean_val = np.mean(image)
        std_val = np.std(image)
        abnormal_mask = cv2.inRange(image, 0, max(0, int(mean_val - 3 * std_val)))
        abnormal_mask2 = cv2.inRange(image, min(255, int(mean_val + 3 * std_val)), 255)
        abnormal = cv2.bitwise_or(abnormal_mask, abnormal_mask2)
        contours3, _ = cv2.findContours(abnormal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours3:
            area = cv2.contourArea(c)
            if area > min_area * 2:
                x, y, w, h = cv2.boundingRect(c)
                counter += 1
                defects.append(Defect(
                    defect_id=f"T-SD-{counter:04d}", defect_type=DefectType.SOLDER_DEFECT,
                    severity=Severity.MAJOR, bbox=BoundingBox(x, y, w, h),
                    confidence=0.72, method=DetectionMethod.TRADITIONAL,
                    description=f"焊点异常，面积={area}px"))
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"传统检测完成: {len(defects)} 个缺陷, {elapsed:.1f}ms")
        return defects


class ONNXDetector:
    AOI_NAMES = ["short_circuit", "open_circuit", "solder_defect",
                 "missing_component", "component_shift", "surface_damage"]
    DEFECT_CN = {
        "short_circuit": ("短路", Severity.CRITICAL), "open_circuit": ("断路", Severity.CRITICAL),
        "solder_defect": ("焊点缺陷", Severity.MAJOR), "solder_bridge": ("焊桥", Severity.CRITICAL),
        "missing_component": ("元器件缺失", Severity.CRITICAL),
        "component_shift": ("元器件偏移", Severity.MAJOR),
        "surface_damage": ("外观损伤", Severity.MINOR), "polarity_reverse": ("极性反接", Severity.CRITICAL),
    }

    def __init__(self):
        self.session = None; self.model_path = None; self.input_size = (640, 640)
        self.class_names = self.AOI_NAMES; self.conf_threshold = 0.5; self.iou_threshold = 0.45
        self.providers = ["CPUExecutionProvider"]
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                self.providers.insert(0, "CUDAExecutionProvider")
            logger.info(f"onnxruntime 可用, 推理后端: {self.providers}")
        except ImportError:
            logger.warning("onnxruntime 未安装，深度学习模式不可用")

    @property
    def is_loaded(self): return self.session is not None

    def load_model(self, model_path: str, class_names: Optional[List[str]] = None) -> bool:
        import onnxruntime as ort
        path = Path(model_path)
        if not path.exists():
            return False
        try:
            if self.session is not None:
                del self.session
            self.session = ort.InferenceSession(str(path), providers=self.providers)
            self.model_path = str(path)
            input_info = self.session.get_inputs()[0]
            input_shape = input_info.shape
            if len(input_shape) == 4 and input_shape[2] != "N" and input_shape[3] != "N":
                h, w = input_shape[2], input_shape[3]
                if isinstance(h, int) and isinstance(w, int):
                    self.input_size = (w, h)
            if class_names:
                self.class_names = class_names
            logger.info(f"模型加载成功: {path.name}, 尺寸={self.input_size}")
            return True
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False

    def _preprocess(self, image):
        import cv2
        import numpy as np  # 延迟加载
        h, w = image.shape[:2]
        th, tw = self.input_size
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pw, ph = tw - nw, th - nh
        top, bottom = ph // 2, ph - ph // 2
        left, right = pw // 2, pw - pw // 2
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(tensor, axis=0), scale, (left, top)

    def _postprocess(self, outputs, scale, pad, orig_size):
        import numpy as np  # 延迟加载
        predictions = []
        raw = outputs[0] if len(outputs.shape) >= 2 else outputs
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T
        nc = len(self.class_names)
        for det in raw:
            if len(det) < 5 + nc:
                continue
            cx, cy, w, h = det[:4]
            obj_conf = float(det[4])
            if obj_conf < self.conf_threshold:
                continue
            scores = det[5:5+nc]
            cid = int(np.argmax(scores))
            final_conf = obj_conf * float(scores[cid])
            if final_conf < self.conf_threshold:
                continue
            x, y = (cx - w/2 - pad[0]) / scale, (cy - h/2 - pad[1]) / scale
            w, h = w / scale, h / scale
            x = max(0, min(x, orig_size[1])); y = max(0, min(y, orig_size[0]))
            w = min(w, orig_size[1] - x); h = min(h, orig_size[0] - y)
            predictions.append({"x": float(x), "y": float(y), "w": float(w), "h": float(h),
                                "confidence": final_conf, "class_id": cid,
                                "class_name": self.class_names[cid] if cid < len(self.class_names) else f"class_{cid}"})
        # NMS
        boxes = sorted(predictions, key=lambda b: b["confidence"], reverse=True)
        kept = []
        for box in boxes:
            keep = True
            for k in kept:
                x1 = max(box["x"], k["x"]); y1 = max(box["y"], k["y"])
                x2 = min(box["x"]+box["w"], k["x"]+k["w"]); y2 = min(box["y"]+box["h"], k["y"]+k["h"])
                inter = max(0, x2-x1) * max(0, y2-y1)
                union = box["w"]*box["h"] + k["w"]*k["h"] - inter
                if union > 0 and inter / union > self.iou_threshold:
                    keep = False; break
            if keep:
                kept.append(box)
        return kept

    def detect(self, image, conf_thresh=0.5, iou_thresh=0.45):
        self.conf_threshold = conf_thresh; self.iou_threshold = iou_thresh
        if not self.is_loaded:
            return []
        t0 = time.perf_counter()
        tensor, scale, pad = self._preprocess(image)
        orig_size = image.shape[:2]
        outputs = self.session.run([self.session.get_outputs()[0].name],
                                   {self.session.get_inputs()[0].name: tensor})[0]
        preds = self._postprocess(outputs, scale, pad, orig_size)
        defects = []
        for i, p in enumerate(preds):
            name = p["class_name"]
            cn_name, severity = self.DEFECT_CN.get(name, (name, Severity.MINOR))
            dtype = DefectType.PSEUDO_DEFECT
            for dt in DefectType:
                if dt.value == cn_name or dt.value == name:
                    dtype = dt; break
            defects.append(Defect(
                defect_id=f"D-{i+1:04d}", defect_type=dtype, severity=severity,
                bbox=BoundingBox(int(p["x"]), int(p["y"]), int(p["w"]), int(p["h"])),
                confidence=p["confidence"], method=DetectionMethod.DEEP_LEARNING,
                description=f"模型检测: {name} ({cn_name}), conf={p['confidence']:.3f}"))
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"ONNX 推理完成: {len(defects)} 个缺陷, {elapsed:.1f}ms")
        return defects


# ============================================================
# 主引擎
# ============================================================

class AOIEngine:
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.traditional = TraditionalDetector()
        self.onnx_detector = ONNXDetector()
        self.mode = "traditional"

    def inspect(self, image, canny_low=50, canny_high=150, clahe_clip=2.0,
                min_area=50, conf_thresh=0.5, iou_thresh=0.45):
        from datetime import datetime
        t_start = time.perf_counter()
        processed = self.preprocessor.process(image, clip_limit=clahe_clip)
        if self.mode == "deeplearning" and self.onnx_detector.is_loaded:
            defects = self.onnx_detector.detect(image, conf_thresh, iou_thresh)
        elif self.mode == "hybrid" and self.onnx_detector.is_loaded:
            defects = self.traditional.detect(processed, canny_low, canny_high, min_area)
            dl = self.onnx_detector.detect(image, conf_thresh, iou_thresh)
            defects = defects + dl
        else:
            defects = self.traditional.detect(processed, canny_low, canny_high, min_area)
        # 去重
        merged = []
        for d in defects:
            dup = False
            for m in merged:
                if d.defect_type == m.defect_type and d.bbox.iou(m.bbox) > 0.5:
                    if d.confidence > m.confidence:
                        merged.remove(m); merged.append(d)
                    dup = True; break
            if not dup:
                merged.append(d)
        total_ms = (time.perf_counter() - t_start) * 1000
        method_label = {"traditional": "传统算法", "deeplearning": "深度学习", "hybrid": "混合检测"}.get(self.mode, self.mode)
        result = InspectionResult(
            task_id=f"AOI-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            product_model=method_label,
            timestamp=datetime.now().isoformat(),
            image_size=(image.shape[1], image.shape[0]),
            defects=merged, total_time_ms=total_ms,
            pass_flag=len([d for d in merged if d.severity == Severity.CRITICAL]) == 0,
        )
        return processed, result


# ============================================================
# 可视化
# ============================================================

def aoi_visualize(image, result):
    import cv2  # 延迟加载
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    color_map = {Severity.CRITICAL: (0, 0, 255), Severity.MAJOR: (0, 165, 255), Severity.MINOR: (0, 255, 255)}
    for defect in result.defects:
        color = color_map.get(defect.severity, (255, 255, 255))
        b = defect.bbox
        cv2.rectangle(vis, (b.x, b.y), (b.x+b.width, b.y+b.height), color, 2)
        label = f"{defect.defect_type.value} {defect.confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        ly = max(b.y - 5, th + 5)
        cv2.rectangle(vis, (b.x, ly-th-3), (b.x+tw+4, ly+2), color, -1)
        cv2.putText(vis, label, (b.x+2, ly-2), font, 0.5, (255,255,255), 1)
    status_text = "PASS" if result.pass_flag else "FAIL"
    sc = (0, 200, 0) if result.pass_flag else (0, 0, 255)
    cv2.putText(vis, status_text, (image.shape[1]-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, sc, 2)
    return vis


# ============================================================
# 懒加载单例
# ============================================================

_aoi_engine = None
_aoi_lock = threading.Lock()


def get_aoi_engine():
    """延迟创建 AOI 引擎，首次使用时才加载 cv2/numpy"""
    global _aoi_engine
    if _aoi_engine is None:
        with _aoi_lock:
            if _aoi_engine is None:
                _aoi_engine = AOIEngine()
                logger.info("AOI 引擎已延迟初始化")
    return _aoi_engine


# ============================================================
# Agent 工具注册
# ============================================================

def aoi_detect_for_agent(image_path: str, mode: str = "traditional",
                         canny_low: int = 50, canny_high: int = 150,
                         clahe_clip: float = 2.0, min_area: int = 50,
                         conf_thresh: float = 0.5, iou_thresh: float = 0.45) -> dict:
    """AOI 检测工具 (供 Agent 调用)"""
    import cv2
    import numpy as np

    path = Path(image_path)
    if not path.exists():
        return {"error": f"图片不存在: {image_path}"}

    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": f"图片读取失败: {image_path}"}

    engine = get_aoi_engine()
    needs_model = mode in ("deeplearning", "hybrid")
    if needs_model and not engine.onnx_detector.is_loaded:
        return {"error": "深度学习模式需要加载 ONNX 模型，请先加载模型或使用传统模式"}

    engine.mode = mode
    with _aoi_lock:
        processed, result = engine.inspect(
            img, canny_low=canny_low, canny_high=canny_high,
            clahe_clip=clahe_clip, min_area=min_area,
            conf_thresh=conf_thresh, iou_thresh=iou_thresh)

    # Format result for Agent consumption
    defects_info = []
    for d in result.defects:
        defects_info.append({
            "id": d.defect_id,
            "type": d.defect_type.value,
            "severity": d.severity.value,
            "confidence": round(d.confidence, 3),
            "method": d.method.value,
            "location": f"({d.bbox.x},{d.bbox.y},{d.bbox.width},{d.bbox.height})",
            "description": d.description,
        })

    return {
        "task_id": result.task_id,
        "pass": result.pass_flag,
        "total_defects": result.defect_count,
        "critical_defects": result.critical_count,
        "detection_time_ms": round(result.total_time_ms, 1),
        "image_size": list(result.image_size),
        "mode": result.product_model,
        "defects": defects_info,
        "summary": f"检测结果: {'合格' if result.pass_flag else '不合格'}，"
                   f"缺陷总数={result.defect_count}，严重缺陷={result.critical_count}，"
                   f"耗时={result.total_time_ms:.0f}ms",
    }


def register_aoi_tools():
    """注册 AOI 检测工具到 tool_registry (让 Agent 可以调用 AOI 检测)"""
    from tools.registry import registry, ToolCategory

    registry.register_func(
        aoi_detect_for_agent,
        name="aoi_detect",
        description=(
            "AOI 电路板缺陷检测。上传 PCB 图片进行自动缺陷检测，"
            "支持传统算法/深度学习/混合模式。返回缺陷列表和判定结果。"
            "可基于检测结果进行智能分析和调参建议。"
        ),
        parameters=[
            {"name": "image_path", "type": "string", "description": "PCB 图片文件路径", "required": True},
            {"name": "mode", "type": "string", "description": "检测模式: traditional/deeplearning/hybrid", "required": False, "default": "traditional"},
            {"name": "canny_low", "type": "number", "description": "Canny边缘检测低阈值", "required": False, "default": 50},
            {"name": "canny_high", "type": "number", "description": "Canny边缘检测高阈值", "required": False, "default": 150},
            {"name": "clahe_clip", "type": "number", "description": "CLAHE对比度限制", "required": False, "default": 2.0},
            {"name": "min_area", "type": "number", "description": "最小缺陷面积", "required": False, "default": 50},
            {"name": "conf_thresh", "type": "number", "description": "深度学习置信度阈值", "required": False, "default": 0.5},
            {"name": "iou_thresh", "type": "number", "description": "NMS IoU阈值", "required": False, "default": 0.45},
        ],
        category=ToolCategory.CUSTOM,
        timeout=60,
    )
    logger.info("AOI 检测工具已注册到 tool_registry")
