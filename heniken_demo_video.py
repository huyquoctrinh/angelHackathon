import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import threading

# model = YOLO('hen3.pt')
nap_model = YOLO('nap_h.pt')
hen_model = YOLO('hen3.pt')
person_model = YOLO('yolov8n.pt')
nap_label = {
    0: "dong", 
    1: "mo"
}
hen_label = {
    0:"heineken can"
}
person_lb = {
    0:"person"
}
lock = threading.Lock()

if 'boxNumber' not in st.session_state:
    st.session_state.boxNumber = 0

class YOLOTransformer(VideoTransformerBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        hen_results = hen_model([frame], imgsz = 640, conf=0.2, iou = 0.7, nms = True)
        # nap_res = nap_model([frame], imgsz = 640, conf=0.25, iou = 0.7, nms = True)
        # person_res = person_model([frame], imgsz = 640, conf=0.25, iou = 0.7, nms = True)
        print(hen_results)
        for r in hen_results:
            annotator = Annotator(img)
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                c = box.cls
                annotator.box_label(b, hen_model.names[int(c)])
            img = annotator.result()
        
        # for r in nap_res:
        #     annotator = Annotator(img)
        #     boxes = r.boxes
        #     for box in boxes:
        #         b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
        #         c = box.cls
        #         lb_nap = self.nap_label[int(c)]
        #         annotator.box_label(b, lb_nap)
        #     # img = annotator.result()
        
        # for r in person_res:
        #     annotator = Annotator(img)
        #     boxes = r.boxes
        #     for box in boxes:
        #         b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
        #         c = box.cls
        #         # lb_nap = self.nap_label[int(c)]
        #         annotator.box_label(b, "person")
        #     # img = annotator.result()
        # img = annotator.result()
        return img

# detectCansModel = YOLO('hen3.pt')
# detectCoverCanModel = YOLO('nap_h.pt')
# countPersonModel = YOLO('yolov8n.pt')

st.title("Heineken Demo with Video Webcam")

# aiMode = st.radio("Chọn Mode", ["Đếm số lon Heineken", "Đếm nắp lon", "Đếm số người"])

# if aiMode == "Đếm số lon Heineken":
#     model = YOLO('hen3.pt')
# elif aiMode == "Đếm nắp lon":
#     model = YOLO('nap_h.pt')
# elif aiMode == "Đếm số người":
#     model = YOLO('yolov8n.pt')

# model_factory = YOLOTransformer
webrtc_streamer(
    key="yolo", 
    video_processor_factory=YOLOTransformer,
        rtc_configuration={  # Add this config
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )

