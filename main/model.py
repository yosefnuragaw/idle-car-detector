
import cv2
import numpy as np
from collections import defaultdict
from datetime import datetime
from skimage.metrics import structural_similarity as compare_ssim


class IdleTracker:
    def __init__(self,
                 iou_treshold = 0.5, 
                 nms_threshold =0.5,
                 ssim_treshold = 0.5,
                 confidence_treshold = 0.5,
                 exception = 1,
                 move_treshold = 1,
                 stop_duration = 0.5,
                 alpha = 0.01,
                 short_term_treshold = 50,
                 long_term_treshold = 50,
                 kernel_noise = (1,1),
                 kernel_close = (1,1)
                 ):
        """
        iou_treshold : treshold score to compute IoU\n
        nms_threshold : treshold score to compute NMS\n
        ssim_treshold : treshold score to apply SSIM\n
        confidence_treshold :Minimum detection score\n
        motion_treshold :Maximum allowed motion distance for a match\n
        short_term_treshold : Threshold for short-term changes\n
        long_term_treshold : Threshold for long-term changes\n 
        alpha : Running average factor for long-term memory\n
        stop_duration : Duration of vehicle detected as idle\n
        """
        self.iou_treshold           = iou_treshold 
        self.nms_treshold           = nms_threshold
        self.ssim_treshold          = ssim_treshold
        self.confidence_treshold    = confidence_treshold
        self.exception              = exception
        self.move_treshold          = move_treshold
        self.stop_duration          = stop_duration  # Time in seconds to consider an object as stopped
        self.alpha                  = alpha  # Running average factor for long-term memory
        self.short_term_treshold    = short_term_treshold  # Threshold for short-term changes
        self.long_term_treshold     = long_term_treshold  # Threshold for long-term changes 
        self.kernel_noise           =  cv2.getStructuringElement(cv2.MORPH_RECT, kernel_noise)
        self.kernel_close           =  cv2.getStructuringElement(cv2.MORPH_RECT, kernel_close)

    def compute_ssim(self,roi_1, roi_2):
        """
        Compute structural similarity index (SSIM)\n
        Return condition and SSIM score of two ROI
        """
        if 0 in roi_1.shape or 0 in roi_2.shape:
            return False,0
        gray_roi1 = cv2.cvtColor(roi_1, cv2.COLOR_BGR2GRAY)
        gray_roi2 = cv2.cvtColor(roi_2, cv2.COLOR_BGR2GRAY)
        ssim = compare_ssim(gray_roi1, gray_roi2)
        if ssim >= self.ssim_treshold:
            return True,ssim 
        else:
            return False,ssim
        
    def inside_of(self,roi1,roi2):
        cond1 = roi1[0] <= roi2[0] and roi1[1] <= roi2[1] and roi1[2] >= roi2[2] and roi1[3] >= roi2[3]
        cond2 = roi2[0] <= roi1[0] and roi2[1] <= roi1[1] and roi2[2] >= roi1[2] and roi2[3] >= roi1[3]
        return cond1 or cond2

        
    def compute_iou(self,roi_1, roi_2):
        """
        Compute Intersection over Union (IoU)\n
        Return IoU score of two ROI
        """
        x1 = max(roi_1[0], roi_2[0])
        y1 = max(roi_1[1], roi_2[1])
        x2 = min(roi_1[2], roi_2[2])
        y2 = min(roi_1[3], roi_2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_roi_1 = (roi_1[2] - roi_1[0]) * (roi_1[3] - roi_1[1])
        area_roi_2 = (roi_2[2] - roi_2[0]) * (roi_2[3] - roi_2[1])
        union = area_roi_1 + area_roi_2 - intersection

        if union == 0:
            return 0
        return intersection/union

    def apply_nms(self,roi, roi_confidences):
        """
        Apply  non-maximum suppression (NMS) on detection result\n
        Return highest ROI with the highest score
        """
        indices = cv2.dnn.NMSBoxes(roi, roi_confidences, score_threshold=self.confidence_treshold, nms_threshold=self.nms_treshold)
        return [roi[i[0] if isinstance(i, (list, np.ndarray)) else i] for i in indices]

    def initialize_yolo(self,weights_path,cfg_path):
        self.net = cv2.dnn.readNet(weights_path, cfg_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.layer_names = self.net.getLayerNames()
        self.output_layer_indices = self.net.getUnconnectedOutLayers().flatten()
        self.output_layers = [self.layer_names[i-1] for i in self.output_layer_indices]

    def compute_foreground(self,short_term_frame):
        """
        Compute masking using foreground truth\n
        Return stable region
        """
        self.gray2 = cv2.cvtColor(short_term_frame, cv2.COLOR_BGR2GRAY)

        # Short-term memory (frame differencing)
        diff_short_term = cv2.absdiff(self.gray1, self.gray2)
        _, thresh_short_term = cv2.threshold(diff_short_term, self.short_term_treshold, 255, cv2.THRESH_BINARY)

        # Update long-term memory (running average)
        cv2.accumulateWeighted(self.gray2, np.float32(self.gray1), self.alpha)
        long_term_bg_uint8 = cv2.convertScaleAbs(self.gray1)
        diff_long_term = cv2.absdiff(self.gray2, long_term_bg_uint8)
        _, thresh_long_term = cv2.threshold(diff_long_term, self.long_term_treshold, 255, cv2.THRESH_BINARY)

        # Invert the combined mask to get the stable regions
        clean_short = cv2.morphologyEx(thresh_short_term, cv2.MORPH_OPEN, self.kernel_noise)
        clean_long = cv2.morphologyEx(thresh_long_term, cv2.MORPH_OPEN, self.kernel_noise)
        combined_thresh = cv2.bitwise_or(clean_short, clean_long)
        closing = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, self.kernel_close)  # Closing operation
        stable_regions_mask = cv2.bitwise_not(closing)

        if self.night:
            short_term_frame = self.preprocess_night_scene(short_term_frame)
        # Extracting the stable regions using the inverted mask
        stable_regions = cv2.bitwise_and(short_term_frame, short_term_frame, mask=stable_regions_mask)
        return stable_regions,thresh_long_term,thresh_short_term,combined_thresh,closing
    
    def update_tracker(self,stable_regions,frame):
        """
        Update tracker for detected object\n
        Delete tracker if obejct is moved of lost
        """
        self.updated_boxes = {}
        for obj_id, tracker in list(self.trackers.items()):
            success, box = tracker.update(stable_regions)
            tracked = True
            time_stationary = self.current_objects[obj_id]['idle']
            if success:
                x, y, w, h = [int(v) for v in box]
                a, b, c, d = [int(v) for v in self.current_objects[obj_id]['bbox']]
                bbox = (x, y, x+w, y+h)
                if self.frame_count % (self.frame_skip//self.refresh_tracker) == 0:
                    tracked = False
                    roi_new = stable_regions[y:y+h, x:x+w]
                    roi_old = self.current_objects[obj_id]['long_term'][y:y+h, x:x+w]
                    dx = abs(a - x)
                    dy = abs(b - y)
                    # try:
                    cond,score = self.compute_ssim(roi_old,roi_new)
                    if cond and(dx <= self.move_treshold and dy <= self.move_treshold): 
                        tracked = True
                        self.current_objects[obj_id]['exception'] =0
                    elif time_stationary >= self.stop_duration-1 and (dx <= self.move_treshold and dy <= self.move_treshold):
                            if self.current_objects[obj_id]['exception'] <  self.exception:
                               
                                self.current_objects[obj_id]['exception'] +=1
                                tracked = True

                
                if tracked:
                    if (bbox[2] > 0 and bbox[0] < self.width and bbox[3] > 0 and bbox[1] < self.height ): 
                            self.updated_boxes[obj_id] = bbox
                            if  time_stationary >= self.stop_duration-1:
                                if self.current_objects[obj_id]['id_park'] == 0:
                                    self.current_objects[obj_id]['id_park'] =self.next_park_id
                                    self.next_park_id +=1

                    else:
                            del self.trackers[obj_id]
                            if obj_id in self.current_objects:
                                del self.current_objects[obj_id]

                else:
                    del self.trackers[obj_id]
                    if obj_id in self.current_objects:
                            del self.current_objects[obj_id]

    def detect_object(self,stable_regions,frame):
        detected_boxes = []
        confidences = []

        blob = cv2.dnn.blobFromImage(stable_regions, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        detections = self.net.forward(self.output_layers)

        for out in detections:
            for detection in out:
                scores = detection[5:]
                class_id = scores.argmax()
                confidence = scores[class_id]

                # Only detect cars with sufficient confidence
                if (class_id  == 2)  and confidence > self.confidence_treshold:
                    center_x = int(detection[0] * self.width)
                    center_y = int(detection[1] * self.height)
                    w = int(detection[2] * self.width)
                    h = int(detection[3] * self.height)
                    x = center_x - w // 2
                    y = center_y - h // 2
                    bbox = (x, y, x + w, y + h)
                    detected_boxes.append(bbox)
                    confidences.append(float(confidence))

        # Apply NMS
        final_boxes = self.apply_nms(detected_boxes, confidences)

        # Check if detected boxes overlap with updated ones
        for detected in final_boxes:
            matched_id = None
            iou=0
            for obj_id, tracked_box in self.updated_boxes.items():
                iou = self.compute_iou(detected, tracked_box)
                if iou > self.iou_treshold or self.inside_of(detected,tracked_box) :
                    matched_id = obj_id
                    break

            if matched_id is None:  # No overlap with tracked boxes, so it's a new object
                self.current_objects[self.next_object_id] = {"bbox": detected,
                                                             "tags": '',
                                                             "id_park":0,
                                                             "long_term" :stable_regions.copy() ,
                                                             "last_seen": datetime.now(),
                                                             'exception' : 0,
                                                             'idle':0
                                                             }
                # Initialize a new tracker for this object
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (detected[0], detected[1], detected[2] - detected[0], detected[3] - detected[1]))
                self.trackers[self.next_object_id] = tracker
                self.next_object_id += 1

        
    def read_video(self,video_path,frame_skip = 128,refresh_tracker =16,tuning =  False,night = False):
        """
        Detect input video using the model
        """
        #Initialization
        self.cap = cv2.VideoCapture(video_path)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = 0
        self.current_objects = {}
        self.next_object_id = 1
        self.next_park_id = 1
        self.trackers = defaultdict(lambda: cv2.TrackerCSRT_create())
        self.last_seen = defaultdict(int)
        self.frame_skip = frame_skip
        self.refresh_tracker = refresh_tracker
        self.night= night

        ret, long_frame = self.cap.read()
        self.gray1 = cv2.cvtColor(long_frame, cv2.COLOR_BGR2GRAY)
        long_term_bg = np.float32(self.gray1)
        self.height, self.width, _ = long_frame.shape
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            stable_region,long,short,combine,dilatation = self.compute_foreground(frame)
            if self.frame_count % (self.refresh_tracker) == 0:
                self.update_tracker(stable_region,frame)
                
            if self.frame_count % self.frame_skip == 0:
                self.detect_object(stable_region,frame)
            
            for obj_id, tracked_box in self.updated_boxes.items():
                x, y, x2,y2 = [int(v) for v in tracked_box]
                duration = datetime.now() - self.current_objects[obj_id]['last_seen'] 
                time_stationary = duration.total_seconds()
                self.current_objects[obj_id]['idle'] = time_stationary
                if  time_stationary <= self.stop_duration:
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(frame, f"{time_stationary:.1f}", (x, y- 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    try:
                        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 1)
                        cv2.putText(frame, f"{self.current_objects[obj_id]['id_park'] }", (x, y- 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(frame, f"IDLE FOR {time_stationary:.1f}", (x, y- 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 2) 
                    except:
                        pass

            if tuning:
                c1 = np.hstack((long, short))
                c2 = np.hstack((combine, dilatation))
                c3 = np.vstack((c1, c2))
                c4 = np.hstack((frame, stable_region))
                c3 = cv2.resize(c3, (1000, 1000))
                c4 = cv2.resize(c4, (1000, 500))
                cv2.imshow('Tuning', c3)
                cv2.imshow('Car Detection', c4)
            else:
                cv2.imshow('Car Detection', frame)

            self.gray1 = self.gray2.copy()
            delay = int(100 / fps)
            
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

            self.frame_count += 1

        self.cap.release()
        cv2.destroyAllWindows()   



