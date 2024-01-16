from model import IdleTracker

if __name__ == "__main__":
    model = IdleTracker(
                 iou_treshold = 0.3, 
                 nms_threshold =0.5,
                 ssim_treshold = 0.4,
                 confidence_treshold = 0.6,
                 exception= 2,
                 move_treshold = 10,
                 stop_duration = 10,
                 alpha = 0.6,
                 short_term_treshold = 50,
                 long_term_treshold = 25,
                 kernel_noise = (1,1),
                 kernel_close=(10,10))

    model.initialize_yolo("config/yolov3.weights","config/yolov3.cfg")
    model.read_video("dataset/ISLab-06.mp4",128,64,True,False)
