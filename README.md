# Idle Car Detection Using YOLOv3 and Background Subtraction Methods
This project is submitted for final stage of Datathon RISTEK University of Indonesia 2023

Member of this project:
- <a href = 'https://www.linkedin.com/in/yosefnw/'>Yosef Nuraga</a>
- <a href = 'https://www.linkedin.com/in/louis-widi-anandaputra-90008815a/'>Louis Widi</a>
- <a href = 'https://www.linkedin.com/in/limbodhiwijaya/'>Lim Bodhi</a>

```
python main.py
```

Configure the parameter
```python
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
```

Change the code snippet below `main.py` to the source of video in your directory.
```python
model.read_video("your-video-source",128,64,True,False)
```
