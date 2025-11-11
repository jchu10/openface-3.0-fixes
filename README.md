### Custom Modifications Made to `openface-test` Package

We made several changes to fix device handling, video processing, and performance.

#### 1. Allow `--device cuda:0` (instead of only `cpu` or `cuda`)

The CLI originally restricted devices:

```python
@click.option('--device', '-d',
              type=click.Choice(['cuda', 'cpu']))
```

We changed it to:

```python
@click.option('--device', '-d', type=str, default='cpu')
```

This allows usage like:

```python
openface
detect - video
video.mp4 - -device
cuda: 0
```

We also updated LandmarkDetector to accept strings like "cuda:0":

```python
if device.startswith("cuda"):
    device_id = device
else:
    device_id = "cpu"
```

#### 2. Fixed image vs numpy frame issue in FaceDetector

`process_video()` passed numpy frames to `get_face()`, but `get_face()` assumed file paths.
We updated preprocess_image() to handle both:

```python
def preprocess_image(self, image_or_array, resize=1.0):
    if isinstance(image_or_array, str):
        img_raw = cv2.imread(image_or_array, cv2.IMREAD_COLOR)
    else:
        img_raw = image_or_array
```        

#### 3. Removed Landmark Detection to Improve Speed

(This is actually redundant since we don't use landmarks for anything downstream)

We skipped:

```python
landmarks = landmark_detector.detect_landmarks(frame, dets)
```

and instead set:

```python
landmark = None
```

This removes the slow alignment model entirely.

#### 4. Added Face Confidence Filtering (critical)

Without landmark filtering, low-confidence duplicate detections reappeared.
We re-added the confidence filter:

```python
dets = [det for det in dets if det[4] >= 0.5]
```

#### 5. Added detection confidence output

We modified the output to include detection confidence:

```python
conf = float(det[4])
results['confidence'].append(conf)
```