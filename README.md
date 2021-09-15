# Flow based Rip Current Detection

## Running the code 
- Compile OpenCV with CUDA option enabled
- Tested with Python=3.7 with Cuda=10.1
- Timelines: `python timelines.py --video ./videos/rip.mp4 --out . --alpha 0.65`
- Filtered Arrow Glyph and Filtered Color Map: `python filtered_arrows_color.py --video ./videos/rip.mp4 --out . --window 442 --mask ./videos/mask.png`

## Computing the alpha value
The adjustment factor alpha for the timelines method can be computed by <img src="https://render.githubusercontent.com/render/math?math=\alpha=d/(\delta \cdot f)">, where <img src="https://render.githubusercontent.com/render/math?math=d"> is the pixel-wise distance between the initial placement of the timeline and the shoreline, <img src="https://render.githubusercontent.com/render/math?math=\delta"> is the pixel-wise velocity of the incominng waves, and <img src="https://render.githubusercontent.com/render/math?math=f"> is the total number of frames.

For example, the rip.mp4 has 442 frames. If the timeline is placed around where waves start breaking, there are about 240 pixels until the shoreline. Also, 1 wave is about 200 frames and propagates roughly 240 pixels. Therfore the wave velocity is roughly 1.2, and the adjustment factor is <img src="https://render.githubusercontent.com/render/math?math=\alpha=240/(442 \cdot 1.2)=0.65">.
