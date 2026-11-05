"""Render a demo video from the network's real output."""
import subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio
code = '''@dataclass
class Greeter:
    """Say hello, politely."""
    name: str = "world"

    def greet(self, times=3):
        # f-strings nest, and the network keeps track
        for i in range(times):
            print(f"hello {self.name!r} #{i + 1:>2}")
        return len(self.name) * 0x10 + 1.5e3

async def main():
    g = Greeter("827 weights")
    await g.greet()
    raise SystemExit(None) from None
'''
Path("demo_in.py").write_text(code)
lab = subprocess.run(["./pylexnet", "--classes", "demo_in.py"], capture_output=True).stdout
assert len(lab) == len(code.encode())
PAL = {0:(205,214,244),1:(203,166,247),2:(137,220,235),3:(166,227,161),4:(127,132,156),5:(250,179,135),6:(137,180,250),7:(249,226,175),8:(245,194,231)}
BG=(30,30,46); W,H=1280,720; S=22; font=ImageFont.truetype(sys.argv[1],S); big=ImageFont.truetype(sys.argv[1],26)
cw=font.getlength("M"); lh=int(S*1.5)
def frame(n, caption):
    im=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(im); x0,y0=40,92
    d.text((40,28),"pylexnet: a Python highlighter that is a neural network",font=big,fill=(205,214,244))
    x=x0;y=y0
    for i,ch in enumerate(code[:n]):
        if ch=="\n": x=x0;y+=lh;continue
        d.text((x,y),ch,font=font,fill=PAL[lab[i]]); x+=cw
    d.rectangle([x,y+2,x+cw-2,y+S+4],fill=(205,214,244))
    d.text((40,H-46),caption,font=font,fill=(127,132,156))
    return np.asarray(im)
w=imageio.get_writer("demo.mp4",fps=30,codec="libx264",quality=8,macro_block_size=8)
cap=""
for n in range(0,len(code)+1,2): w.append_data(frame(n,cap))
last=frame(len(code),cap)
for _ in range(90): w.append_data(last)
w.close(); print("frames ok", Path("demo.mp4").stat().st_size)
