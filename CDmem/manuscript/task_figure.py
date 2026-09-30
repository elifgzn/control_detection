from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os

STIM_DIR = r"H:\PHD\control_detection\CDmem\chosen_stimuli_nolures"
img1_path = os.path.join(STIM_DIR, "pretzel_07s.jpg")
img2_path = os.path.join(STIM_DIR, "canoe_08s.jpg")
img3_path = os.path.join(STIM_DIR, "baseball_01b.jpg")

BG_COLOR = '#616161'
FG_COLOR = 'white'
TEXT_COLOR = 'black'

W = 3.5
H = 3.3
GAP = 0.45
Y_BOT = 0.75
Y_MID = Y_BOT + H / 2
TOP = Y_BOT + H

DX = 0.9

TARGET_SIZE_ENC = 65
TARGET_SIZE_REC = 55

plt.rcParams['svg.fonttype'] = 'none'     # keep text as <text> elements
plt.rcParams['font.family'] = 'Arial'     # use a font Figma has, instead of DejaVu Sans

def draw_screen(ax, x, title, time_text):
    rect = patches.Rectangle((x, Y_BOT), W, H, linewidth=2, edgecolor='black', facecolor=BG_COLOR)
    ax.add_patch(rect)
    ax.text(x + W / 2, TOP + 0.12, title, ha='center', va='bottom', fontsize=14, fontweight='bold', color=TEXT_COLOR)
    ax.text(x + W / 2, Y_BOT - 0.16, time_text, ha='center', va='top', fontsize=10, fontstyle='italic', color=TEXT_COLOR)

def draw_arrow(ax, x_screen_right):
    ax.annotate('', xy=(x_screen_right + GAP - 0.06, Y_MID), xytext=(x_screen_right + 0.06, Y_MID),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

def draw_real_image(ax, x, y, image_path, target_size_pt):
    try:
        img = mpimg.imread(image_path)
        max_dim = max(img.shape[0], img.shape[1])
        zoom = target_size_pt / max_dim
        imagebox = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        ax.add_artist(ab)
    except Exception as e:
        ax.text(x, y, "IMG", color='red', ha='center', va='center', fontsize=9)

def draw_shape(ax, x, y, shape_type):
    if shape_type == "square":
        patch = patches.Rectangle((x - 0.375, y - 0.375), 0.75, 0.75, color='black')
    else:
        patch = patches.Circle((x, y), 0.375, color='black')
    ax.add_patch(patch)

def motion_arrows(ax, cx):
    ax.annotate('', xy=(cx - 0.55, Y_MID + 0.6), xytext=(cx - 0.9, Y_MID),
                arrowprops=dict(arrowstyle='->', color='white', lw=1.75, ls='--', connectionstyle="arc3,rad=-0.5"))
    ax.annotate('', xy=(cx + 0.55, Y_MID - 0.6), xytext=(cx + 0.9, Y_MID),
                arrowprops=dict(arrowstyle='->', color='white', lw=1.75, ls='--', connectionstyle="arc3,rad=0.5"))

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 12))
fig.subplots_adjust(hspace=0.38, top=0.97, bottom=0.03)

for ax in axes:
    ax.axis('off')
    ax.set_xlim(-0.5, 15.4)
    ax.set_ylim(0, 5.15)

# PANEL A
ax0 = axes[0]
ax0.text(-0.5, 4.72, "A. Calibration Phase (Individual Thresholding)", fontsize=15, fontweight='bold', ha='left')
x = 0; cx = x + W / 2
draw_screen(ax0, x, "Fixation", "500-800 ms"); ax0.text(cx, Y_MID, "+", ha='center', va='center', fontsize=40, color=FG_COLOR); draw_arrow(ax0, x + W)
x += W + GAP; cx = x + W / 2
draw_screen(ax0, x, "Continuous Motion", "3000 ms"); draw_shape(ax0, cx - DX, Y_MID, "square"); draw_shape(ax0, cx + DX, Y_MID, "circle"); motion_arrows(ax0, cx); draw_arrow(ax0, x + W)
x += W + GAP; cx = x + W / 2
draw_screen(ax0, x, "Control Detection", "Max 3500 ms"); draw_shape(ax0, cx - DX, Y_MID + 0.45, "square"); draw_shape(ax0, cx + DX, Y_MID + 0.45, "circle")
ax0.text(cx, Y_MID - 0.35, "Which shape did you control?", ha='center', fontsize=10, color=FG_COLOR, fontweight='bold')
ax0.text(cx - DX, Y_MID - 0.85, "A", ha='center', fontsize=14, color=FG_COLOR, fontweight='bold'); ax0.text(cx + DX, Y_MID - 0.85, "S", ha='center', fontsize=14, color=FG_COLOR, fontweight='bold'); draw_arrow(ax0, x + W)
x += W + GAP; cx = x + W / 2
draw_screen(ax0, x, "Feedback", "800 ms"); ax0.text(cx, Y_MID, "Right", ha='center', va='center', fontsize=18, color=FG_COLOR, fontweight='bold')

# PANEL B
ax1 = axes[1]
ax1.text(-0.5, 4.72, "B. Encoding Phase (6 alternating 20-trial miniblocks)", fontsize=15, fontweight='bold', ha='left')
x = 0; cx = x + W / 2
draw_screen(ax1, x, "Fixation", "500-800 ms"); ax1.text(cx, Y_MID, "+", ha='center', va='center', fontsize=40, color=FG_COLOR); draw_arrow(ax1, x + W)
x += W + GAP; cx = x + W / 2
draw_screen(ax1, x, "Continuous Motion", "3000 ms")
draw_real_image(ax1, cx - DX, Y_MID, img1_path, TARGET_SIZE_ENC)
draw_real_image(ax1, cx + DX, Y_MID, img2_path, TARGET_SIZE_ENC)
motion_arrows(ax1, cx); draw_arrow(ax1, x + W)
x += W + GAP; cx = x + W / 2
draw_screen(ax1, x, "Control Detection", "Max 3500 ms")
draw_real_image(ax1, cx - DX, Y_MID + 0.45, img1_path, TARGET_SIZE_ENC)
draw_real_image(ax1, cx + DX, Y_MID + 0.45, img2_path, TARGET_SIZE_ENC)
ax1.text(cx, Y_MID - 0.55, "Which image did you control?", ha='center', fontsize=10, color=FG_COLOR, fontweight='bold')
ax1.text(cx - DX, Y_MID - 1.0, "A", ha='center', fontsize=14, color=FG_COLOR, fontweight='bold')
ax1.text(cx + DX, Y_MID - 1.0, "S", ha='center', fontsize=14, color=FG_COLOR, fontweight='bold'); draw_arrow(ax1, x + W)
x += W + GAP; cx = x + W / 2
draw_screen(ax1, x, "Agency Rating", "Self-paced")
ax1.text(cx, Y_MID + 0.55, "How much control did\nyou feel over the\nshape's movement?",
         ha='center', va='center', fontsize=12, color=FG_COLOR, fontweight='bold', linespacing=1.2)
ax1.text(cx, Y_MID - 0.22, "1    2    3    4    5    6    7", ha='center', va='center', fontsize=13, color=FG_COLOR)
ax1.text(cx - 1.15, Y_MID - 0.65, "Very weak", ha='center', va='center', fontsize=9, color=FG_COLOR)
ax1.text(cx, Y_MID - 0.65, "Moderate", ha='center', va='center', fontsize=9, color=FG_COLOR)
ax1.text(cx + 1.15, Y_MID - 0.65, "Very strong", ha='center', va='center', fontsize=9, color=FG_COLOR)

# PANEL C
ax2 = axes[2]
ax2.text(-0.5, 4.72, "C. Surprise Recognition Memory Test", fontsize=15, fontweight='bold', ha='left')
x = 0; cx = x + W / 2
draw_screen(ax2, x, "Fixation", "500-800 ms"); ax2.text(cx, Y_MID, "+", ha='center', va='center', fontsize=40, color=FG_COLOR); draw_arrow(ax2, x + W)
x += W + GAP; cx = x + W / 2
draw_screen(ax2, x, "Recognition", "Self-paced")
draw_real_image(ax2, cx, 3.18, img3_path, TARGET_SIZE_REC)
ax2.text(cx, 2.20, "Have you seen this image\nduring the experiment before?",
         ha='center', va='top', fontsize=10, color=FG_COLOR, fontweight='bold', linespacing=1.2)
ax2.text(cx - DX, 0.84, "Y\nYes", ha='center', va='bottom', fontsize=16, color=FG_COLOR, fontweight='bold', linespacing=1.1)
ax2.text(cx + DX, 0.84, "N\nNo", ha='center', va='bottom', fontsize=16, color=FG_COLOR, fontweight='bold', linespacing=1.1)

for ax in axes:
    for artist in ax.get_children():
        artist.set_clip_on(False)

out_path_png = Path(__file__).parent / "paradigm_complete_final.png"
out_path_svg = Path(__file__).parent / "paradigm_complete_final.svg"
plt.savefig(out_path_png, dpi=300, bbox_inches='tight')
plt.savefig(out_path_svg, bbox_inches='tight')
plt.close()

print(f"Figure saved: {out_path_png}")
print(f"Figure saved: {out_path_svg}")
