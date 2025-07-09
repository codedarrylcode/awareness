# Awareness

Awareness **scans** long recordings of football matches with custom-trained computer vision models by Roboflow, with lofty aims to democratize (_costly_) data to resource-scarce competitions and teams, in the world of football analytics.

💡 Key highlights

- Tracks + annotates players, ball, goalkeepers and referees
- Spatiotemporal analysis, in the lite form of Pitch Control
- (_WIP_) Visualization app to tweak tactical structure with Pitch Control effects
- (_WIP_) Events tracking consisting of classified actions (e.g. pass, shot, etc.)

## Install

Installed in a [Python 3.11.13](www.python.org) virtual environment.

```
pip install requirements.txt
```

## ⚽ Demos

Input | Output
:-: | :-:
<video src='data/train/08fd33_4.mp4' width=720> | <video src='videos/output/tracking_with_radar.mp4' width=720>

## Credits

- [Roboflow](https://github.com/roboflow/sports/tree/main)
