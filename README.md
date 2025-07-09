# Awareness

Awareness **scans** long recordings of football matches with custom-trained computer vision models by Roboflow, with lofty aims to democratize (_costly_) data to resource-scarce competitions and teams, in the world of football analytics.

💡 Key highlights

- Tracks + annotates players, ball, goalkeepers and referees
- Spatiotemporal analysis, in the lite form of Pitch Control
- (_WIP_) Visualization app to tweak tactical structure with Pitch Control effects
- (_WIP_) Events tracking consisting of classified actions (e.g. pass, shot, etc.)

## 🖥️ Install

Installed in a [Python 3.11.13](www.python.org) virtual environment.

```
pip install requirements.txt
```

Model weights for each custom-trained YOLOv8 model can be downloaded from here:

1. [Ball detection model](https://drive.google.com/uc?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V)
2. [Player detection model](https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q)
3. [Pitch key points detection model](https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf)

## ⚽ Demos

https://github.com/user-attachments/assets/a3583940-6fc0-481b-91f8-288944320e04

https://github.com/user-attachments/assets/82b3282f-4f04-46b7-bbbd-4b95f2caacda

## Credits

- [Roboflow](https://github.com/roboflow/sports/tree/main)
